#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <tuple>

namespace {

constexpr int kWarpSize = 32;
constexpr int kWarpsPerBlock = 4;
constexpr int kThreadsPerBlock = kWarpSize * kWarpsPerBlock;
constexpr int kTileSize = 16;
constexpr int kValuesPerLane = 8;  // ceil(256 / warpSize)
constexpr int kMaxGridBlocks = 65535;

__device__ __forceinline__ float warp_sum(float value) {
#pragma unroll
    for (int offset = kWarpSize / 2; offset > 0; offset /= 2) {
        value += __shfl_down_sync(0xffffffffu, value, offset);
    }
    return value;
}

template <typename scalar_t>
__global__ void flash_attention_forward_kernel(
    const scalar_t* __restrict__ q,
    const scalar_t* __restrict__ k,
    const scalar_t* __restrict__ v,
    scalar_t* __restrict__ out,
    float* __restrict__ lse,
    int64_t query_length,
    int64_t key_length,
    int head_dim,
    int64_t query_groups_per_head,
    int64_t total_query_groups,
    bool causal,
    float scale) {
    extern __shared__ float shared[];
    float* shared_k = shared;
    float* shared_v = shared + kTileSize * head_dim;

    const int lane = threadIdx.x % kWarpSize;
    const int warp = threadIdx.x / kWarpSize;

    for (int64_t group = blockIdx.x; group < total_query_groups;
         group += gridDim.x) {
        const int64_t batch_head = group / query_groups_per_head;
        const int64_t query_group = group % query_groups_per_head;
        const int64_t query_index =
            query_group * kWarpsPerBlock + static_cast<int64_t>(warp);
        const bool query_valid = query_index < query_length;
        const int64_t query_row = batch_head * query_length + query_index;

        float query_values[kValuesPerLane];
        float output_values[kValuesPerLane];
#pragma unroll
        for (int slot = 0; slot < kValuesPerLane; ++slot) {
            const int dim = lane + slot * kWarpSize;
            query_values[slot] =
                (query_valid && dim < head_dim)
                ? static_cast<float>(q[query_row * head_dim + dim])
                : 0.0f;
            output_values[slot] = 0.0f;
        }

        // Only lane zero owns the scalar online-softmax state. The per-key
        // rescaling factors are broadcast to the rest of the warp.
        float running_max = -CUDART_INF_F;
        float running_sum = 0.0f;

        for (int64_t key_start = 0; key_start < key_length;
             key_start += kTileSize) {
            const int tile_count = static_cast<int>(
                key_length - key_start < kTileSize
                    ? key_length - key_start
                    : kTileSize);
            const int tile_elements = tile_count * head_dim;

            for (int element = threadIdx.x; element < tile_elements;
                 element += blockDim.x) {
                const int key_in_tile = element / head_dim;
                const int dim = element - key_in_tile * head_dim;
                const int64_t source =
                    ((batch_head * key_length + key_start + key_in_tile) *
                     head_dim) +
                    dim;
                shared_k[element] = static_cast<float>(k[source]);
                shared_v[element] = static_cast<float>(v[source]);
            }
            __syncthreads();

            if (query_valid) {
                for (int key_in_tile = 0; key_in_tile < tile_count;
                     ++key_in_tile) {
                    const int64_t key_index = key_start + key_in_tile;
                    if (causal && key_index > query_index) {
                        continue;
                    }

                    float dot = 0.0f;
#pragma unroll
                    for (int slot = 0; slot < kValuesPerLane; ++slot) {
                        const int dim = lane + slot * kWarpSize;
                        if (dim < head_dim) {
                            dot += query_values[slot] *
                                   shared_k[key_in_tile * head_dim + dim];
                        }
                    }
                    dot = warp_sum(dot);

                    float old_weight = 0.0f;
                    float new_weight = 0.0f;
                    if (lane == 0) {
                        const float score = dot * scale;
                        if (running_sum == 0.0f) {
                            // Treat the first unmasked element specially to
                            // avoid evaluating -inf - -inf.
                            running_max = score;
                            running_sum = 1.0f;
                            old_weight = 0.0f;
                            new_weight = 1.0f;
                        } else {
                            const float next_max = fmaxf(running_max, score);
                            old_weight = expf(running_max - next_max);
                            new_weight = expf(score - next_max);
                            running_sum =
                                old_weight * running_sum + new_weight;
                            running_max = next_max;
                        }
                    }
                    old_weight =
                        __shfl_sync(0xffffffffu, old_weight, 0);
                    new_weight =
                        __shfl_sync(0xffffffffu, new_weight, 0);

#pragma unroll
                    for (int slot = 0; slot < kValuesPerLane; ++slot) {
                        const int dim = lane + slot * kWarpSize;
                        if (dim < head_dim) {
                            output_values[slot] =
                                old_weight * output_values[slot] +
                                new_weight *
                                    shared_v[key_in_tile * head_dim + dim];
                        }
                    }
                }
            }
            // No warp may leave the tile loop early: all four warps cooperate
            // when overwriting shared memory for the next K/V tile.
            __syncthreads();
        }

        if (query_valid) {
            float inverse_sum =
                lane == 0 ? 1.0f / running_sum : 0.0f;
            inverse_sum = __shfl_sync(0xffffffffu, inverse_sum, 0);
#pragma unroll
            for (int slot = 0; slot < kValuesPerLane; ++slot) {
                const int dim = lane + slot * kWarpSize;
                if (dim < head_dim) {
                    out[query_row * head_dim + dim] =
                        static_cast<scalar_t>(
                            output_values[slot] * inverse_sum);
                }
            }
            if (lane == 0) {
                lse[query_row] = running_max + logf(running_sum);
            }
        }
    }
}

template <typename scalar_t>
__global__ void attention_delta_kernel(
    const scalar_t* __restrict__ dout,
    const scalar_t* __restrict__ out,
    float* __restrict__ delta,
    int64_t total_query_rows,
    int head_dim,
    int64_t total_delta_groups) {
    const int lane = threadIdx.x % kWarpSize;
    const int warp = threadIdx.x / kWarpSize;

    for (int64_t group = blockIdx.x; group < total_delta_groups;
         group += gridDim.x) {
        const int64_t query_row =
            group * kWarpsPerBlock + static_cast<int64_t>(warp);
        const bool query_valid = query_row < total_query_rows;
        float value = 0.0f;
        if (query_valid) {
            const int64_t row_offset = query_row * head_dim;
            for (int dim = lane; dim < head_dim; dim += kWarpSize) {
                value += static_cast<float>(dout[row_offset + dim]) *
                         static_cast<float>(out[row_offset + dim]);
            }
        }
        value = warp_sum(value);
        if (query_valid && lane == 0) {
            delta[query_row] = value;
        }
    }
}

template <typename scalar_t>
__global__ void flash_attention_dq_kernel(
    const scalar_t* __restrict__ dout,
    const scalar_t* __restrict__ q,
    const scalar_t* __restrict__ k,
    const scalar_t* __restrict__ v,
    const float* __restrict__ lse,
    const float* __restrict__ delta,
    scalar_t* __restrict__ dq,
    int64_t query_length,
    int64_t key_length,
    int head_dim,
    int64_t query_groups_per_head,
    int64_t total_query_groups,
    bool causal,
    float scale) {
    extern __shared__ float shared[];
    float* shared_k = shared;
    float* shared_v = shared + kTileSize * head_dim;

    const int lane = threadIdx.x % kWarpSize;
    const int warp = threadIdx.x / kWarpSize;

    for (int64_t group = blockIdx.x; group < total_query_groups;
         group += gridDim.x) {
        const int64_t batch_head = group / query_groups_per_head;
        const int64_t query_group = group % query_groups_per_head;
        const int64_t query_index =
            query_group * kWarpsPerBlock + static_cast<int64_t>(warp);
        const bool query_valid = query_index < query_length;
        const int64_t query_row = batch_head * query_length + query_index;

        float query_values[kValuesPerLane];
        float dout_values[kValuesPerLane];
        float dq_values[kValuesPerLane];
#pragma unroll
        for (int slot = 0; slot < kValuesPerLane; ++slot) {
            const int dim = lane + slot * kWarpSize;
            query_values[slot] =
                (query_valid && dim < head_dim)
                ? static_cast<float>(q[query_row * head_dim + dim])
                : 0.0f;
            dout_values[slot] =
                (query_valid && dim < head_dim)
                ? static_cast<float>(dout[query_row * head_dim + dim])
                : 0.0f;
            dq_values[slot] = 0.0f;
        }

        float row_lse =
            (query_valid && lane == 0) ? lse[query_row] : 0.0f;
        float row_delta =
            (query_valid && lane == 0) ? delta[query_row] : 0.0f;
        row_lse = __shfl_sync(0xffffffffu, row_lse, 0);
        row_delta = __shfl_sync(0xffffffffu, row_delta, 0);

        for (int64_t key_start = 0; key_start < key_length;
             key_start += kTileSize) {
            const int tile_count = static_cast<int>(
                key_length - key_start < kTileSize
                    ? key_length - key_start
                    : kTileSize);
            const int tile_elements = tile_count * head_dim;
            for (int element = threadIdx.x; element < tile_elements;
                 element += blockDim.x) {
                const int key_in_tile = element / head_dim;
                const int dim = element - key_in_tile * head_dim;
                const int64_t source =
                    ((batch_head * key_length + key_start + key_in_tile) *
                     head_dim) +
                    dim;
                shared_k[element] = static_cast<float>(k[source]);
                shared_v[element] = static_cast<float>(v[source]);
            }
            __syncthreads();

            if (query_valid) {
                for (int key_in_tile = 0; key_in_tile < tile_count;
                     ++key_in_tile) {
                    const int64_t key_index = key_start + key_in_tile;
                    if (causal && key_index > query_index) {
                        continue;
                    }

                    float score = 0.0f;
                    float d_probability = 0.0f;
#pragma unroll
                    for (int slot = 0; slot < kValuesPerLane; ++slot) {
                        const int dim = lane + slot * kWarpSize;
                        if (dim < head_dim) {
                            score += query_values[slot] *
                                     shared_k[key_in_tile * head_dim + dim];
                            d_probability +=
                                dout_values[slot] *
                                shared_v[key_in_tile * head_dim + dim];
                        }
                    }
                    score = warp_sum(score);
                    d_probability = warp_sum(d_probability);

                    float d_score = 0.0f;
                    if (lane == 0) {
                        const float probability =
                            expf(score * scale - row_lse);
                        d_score = probability *
                                  (d_probability - row_delta);
                    }
                    d_score = __shfl_sync(0xffffffffu, d_score, 0);
                    const float scaled_d_score = scale * d_score;

#pragma unroll
                    for (int slot = 0; slot < kValuesPerLane; ++slot) {
                        const int dim = lane + slot * kWarpSize;
                        if (dim < head_dim) {
                            dq_values[slot] +=
                                scaled_d_score *
                                shared_k[key_in_tile * head_dim + dim];
                        }
                    }
                }
            }
            __syncthreads();
        }

        if (query_valid) {
#pragma unroll
            for (int slot = 0; slot < kValuesPerLane; ++slot) {
                const int dim = lane + slot * kWarpSize;
                if (dim < head_dim) {
                    dq[query_row * head_dim + dim] =
                        static_cast<scalar_t>(dq_values[slot]);
                }
            }
        }
    }
}

template <typename scalar_t>
__global__ void flash_attention_dkdv_kernel(
    const scalar_t* __restrict__ dout,
    const scalar_t* __restrict__ q,
    const scalar_t* __restrict__ k,
    const scalar_t* __restrict__ v,
    const float* __restrict__ lse,
    const float* __restrict__ delta,
    scalar_t* __restrict__ dk,
    scalar_t* __restrict__ dv,
    int64_t query_length,
    int64_t key_length,
    int head_dim,
    int64_t key_groups_per_head,
    int64_t total_key_groups,
    bool causal,
    float scale) {
    extern __shared__ float shared[];
    float* shared_q = shared;
    float* shared_dout = shared + kTileSize * head_dim;

    const int lane = threadIdx.x % kWarpSize;
    const int warp = threadIdx.x / kWarpSize;

    for (int64_t group = blockIdx.x; group < total_key_groups;
         group += gridDim.x) {
        const int64_t batch_head = group / key_groups_per_head;
        const int64_t key_group = group % key_groups_per_head;
        const int64_t key_index =
            key_group * kWarpsPerBlock + static_cast<int64_t>(warp);
        const bool key_valid = key_index < key_length;
        const int64_t key_row = batch_head * key_length + key_index;

        float key_values[kValuesPerLane];
        float value_values[kValuesPerLane];
        float dk_values[kValuesPerLane];
        float dv_values[kValuesPerLane];
#pragma unroll
        for (int slot = 0; slot < kValuesPerLane; ++slot) {
            const int dim = lane + slot * kWarpSize;
            key_values[slot] =
                (key_valid && dim < head_dim)
                ? static_cast<float>(k[key_row * head_dim + dim])
                : 0.0f;
            value_values[slot] =
                (key_valid && dim < head_dim)
                ? static_cast<float>(v[key_row * head_dim + dim])
                : 0.0f;
            dk_values[slot] = 0.0f;
            dv_values[slot] = 0.0f;
        }

        for (int64_t query_start = 0; query_start < query_length;
             query_start += kTileSize) {
            const int tile_count = static_cast<int>(
                query_length - query_start < kTileSize
                    ? query_length - query_start
                    : kTileSize);
            const int tile_elements = tile_count * head_dim;
            for (int element = threadIdx.x; element < tile_elements;
                 element += blockDim.x) {
                const int query_in_tile = element / head_dim;
                const int dim = element - query_in_tile * head_dim;
                const int64_t source =
                    ((batch_head * query_length + query_start +
                      query_in_tile) *
                     head_dim) +
                    dim;
                shared_q[element] = static_cast<float>(q[source]);
                shared_dout[element] =
                    static_cast<float>(dout[source]);
            }
            __syncthreads();

            if (key_valid) {
                for (int query_in_tile = 0; query_in_tile < tile_count;
                     ++query_in_tile) {
                    const int64_t query_index =
                        query_start + query_in_tile;
                    if (causal && key_index > query_index) {
                        continue;
                    }

                    float score = 0.0f;
                    float d_probability = 0.0f;
#pragma unroll
                    for (int slot = 0; slot < kValuesPerLane; ++slot) {
                        const int dim = lane + slot * kWarpSize;
                        if (dim < head_dim) {
                            score += key_values[slot] *
                                     shared_q[query_in_tile * head_dim + dim];
                            d_probability +=
                                value_values[slot] *
                                shared_dout[query_in_tile * head_dim + dim];
                        }
                    }
                    score = warp_sum(score);
                    d_probability = warp_sum(d_probability);

                    const int64_t query_row =
                        batch_head * query_length + query_index;
                    float probability = 0.0f;
                    float d_score = 0.0f;
                    if (lane == 0) {
                        probability =
                            expf(score * scale - lse[query_row]);
                        d_score = probability *
                                  (d_probability - delta[query_row]);
                    }
                    probability =
                        __shfl_sync(0xffffffffu, probability, 0);
                    d_score = __shfl_sync(0xffffffffu, d_score, 0);
                    const float scaled_d_score = scale * d_score;

#pragma unroll
                    for (int slot = 0; slot < kValuesPerLane; ++slot) {
                        const int dim = lane + slot * kWarpSize;
                        if (dim < head_dim) {
                            dk_values[slot] +=
                                scaled_d_score *
                                shared_q[query_in_tile * head_dim + dim];
                            dv_values[slot] +=
                                probability *
                                shared_dout[query_in_tile * head_dim + dim];
                        }
                    }
                }
            }
            __syncthreads();
        }

        if (key_valid) {
#pragma unroll
            for (int slot = 0; slot < kValuesPerLane; ++slot) {
                const int dim = lane + slot * kWarpSize;
                if (dim < head_dim) {
                    dk[key_row * head_dim + dim] =
                        static_cast<scalar_t>(dk_values[slot]);
                    dv[key_row * head_dim + dim] =
                        static_cast<scalar_t>(dv_values[slot]);
                }
            }
        }
    }
}

int grid_size_for(int64_t groups) {
    return static_cast<int>(
        std::min<int64_t>(groups, static_cast<int64_t>(kMaxGridBlocks)));
}

}  // namespace

std::tuple<torch::Tensor, torch::Tensor> flash_attention_cuda_forward(
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& v,
    bool causal,
    double scale) {
    const c10::cuda::CUDAGuard device_guard(q.device());

    auto out = torch::empty_like(q);
    auto lse = torch::empty(
        {q.size(0), q.size(1), q.size(2)},
        q.options().dtype(torch::kFloat32));

    const int64_t batch_heads = q.size(0) * q.size(1);
    const int64_t query_length = q.size(2);
    const int64_t key_length = k.size(2);
    const int head_dim = static_cast<int>(q.size(3));
    const int64_t query_groups_per_head =
        (query_length - 1) / kWarpsPerBlock + 1;
    const int64_t total_query_groups =
        batch_heads * query_groups_per_head;
    const int blocks = grid_size_for(total_query_groups);
    const size_t shared_bytes =
        2ULL * kTileSize * head_dim * sizeof(float);
    const cudaStream_t stream =
        at::cuda::getCurrentCUDAStream(q.get_device());

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        q.scalar_type(),
        "flash_attention_forward_cuda",
        [&] {
            flash_attention_forward_kernel<scalar_t>
                <<<blocks, kThreadsPerBlock, shared_bytes, stream>>>(
                    q.data_ptr<scalar_t>(),
                    k.data_ptr<scalar_t>(),
                    v.data_ptr<scalar_t>(),
                    out.data_ptr<scalar_t>(),
                    lse.data_ptr<float>(),
                    query_length,
                    key_length,
                    head_dim,
                    query_groups_per_head,
                    total_query_groups,
                    causal,
                    static_cast<float>(scale));
        });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return std::make_tuple(out, lse);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
flash_attention_cuda_backward(
    const torch::Tensor& dout,
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& v,
    const torch::Tensor& out,
    const torch::Tensor& lse,
    bool causal,
    double scale) {
    const c10::cuda::CUDAGuard device_guard(q.device());

    auto dq = torch::empty_like(q);
    auto dk = torch::empty_like(k);
    auto dv = torch::empty_like(v);
    auto delta = torch::empty(
        {q.size(0), q.size(1), q.size(2)},
        q.options().dtype(torch::kFloat32));

    const int64_t batch_heads = q.size(0) * q.size(1);
    const int64_t query_length = q.size(2);
    const int64_t key_length = k.size(2);
    const int head_dim = static_cast<int>(q.size(3));
    const int64_t total_query_rows = batch_heads * query_length;
    const int64_t delta_groups =
        (total_query_rows - 1) / kWarpsPerBlock + 1;
    const int64_t query_groups_per_head =
        (query_length - 1) / kWarpsPerBlock + 1;
    const int64_t key_groups_per_head =
        (key_length - 1) / kWarpsPerBlock + 1;
    const int64_t total_query_groups =
        batch_heads * query_groups_per_head;
    const int64_t total_key_groups =
        batch_heads * key_groups_per_head;

    const int delta_blocks = grid_size_for(delta_groups);
    const int dq_blocks = grid_size_for(total_query_groups);
    const int dkdv_blocks = grid_size_for(total_key_groups);
    const size_t shared_bytes =
        2ULL * kTileSize * head_dim * sizeof(float);
    const cudaStream_t stream =
        at::cuda::getCurrentCUDAStream(q.get_device());

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        q.scalar_type(),
        "flash_attention_backward_cuda",
        [&] {
            attention_delta_kernel<scalar_t>
                <<<delta_blocks, kThreadsPerBlock, 0, stream>>>(
                    dout.data_ptr<scalar_t>(),
                    out.data_ptr<scalar_t>(),
                    delta.data_ptr<float>(),
                    total_query_rows,
                    head_dim,
                    delta_groups);
            C10_CUDA_KERNEL_LAUNCH_CHECK();

            flash_attention_dq_kernel<scalar_t>
                <<<dq_blocks, kThreadsPerBlock, shared_bytes, stream>>>(
                    dout.data_ptr<scalar_t>(),
                    q.data_ptr<scalar_t>(),
                    k.data_ptr<scalar_t>(),
                    v.data_ptr<scalar_t>(),
                    lse.data_ptr<float>(),
                    delta.data_ptr<float>(),
                    dq.data_ptr<scalar_t>(),
                    query_length,
                    key_length,
                    head_dim,
                    query_groups_per_head,
                    total_query_groups,
                    causal,
                    static_cast<float>(scale));
            C10_CUDA_KERNEL_LAUNCH_CHECK();

            flash_attention_dkdv_kernel<scalar_t>
                <<<dkdv_blocks, kThreadsPerBlock, shared_bytes, stream>>>(
                    dout.data_ptr<scalar_t>(),
                    q.data_ptr<scalar_t>(),
                    k.data_ptr<scalar_t>(),
                    v.data_ptr<scalar_t>(),
                    lse.data_ptr<float>(),
                    delta.data_ptr<float>(),
                    dk.data_ptr<scalar_t>(),
                    dv.data_ptr<scalar_t>(),
                    query_length,
                    key_length,
                    head_dim,
                    key_groups_per_head,
                    total_key_groups,
                    causal,
                    static_cast<float>(scale));
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        });

    return std::make_tuple(dq, dk, dv);
}
