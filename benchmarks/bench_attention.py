"""Reproducible forward and backward benchmarks for attention implementations.

The harness deliberately separates correctness validation from timing, uses CUDA
events on GPUs, reports distributions rather than a single mean, and writes a
metadata-rich JSON artifact alongside a flat CSV table.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import shlex
import shutil
import statistics
import subprocess
import sys
import time
from contextlib import contextmanager, nullcontext
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator, Sequence

try:
    import torch
    from torch import Tensor
except ModuleNotFoundError as exc:
    torch = None  # type: ignore[assignment]
    Tensor = Any  # type: ignore[misc,assignment]
    _TORCH_IMPORT_ERROR: ModuleNotFoundError | None = exc
else:
    _TORCH_IMPORT_ERROR = None

try:
    from flash_attention_cuda import (
        extension_available,
        extension_error,
        flash_attention,
        manual_attention,
    )
except ModuleNotFoundError as exc:
    extension_available = None  # type: ignore[assignment]
    extension_error = None  # type: ignore[assignment]
    flash_attention = None  # type: ignore[assignment]
    manual_attention = None  # type: ignore[assignment]
    _FLASH_ATTENTION_IMPORT_ERROR: ModuleNotFoundError | None = exc
else:
    _FLASH_ATTENTION_IMPORT_ERROR = None


DTYPES: dict[str, torch.dtype] = {
    "float16": getattr(torch, "float16", "float16"),
    "bfloat16": getattr(torch, "bfloat16", "bfloat16"),
    "float32": getattr(torch, "float32", "float32"),
}
METHODS = ("cuda", "sdpa", "reference", "auto")
MEBIBYTE = 1024**2


@dataclass(frozen=True)
class ShapeConfig:
    batch: int
    heads: int
    query_length: int
    key_length: int
    head_dim: int
    causal: bool


def _require_torch() -> None:
    if torch is None:
        raise RuntimeError(
            "benchmarks.bench_attention requires PyTorch to run. "
            "Install a compatible torch build before benchmarking."
        ) from _TORCH_IMPORT_ERROR


def _require_runtime_dependencies() -> None:
    _require_torch()
    if flash_attention is None or manual_attention is None:
        raise RuntimeError(
            "benchmarks.bench_attention could not import the flash_attention_cuda "
            "package. Install project dependencies before benchmarking."
        ) from _FLASH_ATTENTION_IMPORT_ERROR


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be non-negative")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark custom CUDA, PyTorch SDPA, and explicit reference attention."
    )
    parser.add_argument(
        "--device", choices=("auto", "cpu", "cuda"), default="auto"
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=METHODS,
        default=None,
        help="Defaults to cuda/sdpa/reference on GPU and sdpa/reference on CPU.",
    )
    parser.add_argument(
        "--dtypes",
        nargs="+",
        choices=tuple(DTYPES),
        default=None,
        help="Defaults to float16 on GPU and float32 on CPU.",
    )
    parser.add_argument("--batch", type=_positive_int, default=2)
    parser.add_argument("--heads", type=_positive_int, default=8)
    parser.add_argument(
        "--query-lengths",
        "--seq-lens",
        dest="query_lengths",
        nargs="+",
        type=_positive_int,
        default=(128, 512, 1024),
    )
    parser.add_argument(
        "--key-lengths",
        nargs="+",
        type=_positive_int,
        default=None,
        help="If omitted, each key length equals its query length.",
    )
    parser.add_argument(
        "--head-dims", nargs="+", type=_positive_int, default=(64,)
    )
    parser.add_argument(
        "--causal",
        nargs="?",
        const="true",
        default="false",
        choices=("false", "true", "both"),
        help=(
            "Select non-causal attention ('false'), causal attention ('true'), "
            "or benchmark both modes ('both'). Passing --causal with no value "
            "is treated as 'true' for CLI compatibility."
        ),
    )
    parser.add_argument(
        "--mode", choices=("forward", "forward-backward"), default="forward"
    )
    parser.add_argument("--warmup", type=_nonnegative_int, default=10)
    parser.add_argument("--iterations", type=_positive_int, default=50)
    parser.add_argument("--repeats", type=_positive_int, default=7)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument(
        "--sdpa-backend",
        choices=("auto", "math", "flash", "efficient", "cudnn"),
        default="auto",
        help="Force a PyTorch SDPA backend when method=sdpa; auto records dynamic dispatch.",
    )
    parser.add_argument(
        "--validation",
        choices=("reference", "none"),
        default="reference",
        help="Compute numerical error against explicit attention before timing.",
    )
    parser.add_argument(
        "--max-validation-elements",
        type=_nonnegative_int,
        default=67_108_864,
        help=(
            "Skip the explicit oracle when B*H*Nq*Nk exceeds this value; "
            "zero disables the limit."
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--tag", default="", help="Optional filename-safe run label.")
    parser.add_argument(
        "--require-cuda",
        action="store_true",
        help="Fail unless a GPU and the compiled custom extension are available and measured.",
    )
    parser.add_argument(
        "--fail-on-error",
        action="store_true",
        help="Return nonzero if any configuration reports error, OOM, or unavailable.",
    )
    return parser


def _resolve_device(parser: argparse.ArgumentParser, requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        parser.error("--device cuda was requested, but torch.cuda.is_available() is false")
    return torch.device(requested)


def _validate_cuda_requirement(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
    device: torch.device,
) -> None:
    if not args.require_cuda:
        return
    if device.type != "cuda":
        parser.error("--require-cuda requires --device cuda (or auto with a visible GPU)")
    if not extension_available():
        parser.error(
            "--require-cuda was set, but the extension is unavailable: "
            f"{extension_error() or 'unknown import error'}"
        )
    if "cuda" not in args.methods:
        parser.error("--require-cuda requires 'cuda' in --methods")


def _percentile(values: Sequence[float], percentile: float) -> float:
    if not values:
        raise ValueError("cannot compute a percentile of an empty sequence")
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _clear_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        _sync(device)
        torch.cuda.empty_cache()


def _is_out_of_memory(exc: BaseException) -> bool:
    cuda_oom = getattr(torch.cuda, "OutOfMemoryError", RuntimeError)
    return isinstance(exc, (MemoryError, cuda_oom)) or "out of memory" in str(exc).lower()


def _safe_command(command: Sequence[str], cwd: Path | None = None) -> str | None:
    try:
        completed = subprocess.run(
            command,
            cwd=cwd,
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    output = (completed.stdout or completed.stderr).strip()
    return output or None


def _git_metadata(repository: Path) -> dict[str, Any]:
    safe_directory = f"safe.directory={repository.resolve()}"
    sha = _safe_command(("git", "-c", safe_directory, "rev-parse", "HEAD"), repository)
    status = _safe_command(
        ("git", "-c", safe_directory, "status", "--porcelain"), repository
    )
    return {
        "commit": sha,
        "dirty": bool(status),
        "status_porcelain": status,
    }


def _cuda_metadata() -> dict[str, Any]:
    if not torch.cuda.is_available():
        return {"available": False, "devices": []}

    devices = []
    for index in range(torch.cuda.device_count()):
        properties = torch.cuda.get_device_properties(index)
        devices.append(
            {
                "index": index,
                "name": properties.name,
                "compute_capability": list(torch.cuda.get_device_capability(index)),
                "total_memory_bytes": properties.total_memory,
                "multiprocessor_count": properties.multi_processor_count,
            }
        )
    return {
        "available": True,
        "device_count": torch.cuda.device_count(),
        "current_device": torch.cuda.current_device(),
        "devices": devices,
        "cudnn_version": torch.backends.cudnn.version(),
        "nvidia_smi": _safe_command(
            (
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total,clocks.sm,clocks.mem",
                "--format=csv,noheader",
            )
        ),
    }


def _environment_metadata(
    args: argparse.Namespace, device: torch.device, run_id: str
) -> dict[str, Any]:
    repository = Path(__file__).resolve().parents[1]
    nvcc = shutil.which("nvcc")
    return {
        "run_id": run_id,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "command": shlex.join(sys.argv),
        "repository": str(repository),
        "git": _git_metadata(repository),
        "platform": {
            "python": sys.version,
            "python_executable": sys.executable,
            "system": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "hostname": platform.node(),
        },
        "software": {
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "nvcc_path": nvcc,
            "nvcc_version": _safe_command((nvcc, "--version")) if nvcc else None,
            "extension_available": extension_available(),
            "extension_error": str(extension_error()) if extension_error() else None,
        },
        "cuda": _cuda_metadata(),
        "numerics": {
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
            "cuda_matmul_allow_tf32": getattr(
                getattr(torch.backends, "cuda", None), "matmul", None
            ).allow_tf32
            if hasattr(getattr(torch.backends, "cuda", None), "matmul")
            else None,
            "cudnn_allow_tf32": getattr(torch.backends.cudnn, "allow_tf32", None),
        },
        "selected_environment": {
            name: os.environ.get(name)
            for name in ("CUDA_VISIBLE_DEVICES", "TORCH_CUDA_ARCH_LIST")
        },
        "benchmark": {
            **{
                key: str(value) if isinstance(value, Path) else value
                for key, value in vars(args).items()
            },
            "resolved_device": str(device),
            "flop_model": (
                "dense-equivalent matmul FLOPs only: 4*B*H*Nq*Nk*D forward, "
                "12*B*H*Nq*Nk*D forward-backward; softmax is excluded"
            ),
            "tokens_definition": "B*Nq query tokens (heads are not counted as tokens)",
            "memory_definition": (
                "max allocated bytes during one operation minus allocated bytes with "
                "Q/K/V and grad_output resident"
            ),
        },
    }


@contextmanager
def _sdpa_backend_context(name: str) -> Iterator[None]:
    if name == "auto":
        yield
        return

    try:
        from torch.nn.attention import SDPBackend, sdpa_kernel
    except ImportError as exc:  # pragma: no cover - only for old unsupported PyTorch.
        raise RuntimeError(
            "forcing an SDPA backend requires torch.nn.attention.sdpa_kernel"
        ) from exc

    enum_names = {
        "math": "MATH",
        "flash": "FLASH_ATTENTION",
        "efficient": "EFFICIENT_ATTENTION",
        "cudnn": "CUDNN_ATTENTION",
    }
    enum_name = enum_names[name]
    backend = getattr(SDPBackend, enum_name, None)
    if backend is None:
        raise RuntimeError(f"this PyTorch build does not expose SDPBackend.{enum_name}")
    with sdpa_kernel(backends=[backend]):
        yield


def _method_context(method: str, sdpa_backend: str):
    return _sdpa_backend_context(sdpa_backend if method == "sdpa" else "auto")


def _inference_context(mode: str):
    return torch.inference_mode() if mode == "forward" else nullcontext()


def _invoke_forward(
    method: str,
    q: Tensor,
    k: Tensor,
    v: Tensor,
    *,
    causal: bool,
) -> Tensor:
    return flash_attention(q, k, v, causal=causal, implementation=method)


def _make_operation(
    method: str,
    q: Tensor,
    k: Tensor,
    v: Tensor,
    grad_output: Tensor | None,
    *,
    causal: bool,
    mode: str,
) -> Callable[[], object]:
    def operation() -> object:
        output = _invoke_forward(method, q, k, v, causal=causal)
        if mode == "forward":
            return output
        if grad_output is None:  # Defensive; the CLI always creates it in this mode.
            raise RuntimeError("forward-backward mode requires grad_output")
        gradients = torch.autograd.grad(
            output,
            (q, k, v),
            grad_outputs=grad_output,
            retain_graph=False,
            create_graph=False,
        )
        return output.detach(), tuple(gradient.detach() for gradient in gradients)

    return operation


def _run_iterations(operation: Callable[[], object], iterations: int) -> None:
    for _ in range(iterations):
        result = operation()
        del result


def _time_operation(
    operation: Callable[[], object],
    *,
    device: torch.device,
    iterations: int,
    repeats: int,
) -> list[float]:
    samples_ms = []
    for _ in range(repeats):
        if device.type == "cuda":
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            _run_iterations(operation, iterations)
            end.record()
            end.synchronize()
            elapsed_ms = start.elapsed_time(end)
        else:
            start_time = time.perf_counter()
            _run_iterations(operation, iterations)
            elapsed_ms = (time.perf_counter() - start_time) * 1_000.0
        samples_ms.append(elapsed_ms / iterations)
    return samples_ms


def _measure_incremental_memory(
    operation: Callable[[], object], device: torch.device
) -> int | None:
    if device.type != "cuda":
        return None
    _clear_cuda(device)
    baseline = torch.cuda.memory_allocated(device)
    torch.cuda.reset_peak_memory_stats(device)
    result = operation()
    _sync(device)
    peak = torch.cuda.max_memory_allocated(device)
    del result
    return max(0, peak - baseline)


def _make_inputs(
    config: ShapeConfig,
    *,
    dtype: torch.dtype,
    device: torch.device,
    seed: int,
    requires_grad: bool,
) -> tuple[Tensor, Tensor, Tensor]:
    generator = torch.Generator(device=device).manual_seed(seed)
    q = torch.randn(
        config.batch,
        config.heads,
        config.query_length,
        config.head_dim,
        dtype=dtype,
        device=device,
        generator=generator,
    ).requires_grad_(requires_grad)
    k = torch.randn(
        config.batch,
        config.heads,
        config.key_length,
        config.head_dim,
        dtype=dtype,
        device=device,
        generator=generator,
    ).requires_grad_(requires_grad)
    v = torch.randn(
        config.batch,
        config.heads,
        config.key_length,
        config.head_dim,
        dtype=dtype,
        device=device,
        generator=generator,
    ).requires_grad_(requires_grad)
    return q, k, v


def _method_unavailable_reason(
    method: str,
    *,
    config: ShapeConfig,
    dtype: torch.dtype,
    device: torch.device,
) -> str | None:
    if method != "cuda":
        return None
    if device.type != "cuda":
        return "custom CUDA method requires a CUDA device"
    if not extension_available():
        return f"compiled extension unavailable: {extension_error() or 'unknown error'}"
    if dtype not in (torch.float16, torch.bfloat16, torch.float32):
        return f"custom CUDA method does not support {dtype}"
    if config.head_dim > 256:
        return "custom CUDA method supports head dimensions from 1 through 256"
    return None


def _base_row(
    run_id: str,
    method: str,
    dtype_name: str,
    device: torch.device,
    config: ShapeConfig,
    args: argparse.Namespace,
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "status": None,
        "error": None,
        "method": method,
        "device": str(device),
        "dtype": dtype_name,
        **asdict(config),
        "mode": args.mode,
        "sdpa_backend": args.sdpa_backend if method == "sdpa" else "not_applicable",
        "warmup": args.warmup,
        "iterations": args.iterations,
        "repeats": args.repeats,
        "latency_ms_p20": None,
        "latency_ms_median": None,
        "latency_ms_p80": None,
        "latency_ms_p95": None,
        "latency_samples_ms": None,
        "query_tokens_per_second": None,
        "dense_equivalent_tflops": None,
        "incremental_peak_memory_bytes": None,
        "incremental_peak_memory_mib": None,
        "max_abs_error": None,
        "mean_abs_error": None,
        "max_relative_error": None,
        "output_mean": None,
        "reference_status": None,
    }


def _error_metrics(output: Tensor, reference_cpu: Tensor | None) -> dict[str, float | None]:
    output_cpu = output.detach().float().cpu()
    metrics: dict[str, float | None] = {
        "max_abs_error": None,
        "mean_abs_error": None,
        "max_relative_error": None,
        "output_mean": output_cpu.mean().item(),
    }
    if reference_cpu is None:
        return metrics
    difference = (output_cpu - reference_cpu).abs()
    relative = difference / reference_cpu.abs().clamp_min(1e-8)
    metrics.update(
        {
            "max_abs_error": difference.max().item(),
            "mean_abs_error": difference.mean().item(),
            "max_relative_error": relative.max().item(),
        }
    )
    return metrics


def _dense_equivalent_flops(config: ShapeConfig, mode: str) -> int:
    multiplier = 4 if mode == "forward" else 12
    return (
        multiplier
        * config.batch
        * config.heads
        * config.query_length
        * config.key_length
        * config.head_dim
    )


def _benchmark_method(
    *,
    run_id: str,
    method: str,
    dtype_name: str,
    dtype: torch.dtype,
    device: torch.device,
    config: ShapeConfig,
    q: Tensor,
    k: Tensor,
    v: Tensor,
    grad_output: Tensor | None,
    reference_cpu: Tensor | None,
    reference_status: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    row = _base_row(run_id, method, dtype_name, device, config, args)
    row["reference_status"] = reference_status
    unavailable = _method_unavailable_reason(
        method, config=config, dtype=dtype, device=device
    )
    if unavailable is not None:
        row.update(status="unavailable", error=unavailable)
        return row

    operation = _make_operation(
        method,
        q,
        k,
        v,
        grad_output,
        causal=config.causal,
        mode=args.mode,
    )
    try:
        with _method_context(method, args.sdpa_backend), torch.inference_mode():
            validation_output = _invoke_forward(
                method, q, k, v, causal=config.causal
            )
            _sync(device)
            row.update(_error_metrics(validation_output, reference_cpu))
            del validation_output

        with _method_context(method, args.sdpa_backend), _inference_context(args.mode):
            _run_iterations(operation, args.warmup)
            _sync(device)
            samples_ms = _time_operation(
                operation,
                device=device,
                iterations=args.iterations,
                repeats=args.repeats,
            )

        with _method_context(method, args.sdpa_backend), _inference_context(args.mode):
            incremental_memory = _measure_incremental_memory(operation, device)

        median_ms = statistics.median(samples_ms)
        latency_seconds = median_ms / 1_000.0
        query_tokens = config.batch * config.query_length
        flops = _dense_equivalent_flops(config, args.mode)
        row.update(
            status="ok",
            latency_ms_p20=_percentile(samples_ms, 0.20),
            latency_ms_median=median_ms,
            latency_ms_p80=_percentile(samples_ms, 0.80),
            latency_ms_p95=_percentile(samples_ms, 0.95),
            latency_samples_ms=samples_ms,
            query_tokens_per_second=query_tokens / latency_seconds,
            dense_equivalent_tflops=flops / latency_seconds / 1.0e12,
            incremental_peak_memory_bytes=incremental_memory,
            incremental_peak_memory_mib=(
                incremental_memory / MEBIBYTE if incremental_memory is not None else None
            ),
        )
    except Exception as exc:  # Continue the matrix and preserve the exact failure.
        row.update(
            status="oom" if _is_out_of_memory(exc) else "error",
            error=f"{type(exc).__name__}: {exc}",
        )
    finally:
        _clear_cuda(device)
    return row


def _reference_for_validation(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    config: ShapeConfig,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[Tensor | None, str]:
    if args.validation == "none":
        return None, "disabled"
    score_elements = (
        config.batch * config.heads * config.query_length * config.key_length
    )
    if args.max_validation_elements and score_elements > args.max_validation_elements:
        return None, f"skipped_size_limit:{score_elements}"
    try:
        with torch.inference_mode():
            output = manual_attention(
                q.detach(), k.detach(), v.detach(), causal=config.causal
            )
            _sync(device)
            reference_cpu = output.detach().float().cpu()
            del output
        _clear_cuda(device)
        return reference_cpu, "ok"
    except Exception as exc:
        _clear_cuda(device)
        status = "oom" if _is_out_of_memory(exc) else "error"
        return None, f"{status}:{type(exc).__name__}:{exc}"


def _configuration_seed(base_seed: int, config: ShapeConfig, dtype_index: int) -> int:
    # Stable arithmetic avoids Python's process-randomized hash().
    return (
        base_seed
        + dtype_index * 1_000_003
        + config.batch * 101
        + config.heads * 1_009
        + config.query_length * 10_007
        + config.key_length * 100_003
        + config.head_dim * 97
        + int(config.causal)
    ) % (2**63 - 1)


def _iter_shapes(args: argparse.Namespace) -> Iterator[ShapeConfig]:
    causal_modes = {
        "false": (False,),
        "true": (True,),
        "both": (False, True),
    }[args.causal]
    if args.key_lengths is None:
        length_pairs = ((length, length) for length in args.query_lengths)
    else:
        length_pairs = (
            (query_length, key_length)
            for query_length in args.query_lengths
            for key_length in args.key_lengths
        )
    for query_length, key_length in length_pairs:
        for head_dim in args.head_dims:
            for causal in causal_modes:
                yield ShapeConfig(
                    batch=args.batch,
                    heads=args.heads,
                    query_length=query_length,
                    key_length=key_length,
                    head_dim=head_dim,
                    causal=causal,
                )


def run_benchmarks(
    args: argparse.Namespace, device: torch.device, run_id: str
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dtype_index, dtype_name in enumerate(args.dtypes):
        dtype = DTYPES[dtype_name]
        for config in _iter_shapes(args):
            if config.causal and config.query_length != config.key_length:
                for method in args.methods:
                    row = _base_row(run_id, method, dtype_name, device, config, args)
                    row.update(
                        status="invalid_configuration",
                        error="causal attention requires query_length == key_length",
                        reference_status="not_run",
                    )
                    rows.append(row)
                continue

            seed = _configuration_seed(args.seed, config, dtype_index)
            try:
                q, k, v = _make_inputs(
                    config,
                    dtype=dtype,
                    device=device,
                    seed=seed,
                    requires_grad=args.mode == "forward-backward",
                )
                grad_output = (
                    torch.randn_like(q) if args.mode == "forward-backward" else None
                )
            except Exception as exc:
                status = "oom" if _is_out_of_memory(exc) else "error"
                for method in args.methods:
                    row = _base_row(run_id, method, dtype_name, device, config, args)
                    row.update(
                        status=status,
                        error=f"input allocation failed: {type(exc).__name__}: {exc}",
                        reference_status="not_run",
                    )
                    rows.append(row)
                _clear_cuda(device)
                continue

            reference_cpu, reference_status = _reference_for_validation(
                q, k, v, config, args, device
            )
            for method in args.methods:
                row = _benchmark_method(
                    run_id=run_id,
                    method=method,
                    dtype_name=dtype_name,
                    dtype=dtype,
                    device=device,
                    config=config,
                    q=q,
                    k=k,
                    v=v,
                    grad_output=grad_output,
                    reference_cpu=reference_cpu,
                    reference_status=reference_status,
                    args=args,
                )
                rows.append(row)
                latency = row["latency_ms_median"]
                latency_text = f"{latency:.4f} ms" if latency is not None else "-"
                print(
                    f"{row['status']:>21}  {method:<9} {dtype_name:<8} "
                    f"Nq={config.query_length:<5} Nk={config.key_length:<5} "
                    f"D={config.head_dim:<3} {latency_text}"
                )

            del q, k, v, grad_output, reference_cpu
            _clear_cuda(device)
    return rows


def _safe_tag(tag: str) -> str:
    normalized = "".join(character if character.isalnum() else "-" for character in tag)
    return normalized.strip("-")[:64]


def write_artifacts(
    rows: list[dict[str, Any]],
    metadata: dict[str, Any],
    output_dir: Path,
    run_id: str,
    tag: str,
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_{_safe_tag(tag)}" if _safe_tag(tag) else ""
    stem = f"benchmark_{run_id}{suffix}"
    csv_path = output_dir / f"{stem}.csv"
    json_path = output_dir / f"{stem}.json"

    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {"schema_version": 1, "metadata": metadata, "results": rows},
            handle,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        handle.write("\n")

    fieldnames = list(rows[0]) if rows else []
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            serialized = {
                key: json.dumps(value) if isinstance(value, (list, dict)) else value
                for key, value in row.items()
            }
            writer.writerow(serialized)
    return csv_path, json_path


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    _require_runtime_dependencies()
    device = _resolve_device(parser, args.device)
    if args.methods is None:
        args.methods = ["cuda", "sdpa", "reference"] if device.type == "cuda" else [
            "sdpa",
            "reference",
        ]
    if args.dtypes is None:
        args.dtypes = ["float16"] if device.type == "cuda" else ["float32"]
    _validate_cuda_requirement(parser, args, device)

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    metadata = _environment_metadata(args, device, run_id)
    print(
        f"Benchmark run {run_id}: device={device}, methods={','.join(args.methods)}, "
        f"dtypes={','.join(args.dtypes)}, mode={args.mode}"
    )
    rows = run_benchmarks(args, device, run_id)
    csv_path, json_path = write_artifacts(
        rows, metadata, args.output_dir, run_id, args.tag
    )
    print(f"CSV:  {csv_path}")
    print(f"JSON: {json_path}")

    successful = [row for row in rows if row["status"] == "ok"]
    custom_successful = [
        row for row in successful if row["method"] == "cuda"
    ]
    unsuccessful = [row for row in rows if row["status"] != "ok"]
    if args.require_cuda and not custom_successful:
        print("ERROR: --require-cuda was set but no custom CUDA measurement succeeded", file=sys.stderr)
        return 2
    if not successful:
        print("ERROR: no benchmark configuration completed successfully", file=sys.stderr)
        return 1
    if args.fail_on_error and unsuccessful:
        print(
            f"ERROR: {len(unsuccessful)} configuration(s) were unsuccessful",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
