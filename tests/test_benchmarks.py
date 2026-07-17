"""Lightweight tests for benchmark CLI helpers and report artifact rendering."""

from __future__ import annotations

import json

from benchmarks.bench_attention import _iter_shapes, build_parser
from benchmarks.render_report_artifacts import render_artifacts


def test_benchmark_cli_defaults_to_noncausal_mode() -> None:
    args = build_parser().parse_args([])
    shapes = list(_iter_shapes(args))

    assert args.causal == "false"
    assert shapes
    assert {shape.causal for shape in shapes} == {False}


def test_benchmark_cli_supports_flag_form_for_causal_true() -> None:
    args = build_parser().parse_args(["--causal"])
    shapes = list(_iter_shapes(args))

    assert args.causal == "true"
    assert shapes
    assert {shape.causal for shape in shapes} == {True}


def test_benchmark_cli_supports_both_causal_modes() -> None:
    args = build_parser().parse_args(
        ["--query-lengths", "128", "--head-dims", "64", "--causal", "both"]
    )
    shapes = list(_iter_shapes(args))

    assert len(shapes) == 2
    assert [shape.causal for shape in shapes] == [False, True]


def test_render_report_artifacts_writes_expected_tex_files(tmp_path) -> None:
    benchmark_json = tmp_path / "benchmark.json"
    benchmark_json.write_text(
        json.dumps(
            {
                "metadata": {
                    "run_id": "demo-run",
                    "timestamp_utc": "2026-07-17T00:00:00Z",
                    "command": "python -m benchmarks.bench_attention",
                    "git": {"commit": "abc123", "dirty": False},
                    "software": {
                        "torch": "2.12.0",
                        "torch_cuda": "12.6",
                        "nvcc_path": "/usr/local/cuda/bin/nvcc",
                    },
                    "cuda": {
                        "devices": [
                            {"name": "NVIDIA T4", "compute_capability": [7, 5]}
                        ]
                    },
                },
                "results": [
                    {
                        "status": "ok",
                        "method": "cuda",
                        "dtype": "float16",
                        "query_length": 128,
                        "key_length": 128,
                        "head_dim": 64,
                        "causal": False,
                        "mode": "forward",
                        "latency_ms_median": 1.2,
                        "incremental_peak_memory_mib": 6.5,
                        "max_abs_error": 0.001,
                        "mean_abs_error": 0.0002,
                        "max_relative_error": 0.01,
                        "dense_equivalent_tflops": 0.9,
                    },
                    {
                        "status": "ok",
                        "method": "sdpa",
                        "dtype": "float16",
                        "query_length": 128,
                        "key_length": 128,
                        "head_dim": 64,
                        "causal": False,
                        "mode": "forward",
                        "latency_ms_median": 1.8,
                        "incremental_peak_memory_mib": 8.0,
                        "max_abs_error": 0.0,
                        "mean_abs_error": 0.0,
                        "max_relative_error": 0.0,
                        "dense_equivalent_tflops": 0.6,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    written = render_artifacts(benchmark_json, tmp_path / "generated")

    assert set(written) == {
        "environment.tex",
        "correctness_summary.tex",
        "performance_summary.tex",
        "memory_summary.tex",
        "profiling_summary.tex",
    }
    performance = written["performance_summary.tex"].read_text(encoding="utf-8")
    assert "vs.\\ SDPA" in performance
    assert "128" in performance
