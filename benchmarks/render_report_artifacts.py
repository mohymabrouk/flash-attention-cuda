"""Render benchmark JSON into optional LaTeX snippets for the report."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Convert a benchmark JSON artifact into optional LaTeX snippets "
            "consumed by report/main.tex."
        )
    )
    parser.add_argument(
        "--benchmark-json",
        type=Path,
        required=True,
        help="Path to the JSON artifact emitted by benchmarks.bench_attention.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("report/generated"),
        help="Directory where generated .tex files should be written.",
    )
    return parser


def _latex_escape(value: Any) -> str:
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    for source, target in replacements.items():
        text = text.replace(source, target)
    return text


def _fmt_float(value: Any, digits: int = 4) -> str:
    if value is None:
        return "--"
    return f"{float(value):.{digits}f}"


def _fmt_memory_mib(value: Any) -> str:
    if value is None:
        return "--"
    return f"{float(value):.2f}"


def _status_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for row in rows:
        counts[str(row.get("status", "unknown"))] += 1
    return dict(sorted(counts.items()))


def _best_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    successful = [row for row in rows if row.get("status") == "ok"]
    successful.sort(
        key=lambda row: (
            row["query_length"],
            row["key_length"],
            row["head_dim"],
            str(row["dtype"]),
            bool(row["causal"]),
            row["latency_ms_median"],
            str(row["method"]),
        )
    )

    chosen: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in successful:
        key = (
            row["query_length"],
            row["key_length"],
            row["head_dim"],
            row["dtype"],
            row["causal"],
            row["method"],
        )
        chosen.setdefault(key, row)
    return list(chosen.values())


def _comparison_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        if row.get("status") != "ok":
            continue
        key = (
            row["query_length"],
            row["key_length"],
            row["head_dim"],
            row["dtype"],
            row["causal"],
            row["mode"],
        )
        groups[key][str(row["method"])] = row

    comparisons: list[dict[str, Any]] = []
    for key, methods in sorted(groups.items()):
        if "cuda" not in methods:
            continue
        candidate = {
            "query_length": key[0],
            "key_length": key[1],
            "head_dim": key[2],
            "dtype": key[3],
            "causal": key[4],
            "mode": key[5],
            "cuda_latency_ms": methods["cuda"]["latency_ms_median"],
            "cuda_memory_mib": methods["cuda"]["incremental_peak_memory_mib"],
            "cuda_max_abs_error": methods["cuda"]["max_abs_error"],
            "cuda_tflops": methods["cuda"]["dense_equivalent_tflops"],
            "speedup_vs_sdpa": None,
            "speedup_vs_reference": None,
        }
        if "sdpa" in methods and methods["sdpa"]["latency_ms_median"]:
            candidate["speedup_vs_sdpa"] = (
                methods["sdpa"]["latency_ms_median"]
                / methods["cuda"]["latency_ms_median"]
            )
        if "reference" in methods and methods["reference"]["latency_ms_median"]:
            candidate["speedup_vs_reference"] = (
                methods["reference"]["latency_ms_median"]
                / methods["cuda"]["latency_ms_median"]
            )
        comparisons.append(candidate)
    return comparisons


def _render_environment(metadata: dict[str, Any], source_path: Path) -> str:
    git = metadata.get("git", {})
    software = metadata.get("software", {})
    cuda = metadata.get("cuda", {})
    devices = cuda.get("devices", [])
    device_names = ", ".join(
        _latex_escape(device.get("name", "unknown")) for device in devices
    ) or "none"
    return rf"""\begin{{table}}[htbp]
\centering
\caption{{Execution environment captured from the benchmark artifact.}}
\label{{tab:generated-environment}}
\begin{{tabularx}}{{\textwidth}}{{@{{}}l X@{{}}}}
\toprule
Field & Value \\
\midrule
Benchmark JSON & \texttt{{{_latex_escape(source_path)}}} \\
Run identifier & \texttt{{{_latex_escape(metadata.get("run_id", "--"))}}} \\
Timestamp (UTC) & {_latex_escape(metadata.get("timestamp_utc", "--"))} \\
Git commit & \texttt{{{_latex_escape(git.get("commit", "--"))}}} \\
Dirty worktree & {_latex_escape(git.get("dirty", "--"))} \\
PyTorch & {_latex_escape(software.get("torch", "--"))} \\
PyTorch CUDA & {_latex_escape(software.get("torch_cuda", "--"))} \\
nvcc & {_latex_escape(software.get("nvcc_path", "--"))} \\
CUDA device(s) & {device_names} \\
Benchmark command & \texttt{{{_latex_escape(metadata.get("command", "--"))}}} \\
\bottomrule
\end{{tabularx}}
\end{{table}}
"""


def _render_correctness(rows: list[dict[str, Any]], source_path: Path) -> str:
    successful = [row for row in rows if row.get("status") == "ok"]
    if not successful:
        return rf"""\begin{{resultplaceholder}}[Correctness summary missing]
No successful benchmark row was available in \texttt{{{_latex_escape(source_path)}}}, so no
numerical summary could be generated. Populate this file after a successful benchmark run.
\end{{resultplaceholder}}
"""

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Maximum observed forward-error metrics among successful benchmark rows.}",
        r"\label{tab:generated-correctness}",
        r"\begin{tabular}{@{}l c c c c@{}}",
        r"\toprule",
        r"Method & Rows & Max abs.\ error & Mean abs.\ error & Max rel.\ error \\",
        r"\midrule",
    ]
    by_method: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in successful:
        by_method[str(row["method"])].append(row)
    for method, method_rows in sorted(by_method.items()):
        lines.append(
            " ".join(
                [
                    _latex_escape(method),
                    "&",
                    str(len(method_rows)),
                    "&",
                    _fmt_float(max(row["max_abs_error"] or 0.0 for row in method_rows)),
                    "&",
                    _fmt_float(max(row["mean_abs_error"] or 0.0 for row in method_rows)),
                    "&",
                    _fmt_float(max(row["max_relative_error"] or 0.0 for row in method_rows)),
                    r"\\",
                ]
            )
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
            rf"\noindent\textit{{Source artifact:}} \texttt{{{_latex_escape(source_path)}}}",
            "",
        ]
    )
    return "\n".join(lines)


def _render_performance(rows: list[dict[str, Any]], source_path: Path) -> str:
    comparisons = _comparison_rows(rows)
    if not comparisons:
        return rf"""\begin{{resultplaceholder}}[Performance comparison pending]
The benchmark artifact \texttt{{{_latex_escape(source_path)}}} does not yet contain a successful
custom CUDA row with a comparable baseline, so no speedup table was emitted.
\end{{resultplaceholder}}
"""

    lines = [
        r"\begin{longtable}{@{}r r r l c r r r r@{}}",
        r"\caption{Latency and speedup summary for successful custom-CUDA benchmark rows.}"
        r"\label{tab:generated-performance}\\",
        r"\toprule",
        r"$N_q$ & $N_k$ & $D$ & Dtype & Causal & CUDA ms & TFLOP/s & vs.\ SDPA & vs.\ ref. \\",
        r"\midrule",
        r"\endfirsthead",
        r"\toprule",
        r"$N_q$ & $N_k$ & $D$ & Dtype & Causal & CUDA ms & TFLOP/s & vs.\ SDPA & vs.\ ref. \\",
        r"\midrule",
        r"\endhead",
    ]
    for row in comparisons:
        lines.append(
            " ".join(
                [
                    str(row["query_length"]),
                    "&",
                    str(row["key_length"]),
                    "&",
                    str(row["head_dim"]),
                    "&",
                    _latex_escape(row["dtype"]),
                    "&",
                    ("yes" if row["causal"] else "no"),
                    "&",
                    _fmt_float(row["cuda_latency_ms"]),
                    "&",
                    _fmt_float(row["cuda_tflops"]),
                    "&",
                    _fmt_float(row["speedup_vs_sdpa"], digits=3),
                    "&",
                    _fmt_float(row["speedup_vs_reference"], digits=3),
                    r"\\",
                ]
            )
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{longtable}",
            rf"\noindent\textit{{Source artifact:}} \texttt{{{_latex_escape(source_path)}}}",
            "",
        ]
    )
    return "\n".join(lines)


def _render_memory(rows: list[dict[str, Any]], source_path: Path) -> str:
    best_rows = _best_rows(rows)
    successful = [row for row in best_rows if row.get("status") == "ok"]
    if not successful:
        return rf"""\begin{{resultplaceholder}}[Memory summary pending]
No successful rows were available in \texttt{{{_latex_escape(source_path)}}} for the memory table.
\end{{resultplaceholder}}
"""

    lines = [
        r"\begin{longtable}{@{}l r r r l c r@{}}",
        r"\caption{Incremental peak allocator memory recorded during successful benchmark rows.}"
        r"\label{tab:generated-memory}\\",
        r"\toprule",
        r"Method & $N_q$ & $N_k$ & $D$ & Dtype & Causal & Peak MiB \\",
        r"\midrule",
        r"\endfirsthead",
        r"\toprule",
        r"Method & $N_q$ & $N_k$ & $D$ & Dtype & Causal & Peak MiB \\",
        r"\midrule",
        r"\endhead",
    ]
    for row in successful:
        lines.append(
            " ".join(
                [
                    _latex_escape(row["method"]),
                    "&",
                    str(row["query_length"]),
                    "&",
                    str(row["key_length"]),
                    "&",
                    str(row["head_dim"]),
                    "&",
                    _latex_escape(row["dtype"]),
                    "&",
                    ("yes" if row["causal"] else "no"),
                    "&",
                    _fmt_memory_mib(row["incremental_peak_memory_mib"]),
                    r"\\",
                ]
            )
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{longtable}",
            rf"\noindent\textit{{Source artifact:}} \texttt{{{_latex_escape(source_path)}}}",
            "",
        ]
    )
    return "\n".join(lines)


def _render_profiling(rows: list[dict[str, Any]], source_path: Path) -> str:
    counts = _status_counts(rows)
    status_summary = ", ".join(
        f"{_latex_escape(status)}={count}" for status, count in counts.items()
    )
    return rf"""\begin{{reprobox}}
The benchmark JSON \texttt{{{_latex_escape(source_path)}}} does not contain Nsight Compute or
Compute Sanitizer counters. This generated file therefore records only benchmark status counts
({status_summary}) and acts as a reminder that profiler-based explanation must be attached as a
separate artifact before any microarchitectural claim is promoted into the main text.
\end{{reprobox}}
"""


def render_artifacts(
    benchmark_json: Path, output_dir: Path
) -> dict[str, Path]:
    payload = json.loads(benchmark_json.read_text(encoding="utf-8"))
    metadata = payload.get("metadata", {})
    rows = payload.get("results", [])
    output_dir.mkdir(parents=True, exist_ok=True)

    rendered = {
        "environment.tex": _render_environment(metadata, benchmark_json),
        "correctness_summary.tex": _render_correctness(rows, benchmark_json),
        "performance_summary.tex": _render_performance(rows, benchmark_json),
        "memory_summary.tex": _render_memory(rows, benchmark_json),
        "profiling_summary.tex": _render_profiling(rows, benchmark_json),
    }

    written: dict[str, Path] = {}
    for name, content in rendered.items():
        destination = output_dir / name
        destination.write_text(content, encoding="utf-8")
        written[name] = destination
    return written


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    written = render_artifacts(args.benchmark_json, args.output_dir)
    for name, destination in sorted(written.items()):
        print(f"{name}: {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
