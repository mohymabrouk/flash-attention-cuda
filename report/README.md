# Technical report

The report is deliberately self-contained and compiles without benchmark data. Missing empirical artifacts are rendered as visible `RESULT_PENDING` boxes; no synthetic GPU measurements are substituted.

## Build with `latexmk`

From this directory:

```bash
mkdir -p build
latexmk main.tex
```

The configured output is `build/main.pdf`.

## Portable build with `pdflatex` and BibTeX

If `latexmk` is unavailable:

```bash
pdflatex -interaction=nonstopmode -halt-on-error main.tex
bibtex main
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

## Supplying future results

The benchmark pipeline should generate the optional files documented in `sections/appendix_result_schema.tex`, especially:

- `generated/environment.tex`
- `generated/correctness_summary.tex`
- `generated/performance_summary.tex`
- `generated/memory_summary.tex`
- `generated/profiling_summary.tex`

The repository includes a helper to create these files from one benchmark JSON artifact:

```bash
python -m benchmarks.render_report_artifacts \
  --benchmark-json ../results/runs/benchmark_<run-id>.json \
  --output-dir generated
```

Each generated file must be derived from checked-in CSV/JSON output and include the source artifact path. The report never treats a missing file as a zero, failed run, or successful result.

LaTeX auxiliary files and the generated PDF are build products. They should be ignored by the repository-level `.gitignore`; the report source and bibliography should remain tracked.
