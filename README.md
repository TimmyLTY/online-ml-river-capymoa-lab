# Online Machine Learning with River and CapyMOA

**Tianyi LIANG · ENSTA Paris**

Lab Steps 1–9 and the AutoML bonus, using fixed **seed 42**.

- [Report (PDF)](report/report.pdf) · [Overleaf ZIP](report/overleaf.zip)
- [Requirement audit](docs/lab-checklist.md) · [Protocol](docs/protocol.md)

## Structure

```text
configs/    Experiment settings
src/        Data preparation, models and experiment runner
scripts/    Environment check, analysis, validation and report packaging
results/    Summary measurements, alarm positions and provenance
report/     Standalone LaTeX project, figures, tables, PDF and Overleaf ZIP
docs/       Protocol and Lab requirement audit
```

## Reproduce

Tested on macOS Apple Silicon with Python 3.11 and Java 17. Create the environment
with Conda; the data script downloads the required datasets.

```bash
conda env create -f environment.yml
conda activate online-ml-lab
python scripts/check_environment.py
python src/data.py
python src/run_experiments.py
python scripts/analyze_results.py
python scripts/verify_results.py
bash scripts/build_report.sh
python scripts/package_overleaf.py
```

The experiment runs 86 configurations plus 24 detector replays during analysis.
Datasets and per-instance traces are generated locally and ignored by Git.
Runtime depends on hardware. PDF compilation needs `pdflatex` (TeX Live/MacTeX);
the supplied PDF and Overleaf ZIP can be used without rerunning Python.
Upload the ZIP as a new Overleaf project, select `main.tex` and **pdfLaTeX**.
