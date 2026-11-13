# Online Machine Learning with River and CapyMOA

**Tianyi LIANG · ENSTA Paris**

Lab Steps 1–9 and the AutoML bonus, using fixed **seed 42**.

- [Final report (PDF)](Lab%20Online%20ML_Tianyi%20LIANG%20River_CapyMOA.pdf)
- [Requirement audit](docs/lab-checklist.md) · [Protocol](docs/protocol.md)

## Structure

```text
configs/    Experiment settings
src/        Data preparation, models and experiment runner
scripts/    Environment check, analysis and validation
results/    Summary measurements, alarm positions and provenance
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
```

The experiment runs 86 configurations plus 24 detector replays during analysis.
Datasets and per-instance traces are generated locally and ignored by Git.
Runtime depends on hardware. Analysis regenerates figures and table inputs under
`results/generated/` (ignored by Git). The final PDF is supplied by the author;
LaTeX sources and report compilation files are not included. These commands
reproduce the experiments, not the final PDF layout.
