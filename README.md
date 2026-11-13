# Online Machine Learning with River and CapyMOA

**Tianyi LIANG · M2 Data Science, Institut Polytechnique de Paris**

Completed course lab with a concise **9-page English report**, editable LaTeX,
executed notebook, and reproducible experiments. The main analysis uses the
single fixed random seed **42**. All selected real datasets are used in full.

## Deliverables

- [PDF report](report/report.pdf) — Steps 1–9 and Step 10 AutoML bonus.
- [LaTeX source](report/main.tex) — compiled with `scripts/build_report.sh`.
- [Executed results notebook](notebooks/01_lab_results.ipynb).
- [Notebook HTML export](results/lab_results.html) — download and open locally.
- [Experiment protocol](docs/protocol.md) and [step coverage](docs/lab-checklist.md).
- [Per-run measurements](results/tables/runs.csv), [exact detector alarms](results/tables/detector-alarms.json),
  [metric validation](results/validation.json), [report/notebook checks](results/artifact-validation.json), and [data manifest](results/data-manifest.json).

The experiment matrix contains **86 completed runs**: 16 River regression,
32 classification across both libraries, 24 detector/reset policies, 6 streaming
clustering runs, and 8 AutoML runs. Another 24 detector-only replays compare
alarms on identical baseline error traces. Raw `.npz` files retain predictions,
true labels, indices, and per-instance latency; JSON files record metrics,
configuration identity, memory, and timing.

## Reproduce

```bash
conda env create -f environment.yml
conda activate online-ml-lab
python scripts/register_kernel.py
python check_environment.py
python src/data.py
python src/run_experiments.py
python scripts/export_model_configs.py
python scripts/analyze_results.py
python scripts/verify_results.py
python scripts/build_results_notebook.py
bash scripts/build_report.sh
```

The final command needs a LaTeX installation with `pdflatex`, `geometry`,
`lmodern`, `microtype`, `booktabs`, `amsmath`, `amssymb`, `caption`, `fancyhdr`,
`enumitem` and `hyperref`. On macOS the script also recognizes
`/Library/TeX/texbin/pdflatex`. A notebook screenshot is retained as a captured
artifact in `results/figures/`; the numerical tables/plots are generated from
saved run outputs.

`src/run_experiments.py` runs isolated workers sequentially, with numerical
threads limited to one. It resumes only matching source/config/data fingerprints.
Removing existing raw results forces fresh measurements; timing values will vary
with hardware and desktop load. `results/protocol-revision.json` documents the
user-requested reduction from three planned seeds to seed 42 and the reuse of
unchanged completed seed-42 results. No seed-43 or seed-44 experiment is reported.

## Experiment design

- **Regression:** TrumpApproval, ChickWeights, UCI Bike Sharing daily data, and
  FriedmanShift; HT, ten-tree ARF, LinearAdam, windowed KNN; MAE/RMSE/R2 and cost.
- **Classification:** full Elec2, Phishing, Wisconsin Diagnostic Breast Cancer,
  and SEAReversal; HT, ten-tree ADWIN bagging, logistic learner, windowed KNN.
- **Drift:** ADWIN/Page–Hinkley/KSWIN for regression and ADWIN/DDM/EDDM for
  classification; alarm positions, synthetic delay, and live reset ablations.
- **Clustering:** CluStream, ClusTree and DenStream on Iris and MovingBlobs;
  block-prequential evaluation, silhouette, ARI and squared nearest-center distance.
- **AutoML:** CapyMOA Bandit and Successive Halving with the same four-model pool.

The first 50 examples are a shared training-only warm-up. Scaling uses only this
prefix. Both libraries receive identical stream arrays and order. The linear
counterparts use different native optimizers (River Adam versus CapyMOA
inverse-scaling SGD); the report explicitly limits cross-library attribution.

## Environment and artifacts

Python 3.11.18 · OpenJDK 17.0.18 · River 0.26.1 · CapyMOA 0.15.1.
`requirements.txt` pins direct dependencies; `requirements-lock.txt` records the
Python package snapshot; `conda-osx-arm64.lock.txt` records the Conda platform
packages. Use `environment.yml` for portable setup.

Raw datasets and prepared caches stay in ignored `data/`; downloads and feature
construction are implemented in `src/data.py`. Plot PDFs/PNGs, result records,
the executed notebook and the report are versioned. Local build products and
logs are ignored.

To browse interactively, run `jupyter lab` and choose the kernel
**Online ML Lab (River + CapyMOA)**. `00_environment_check.ipynb` is an installation
check; `01_lab_results.ipynb` contains the full experiment results.

The handout's deadline is **20 November 2026, 23:59 CET**. The repository and PDF
are prepared; email and Moodle submission remain the student's actions.
