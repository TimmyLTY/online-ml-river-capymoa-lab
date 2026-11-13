# Setup validation

Verified on 13 November 2026 on macOS arm64.

| Component | Installed version |
| --- | --- |
| Python | 3.11.18 |
| OpenJDK | 17.0.18 |
| River | 0.26.1 |
| CapyMOA | 0.15.1 |
| JPype | 1.7.1 |
| JupyterLab | 4.6.6 |
| ipykernel | 7.4.0 |

Completed checks:

- `python -m pip check`: no broken requirements.
- River: 100-instance classification and regression prediction/training loops,
  plus ADWIN updates.
- CapyMOA: Java startup, a 100-instance classifier loop, and a 100-instance
  Clustream run with finite micro-cluster centers.
- Registered Jupyter kernel with the environment's Python interpreter and Java.
- Executed `notebooks/00_environment_check.ipynb` using the registered kernel.
- Exported the executed notebook to HTML using nbconvert.

All learning checks use tiny synthetic streams solely to exercise installed
code paths. No full Lab datasets, benchmark comparisons, report results, or
AutoML bonus experiments have been produced.

The smoke checks and registered kernel use a 2 GiB maximum Java heap
(`CAPYMOA_JVM_ARGS=-Xmx2g -Xss10M`). Future experiments should record this setting
and deliberately revise it if required; a Java heap limit is not a bound on
total Python/JVM process memory.

Local validation outputs are generated under `validation-results/` and
`artifacts/`, which are ignored by Git. The committed notebook has no outputs.
To repeat validation:

```bash
python -m pip check
python check_environment.py
jupyter nbconvert --to notebook --execute notebooks/00_environment_check.ipynb \
  --output-dir validation-results --output 00_environment_check.executed \
  --ExecutePreprocessor.timeout=180
jupyter nbconvert --to html validation-results/00_environment_check.executed.ipynb \
  --output-dir artifacts --output environment-check
```
