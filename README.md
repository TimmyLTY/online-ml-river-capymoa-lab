# Online Machine Learning with River and CapyMOA

Course lab repository for Tianyi LIANG (M2 Data Science, Institut Polytechnique de Paris).

**Status:** environment preparation only. Full experiments and the final report
have not started. The small synthetic runs in `check_environment.py` are
installation checks, not experimental results.

## Setup

Use Conda/Miniforge with the conda-forge channel:

```bash
conda env create -f environment.yml
conda activate online-ml-lab
python scripts/register_kernel.py
python -m pip check
python check_environment.py
jupyter lab
```

Select **Online ML Lab (River + CapyMOA)** and open
`notebooks/00_environment_check.ipynb`.

The environment includes Python 3.11, OpenJDK 17, River, CapyMOA, JupyterLab,
plotting tools, and HTML notebook export. Conda activation configures Java;
kernel registration also records this environment's Java location so the
kernel works when Jupyter is launched from another environment. PyTorch is
optional for CapyMOA's deep-learning methods and is not installed at this stage.

`requirements.txt` pins direct Python dependencies. `requirements-lock.txt`
records all pip-visible packages from the verified macOS arm64 environment.
`conda-osx-arm64.lock.txt` records Conda packages for the same platform.
Use `environment.yml` for portable setup; exact lock files may not work on
other platforms. To reproduce the recorded macOS environment:

```bash
conda create -n online-ml-lab-locked --file conda-osx-arm64.lock.txt
conda activate online-ml-lab-locked
python -m pip install -r requirements-lock.txt
python scripts/register_kernel.py
python check_environment.py
```

The kernel registration uses the name `online-ml-lab`; running it in another
environment intentionally updates that named kernel.

## Working tree

| Path | Purpose |
| --- | --- |
| `check_environment.py` | Offline import, JVM, learning, and clustering checks |
| `notebooks/00_environment_check.ipynb` | Jupyter smoke check only |
| `scripts/register_kernel.py` | Register the project kernel with its Java settings |
| `docs/lab-checklist.md` | Requirements and next-session checklist |
| `data/` | Local datasets; downloads excluded from Git |
| `src/` | Reserved for experiment implementation |
| `artifacts/` | Generated results, figures, and notebook HTML exports |
| `report/` | Reserved for report sources and the final PDF |
| `validation-results/` | Local setup validation outputs |

Dataset selection, model matching, and full evaluations will be carried out
in the next session. Keep seeds, stream order, preprocessing, hyperparameters,
and timing boundaries explicit when implementing them.

## References

- [River installation](https://riverml.xyz/latest/introduction/installation/)
- [CapyMOA setup](https://capymoa.org/setup/)
- Course handout: *Lab: Online Machine Learning with River and CapyMOA*,
  9 November 2026. The handout remains outside this public repository.

## Submission target

The final deliverables are a 5–10 page PDF report and either exported notebook
HTML or a code repository link. The course deadline is **20 November 2026,
23:59 CET**. This repository is a preparation scaffold, not a completed submission.
