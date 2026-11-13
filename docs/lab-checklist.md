# Lab coverage and submission checklist

Source: course handout dated 9 November 2026. The report uses **seed 42** and
complete selected real datasets. Sections follow the Lab Steps.

| Lab requirement | Completed evidence |
| --- | --- |
| Step 1: River installation | `environment.yml`, `check_environment.py`, `docs/setup-validation.md` |
| Steps 2–3: four regression families, 3 real + 1 synthetic, metrics and cost | `configs/experiment.json`, `results/tables/regression.tex`, report section 2 |
| Step 4: regression detectors, positions/delay, handling effect | `results/tables/detector-alarms.json`, `regression-drift.tex`, report section 3 |
| Step 5: four classifiers, 3 real + 1 synthetic, multiple metrics and drift | `classification.tex`, `classification-drift.tex`, report sections 4–5 |
| Step 6: written analysis | `report/main.tex`, `report/report.pdf`, report sections 2–9 |
| Step 7: CapyMOA and Java installation | Same reproducible environment and setup checks |
| Step 8: matched classification comparison | `cross-library.tex`, report section 6; optimizer differences explicitly documented |
| Step 9: several stream clusterers, quality and evolution | `clustering.tex`, cluster figures and window records, report section 7 |
| Step 10 bonus: AutoML approaches | Bandit + Successive Halving on all four classification streams; report section 8 |
| Test-then-train, fixed seeds, same data and aligned settings | `docs/protocol.md`, `src/models.py`, `results/data-manifest.json` |
| Performance, latency, runtime and memory | `results/tables/runs.csv`, raw records and traces |
| Evidence validation | `scripts/verify_results.py`, `results/validation.json` |
| 5–10 page PDF and code/notebook deliverable | 9-page `report/report.pdf`, executed notebook + HTML, GitHub repository |

- [x] Experiments completed and full coverage checked.
- [x] Metrics recomputed from stored predictions; cross-library targets/order matched.
- [x] Single-seed report and editable LaTeX prepared.
- [x] Executed results notebook and HTML export prepared.
- [ ] Student sends the PDF and HTML or repository link through the course email route.
- [ ] Student uploads the same required deliverables to Moodle.

Deadline: **20 November 2026, 23:59 CET**. Maximum two students per group;
the handout offers a solo bonus. No email or Moodle submission was performed.
