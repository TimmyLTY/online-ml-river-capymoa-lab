# Lab requirement audit

Audited against all three pages of **Lab: Online Machine Learning with River and
CapyMOA**, dated 9 November 2026. Audit date: 13 November 2026.
Author affiliation: **Tianyi LIANG — ENSTA Paris**.

## Final PDF update

The original audit below describes the experiment coverage checked before the
author's final PDF replacement. The final author-supplied PDF is preserved
unchanged. Its first page has two unresolved citation markers `[?]`; its §9.1
still lists `scripts/build_report.sh`, which belonged to the earlier source
package. Follow the current README for experiment reproduction. LaTeX sources,
Overleaf ZIPs and compilation files have been removed at the author's request.
The final PDF has 9 pages. No new full content audit is claimed for this revision.

## Step-by-step coverage

| Requirement | Code / evidence | Report |
| --- | --- | --- |
| Step 1: install River, Python ≥3.11 | `environment.yml`; `scripts/check_environment.py` passes regression, classification and drift smoke checks | §1 |
| Steps 2–3: four regression families | `src/models.py`: HT, ten-tree ARF, LinearAdam, windowed KNN | §2, Table 2 |
| Three real regression streams + one synthetic | Full TrumpApproval (1001), ChickWeights (578), BikeDaily (731), FriedmanShift (10000); `src/data.py` | §1–2, Tables 1–2 |
| Several regression metrics and running time | MAE, RMSE, R², latency, runtime and RSS; `results/tables/runs.csv` | Table 2, Figure 1 and analysis |
| Step 4: detectors on those same streams | ADWIN, Page–Hinkley, KSWIN replay identical HT errors; separate live model-reset experiments | §3 |
| Number and position of detected drifts | Full positions in `results/tables/detector-alarms.json`; counts and first/last positions for every dataset | Table 3, Figure 2 |
| Detection delay when the change is known | Synthetic change index 5000; first subsequent alarm minus 5000 | Table 3; real-stream Pre/Hit/Delay are not applicable |
| Effect of drift handling on model performance | 12 regression reset policies versus no-reset HT; global and post-change metrics | Table 3, Figure 2 and analysis |
| Step 5: repeat with classification | HT, ADWIN bagging, adaptive logistic regression, KNN; full Elec2 (45312), Phishing (1250), BreastCancer (569), SEAReversal (10000) | §4–5 |
| Accuracy, F1 and Kappa | Accuracy, macro-F1, Cohen's κ plus cost | Table 4, Figure 3 |
| Classification detectors, delay and handling | ADWIN, DDM, EDDM; 12 live reset policies; complete alarm positions | Table 5, Figure 4 |
| Step 6: analyze Part 1 | Accuracy/cost, drift response and situations favoring each model | §2–5, §9 |
| Step 7: install CapyMOA and Java | Pinned CapyMOA and OpenJDK 17; JVM classification/clustering smoke checks pass | §1 |
| Step 8: same / closest four classifiers | Matched explicit tree, ensemble, KNN and initial linear settings; native Adam/SGD difference disclosed | §6 |
| Compare accuracy, total processing time and latency | All three measures shown directly for both libraries | Table 6, Figure 5 and analysis |
| Step 9: several stream clusterers | CluStream, ClusTree, DenStream; Iris and MovingBlobs, labels withheld from training | §7 |
| Clustering quality and evolution | Held-out-block silhouette, ARI, squared distance, cluster counts and center snapshots | Table 7, Figures 6–7 |
| Step 10 optional AutoML comparison | Bandit and Successive Halving with identical four-candidate pools on all four classification streams | §8, Table 8, Figure 8 |

## Shared protocol and deliverables

| Requirement | Verification |
| --- | --- |
| Test before train | Reviewed `supervised()`: predict before learning; reset trains the current item once. Clustering freezes centers before scoring each block, then trains that block. The first 50 observations are a disclosed training-only warm-up. |
| Fixed random seeds | Only seed 42 is configured. One fixed seed meets the handout; repeat-seed uncertainty is not claimed. |
| Same datasets and comparable parameters | Ten regenerated arrays exactly match saved SHA-256 checksums. Cross-library target/order arrays match. Explicit common settings are shared through `configs/experiment.json`; native implementation differences are documented, including closest-equivalent Adam versus SGD. |
| No future-data preprocessing / target leakage | Scalers use only the 50-item prefix. Bike target components are excluded. Synthetic changes are fixed in advance. Clustering learners receive a constant placeholder class, never the reference class. |
| Predictive metrics and cost | Saved predictions reproduce reported supervised metrics; saved latency traces reproduce mean/p95 latency. Runtime and sampled process RSS are recorded. Weighted clustering metrics reproduce block records. |
| Brief methods and experimental setup | §§1–8 cover mechanisms, datasets, hyperparameters, metrics and hardware. Detailed constructor settings are in `results/model-configurations.json`. |
| Tables, figures and interpretation | Tables 1–8, Figures 1–9, and per-step analysis; installation steps share the environment section. |
| Conclusions on model choice | §9 states choices and limitations. |
| PDF, 5–10 pages | 9-page PDF including references. |
| HTML notebook **or** code repository | The public GitHub repository supplies the code option; duplicate notebooks/HTML are not required and have been removed. |

## Corrections made during this audit

- Added **total runtime for both libraries** to Table 6; it was previously only in the saved measurements.
- Added **first/last alarm positions on every dataset** to Tables 3 and 5; exact lists remain in JSON. Clarified that real-stream Pre/Hit/Delay are not applicable without labelled change locations.
- Displayed runtimes to three decimal places, avoiding small positive times rounded to `0.00`.
- Added brief clustering mechanism descriptions and changed the author's affiliation to ENSTA Paris.
- Removed duplicate exports, environment files and per-instance traces from the tracked repository. Raw traces remain local and are regenerated by the runner.

No remaining experimental-content or report-deliverable omission was found in
the original audit; see the final PDF update above for the later revision. The native optimizer mismatch is an explicit limitation of the
closest-equivalent comparison, not evidence of intrinsic library superiority.

Course submission (email and Moodle, deadline 20 November 2026 at 23:59 CET)
is a separate student action. This audit does not send or submit anything.
