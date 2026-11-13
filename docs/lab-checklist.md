# Lab requirements and next-session checklist

Source: the three-page course handout dated 9 November 2026. Suggested methods
and datasets must be checked against the installed library versions before use.

## Preparation

- [x] Read the course handout and separate setup from experimental work.
- [x] Validate isolated River and CapyMOA installation, including the JVM.
- [x] Validate the Jupyter kernel and HTML export.
- [x] Record dependency versions and initialize the submission repository.

## Evaluation protocol (all experiments)

- [ ] Prequential evaluation: predict/test before training on each instance.
- [ ] Fix random seeds, dataset contents and order, and comparable hyperparameters.
- [ ] Report predictive metrics, latency per instance, total runtime, and memory
  when possible. Define whether preprocessing and initialization are timed.
- [ ] Document hardware, software, and any unavoidable cross-library differences.

## Part 1: River

- [x] Step 1: installation (covered in preparation).
- [ ] Steps 2–3: four regression methods (tree, ensemble, linear with adaptive
  optimizer, and one other); three real datasets plus one synthetic dataset;
  multiple metrics such as MAE, RMSE and R2, and runtime.
- [ ] Step 4: compare drift detectors on those datasets; report number and
  positions of alarms, delay where synthetic drift locations are known, and
  predictive impact of drift handling.
- [ ] Step 5: repeat method/dataset comparisons and drift analysis for
  classification: four methods, three real datasets plus one synthetic,
  accuracy/F1/Kappa and computation cost.
- [ ] Step 6: analyze the Part 1 findings in the report.

## Part 2: CapyMOA

- [x] Step 7: installation (covered in preparation).
- [ ] Step 8: the same four classification methods, or closest equivalents;
  compare accuracy, processing time and latency with River on the same data.
- [ ] Step 9: multiple streaming clustering methods; examine quality and
  evolution over time on unsupervised datasets.
- [ ] Step 10 (optional bonus): evaluate CapyMOA AutoML approaches.

## Final deliverables

- [ ] PDF report only, 5–10 pages: methods, experimental setup, tables/figures,
  accuracy/cost trade-offs, drift analysis, cross-library comparison and conclusions.
- [ ] Notebook HTML export(s) or code repository link.
- [ ] Submit the same required deliverables via the course email route and Moodle.
- [ ] Deadline: 20 November 2026, 23:59 CET. Maximum group size: two; solo bonus.

Submission actions are future user actions, not part of environment preparation.

## Next session

1. Verify dataset and learner availability in the pinned versions.
2. Decide the datasets, four learners per task, matched CapyMOA counterparts,
   drift scenarios, and a shared evaluation protocol.
3. Implement reproducible stream adapters and evaluation with configuration files.
4. Run full experiments, save raw measurements, and create tables and figures.
5. Write the evidence-based report and check every deliverable against this list.
