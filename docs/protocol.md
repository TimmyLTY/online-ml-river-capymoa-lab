# Frozen experiment protocol

Configuration: `configs/experiment.json`. All real datasets are used in full;
there is no prefix/subsample substitution. Seed: 42.

## Data and preprocessing

Regression: River TrumpApproval (1001), ChickWeights (578), UCI Bike Sharing
**daily** data (731), and 10000 generated Friedman observations. Chick ID is
excluded and diet is one-hot encoded. Bike features exclude ID, date and the
label components `casual` and `registered`; nominal weather/calendar categories
are one-hot encoded using their documented domains. All four preserve temporal
or generated order.

Classification: River Elec2 (45312), Phishing (1250), scikit-learn Wisconsin
Diagnostic Breast Cancer (569), and a 10000-instance SEA-based stream.
Elec2 preserves source order; Phishing and BreastCancer use a seed-specific
permutation because no temporal interpretation is asserted for their row order.
All original numeric Elec2 features, including encoded weekday, are retained.

Synthetic regression uses the standard Friedman mean plus independent N(0,1)
noise. At zero-based index 5000, y becomes 30-y. Synthetic classification uses
uniform [0,10]^3 features and y = 1[x0+x1>8]; the label is reversed at index 5000,
with independent 5% label noise. This is an explicit abrupt SEA-based label
reversal, not an unmodified SEA generator. Both changes preserve P(X).

The first 50 observations form a common training-only warm-up. Fixed feature
means and population SDs are fitted on these 50 observations only (zero SD -> 1).
Regression targets are also standardized using only warm-up labels; predictions
and reported MAE/RMSE return to original units. Fixed scaling avoids changing
coordinates for retained KNN examples. Both libraries consume identical cached
arrays in identical order. Every prepared array has a SHA-256 checksum in
`results/data-manifest.json`.

## Four methods per supervised task

- Regression: HoeffdingTreeRegressor, ARFRegressor (10 trees), LinearRegression
  (Adam learning rate .01, intercept rate .01), KNNRegressor (k=5, window=500).
  Trees use mean leaves, grace period 50, delta=1e-7, tau=.05. ARF retains its
  native random feature selection, Poisson resampling and drift adaptation.
- Classification: HoeffdingTreeClassifier, ADWINBaggingClassifier (10 trees),
  LogisticRegression (Adam .01, no L2), KNNClassifier (k=5, window=500,
  unweighted votes). Trees use majority leaves, grace=50, delta=1e-7,
  tau=.05, information gain and no pre-pruning.
- CapyMOA counterparts: HoeffdingTree, OnlineAdwinBagging, SGDClassifier with
  logistic loss, KNN. Match explicit tree settings, ensemble size, k, window,
  regularization and initial learning rate. CapyMOA's sklearn SGD wrapper uses
  inverse-scaling learning rate (eta0=.01, default power_t=.5), whereas River
  uses coordinate-wise Adam; this is a closest-equivalent linear family
  comparison, not a controlled implementation-speed comparison. Native split
  observers, distance normalization, Poisson RNGs, tie handling and internal
  drift logic can also differ. No library is emulated by wrapping the other.

All remaining constructor defaults belong to the pinned versions. Factories in
`src/models.py` specify overrides; instantiated parameter records are retained
in `results/model-configurations.json`.

## Prequential evaluation and cost

Warm-up examples are trained but not scored. For each subsequent example:
construct library input, predict, then learn. Missing classification predictions
use label 0; missing standardized regression predictions use 0 (warm-up mean).
Report MAE, RMSE, R2 or accuracy, macro-F1, Cohen's kappa.

One sequential worker process runs each configuration. Numerical thread counts
are 1 and the Java maximum heap is 2 GiB. Per-instance latency includes input
conversion, prediction and learning, and detector/reset work where applicable.
It excludes shared data loading/scaling, package/JVM import, warm-up, metric
aggregation and disk output. Total evaluation runtime additionally includes
loop bookkeeping and RSS sampling. Initialization time is separately recorded.
RSS is sampled every 250 observations and at completion after warm-up: it is
whole-process resident memory, including Python/JVM, not model-only memory.
Sampled peaks can miss between-sample spikes. JVM JIT and desktop load affect
these measurements. The report uses one fixed seed (42), as requested, and
makes no claim about between-seed variation. Previously completed seed-42 runs
remain valid when removing the planned additional seeds; `protocol-revision.json`
records this scope reduction without changing numerical measurements.

## Drift comparison and handling ablation

Use the same no-reset Hoeffding-tree prediction errors for a detector-only
comparison, so all detectors observe identical signals. Regression input is
min(|y-pred|/(3*sigma_warmup),1); classification input is binary error.
Regression detectors: ADWIN(delta=.002), Page-Hinkley(min=30, delta=.005,
threshold=50, alpha=.9999, both directions), KSWIN(alpha=.005, window=100,
stat=30, seed). Classification: ADWIN, DDM(warm=30, warning=2, drift=3),
EDDM(warm=30, alpha=.95, beta=.9). Reinitialize a detector after its alarm.

Then run each detector in a live reset policy. Predict first; update the
detector from that error; on alarm reinitialize tree and detector; train the
current example exactly once. Keep the common scaler and target scale fixed.
Compare each reset policy to the no-reset HT, globally and post-change.
Alarm positions are zero-based original-stream indices (including warm-up).
Delay is first post-change alarm index minus 5000; report missed detections as
missing, never zero. Count pre-change alarms separately. Real-stream alarm
counts have no known true-change labels and are not detection accuracy.

## Clustering (Step 9)

Use full Iris (150, shuffled per seed, labels hidden from learners) and 6000
MovingBlobs observations (three Gaussian groups, noise SD .55, translated by
[3,1.5] at index 3000). Scaling and warm-up follow the common protocol.
CluStream: 30 microclusters, k=3, horizon=1000. ClusTree: horizon=1000,
max_height=3. DenStream: horizon=1000, epsilon=.6, beta=.2, mu=2,
init_points=50, offline=2, lambda=.01, speed=100.

Use block-prequential evaluation: freeze current centers, assign the next block
by nearest center, measure quality, then train on that block. Blocks have 25
Iris or 250 MovingBlobs observations. Evaluate squared nearest-center distance,
silhouette where defined, and ARI against withheld reference labels. ARI labels
are used only for evaluation. Use macrocenters when implemented; ClusTree
exposes microcenters. Its partition granularity is therefore different. Missing
centers/undefined silhouette remain missing. Nearest-center assignment forces
assignment even for DenStream; it does not evaluate its native noise rejection.
Report weighted block means, valid-block counts, evolution and snapshots.
Clustering timing includes metric computation and center extraction; unlike
supervised timing, its amortized latency includes quality evaluation overhead.

## AutoML (Step 10 bonus)

Compare BanditClassifier (epsilon=.1, burn-in=100) with
SuccessiveHalvingClassifier (budget=12000, eta=2, min_models=1), using the
same four CapyMOA classifier instances and full four classification streams.
Compare to each static candidate, not an oracle-selected held-out model.
This bonus compares two genuine CapyMOA AutoML approaches; AutoClass is not
part of this selected comparison. No additional hyperparameter tuning is done.

## Report budget

English LaTeX article, 10 pages maximum including references. Target 9 pages:
protocol/data/methods 1; River regression 1; regression drift 1;
River classification 1; classification drift 1; cross-library 1;
clustering 1; AutoML/conclusions/reproducibility/references 1–2.

### Clustering transport detail

The pinned MOA clustering implementation expects a finite placeholder class
field in its Java instance representation. Each clustering input therefore
carries the constant placeholder 0 via `LabeledInstance`; the actual reference
class is never passed to the learner. This avoids NaN propagation from missing
class fields. Returned centers are checked to have exactly the feature dimension.
