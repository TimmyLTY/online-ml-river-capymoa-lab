# Dataset provenance and preparation

Run `python src/data.py` from the repository root. All selected real datasets
are used in full, with a 50-instance training-only prefix. Only seed 42 is used
in the submitted experiment matrix. The generated manifest records array
checksums, feature names, sample counts, source links, ordering and known drifts.

| Dataset | Complete size | Source / attribution | Preparation |
| --- | ---: | --- | --- |
| TrumpApproval | 1001 | River / FiveThirtyEight, [loader](https://riverml.xyz/latest/api/datasets/TrumpApproval/) | Original time order, six numeric features |
| ChickWeights | 578 | River / ChickWeight, [loader](https://riverml.xyz/latest/api/datasets/ChickWeights/) | Drop chick ID; one-hot diet; preserve time order |
| BikeDaily | 731 | [UCI Bike Sharing](https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset), Hadi Fanaee-T | Full `day.csv`; exclude `casual`, `registered`, ID/date; one-hot nominal calendar/weather fields |
| Elec2 | 45312 | River / NSW electricity, [loader](https://riverml.xyz/latest/api/datasets/Elec2/) | Full source stream, numeric feature conversion |
| Phishing | 1250 | River / UCI, [loader](https://riverml.xyz/latest/api/datasets/Phishing/) | Complete static dataset, seeded permutation |
| BreastCancer | 569 | [scikit-learn / UCI Wisconsin Diagnostic](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html) | Complete static dataset, seeded permutation |
| Iris | 150 | [scikit-learn / Fisher Iris](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_iris.html) | Complete static dataset, seeded permutation; labels withheld from clusterers |
| FriedmanShift | 10000 | Explicit generator in `src/data.py` | Friedman function plus Gaussian noise; reverse response at 5000 |
| SEAReversal | 10000 | Explicit SEA-based generator in `src/data.py` | Reverse binary rule at 5000, 5% label noise |
| MovingBlobs | 6000 | Explicit generator in `src/data.py` | Three Gaussian groups; center translation at 3000 |

Source dataset licenses/terms remain with their providers; this repository does
not redistribute the original source downloads. `results/raw/` contains derived
prediction and evaluation traces, not the full input feature tables. Consult the
linked source descriptions and citations before reusing the data beyond the lab.

No preprocessing fits statistics on the scored future observations. Cache files
are local and ignored by Git. The same prepared arrays feed River and CapyMOA.
