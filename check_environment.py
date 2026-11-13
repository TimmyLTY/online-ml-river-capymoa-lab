"""Offline installation checks only; these are not lab experiment results."""

import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import random
import sys


def check_environment():
    if sys.version_info < (3, 11):
        raise RuntimeError("Python 3.11 or newer is required.")
    # Direct interpreter execution may bypass Conda's Java activation hook.
    bundled_java = Path(sys.prefix) / "lib" / "jvm"
    if (bundled_java / "bin" / "java").exists():
        os.environ["JAVA_HOME"] = str(bundled_java)
    os.environ.setdefault("CAPYMOA_JVM_ARGS", "-Xmx2g -Xss10M")

    import numpy as np
    from river import datasets, drift, linear_model, optim, preprocessing, tree
    import capymoa
    import jpype
    from capymoa.classifier import HoeffdingTree
    from capymoa.cluster import Clustream
    from capymoa.core import Instance
    from capymoa.stream.generator import SEA

    random.seed(42)
    np.random.seed(42)
    classifier = tree.HoeffdingTreeClassifier()
    predictions = []
    for x, y in datasets.synth.SEA(seed=42).take(100):
        predictions.append(classifier.predict_one(x))
        classifier.learn_one(x, y)
    if not any(p is not None for p in predictions[1:]):
        raise RuntimeError("River classifier did not produce predictions.")

    regressor = preprocessing.StandardScaler() | linear_model.LinearRegression(
        optimizer=optim.Adam()
    )
    regression_predictions = []
    for x, y in datasets.synth.Friedman(seed=42).take(100):
        regression_predictions.append(regressor.predict_one(x))
        regressor.learn_one(x, y)
    if not all(math.isfinite(p) for p in regression_predictions):
        raise RuntimeError("River regression returned non-finite predictions.")

    detector = drift.ADWIN()
    for value in [0] * 100 + [1] * 100:
        detector.update(value)

    stream = SEA(instance_random_seed=42)
    moa_classifier = HoeffdingTree(schema=stream.get_schema())
    moa_predictions = []
    for _ in range(100):
        instance = stream.next_instance()
        moa_predictions.append(moa_classifier.predict(instance))
        moa_classifier.train(instance)
    if not any(p is not None for p in moa_predictions[1:]):
        raise RuntimeError("CapyMOA classifier did not produce predictions.")

    # Feed only features into the clusterer. No dataset download is needed.
    clusterer = Clustream(schema=stream.get_schema(), max_num_kernels=10)
    for _ in range(100):
        x = stream.next_instance().x
        clusterer.train(Instance.from_array(stream.get_schema(), x))
    centers = clusterer.get_micro_clustering_result().get_centers()
    if not len(centers) or not np.isfinite(np.asarray(centers)).all():
        raise RuntimeError("CapyMOA clustering did not produce finite centers.")

    packages = ["river", "capymoa", "jpype1", "numpy", "pandas", "scipy",
                "scikit-learn", "jupyterlab", "ipykernel", "nbconvert"]
    result = {
        "status": "passed",
        "purpose": "installation smoke checks; not experimental findings",
        "python": platform.python_version(),
        "platform": f"{platform.system()} {platform.machine()}",
        "java": str(jpype.java.lang.System.getProperty("java.version")),
        "packages": {p: importlib.metadata.version(p) for p in packages},
        "checks": ["River classification (100 instances)",
                   "River regression (100 instances)", "River ADWIN updates",
                   "CapyMOA JVM classification (100 instances)",
                   "CapyMOA JVM clustering (100 instances)"],
    }
    print(json.dumps(result, indent=2))
    print("Environment ready.")
    return result


if __name__ == "__main__":
    check_environment()
