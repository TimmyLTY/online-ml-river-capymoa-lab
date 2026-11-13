"""Register this interpreter and its Java runtime as the lab Jupyter kernel."""

import os
from pathlib import Path
import sys

from ipykernel.kernelspec import install


def main():
    candidates = [Path(sys.prefix) / "lib" / "jvm", Path(sys.prefix) / "Library"]
    if os.environ.get("JAVA_HOME"):
        candidates.append(Path(os.environ["JAVA_HOME"]))
    java_home = next(
        (p for p in candidates if (p / "bin" / "java").exists()
         or (p / "bin" / "java.exe").exists()), None
    )
    if java_home is None:
        raise RuntimeError("Activate the lab Conda environment with OpenJDK first.")
    path = install(
        user=True,
        kernel_name="online-ml-lab",
        display_name="Online ML Lab (River + CapyMOA)",
        env={"JAVA_HOME": str(java_home), "CAPYMOA_JVM_ARGS": "-Xmx2g -Xss10M"},
    )
    print(f"Registered kernel: {path}")


if __name__ == "__main__":
    main()
