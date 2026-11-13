#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../report"
mkdir -p build
latex_bin="${PDFLATEX:-pdflatex}"
if ! command -v "$latex_bin" >/dev/null 2>&1 && [[ -x /Library/TeX/texbin/pdflatex ]]; then
  latex_bin=/Library/TeX/texbin/pdflatex
fi
"$latex_bin" -interaction=nonstopmode -halt-on-error -output-directory=build main.tex > build/compile-pass1.log
"$latex_bin" -interaction=nonstopmode -halt-on-error -output-directory=build main.tex > build/compile-pass2.log
cp build/main.pdf report.pdf
