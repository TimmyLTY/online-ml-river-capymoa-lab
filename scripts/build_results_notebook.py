"""Build and execute the submission notebook from saved full experimental outputs."""
from pathlib import Path
import nbformat
from nbclient import NotebookClient
from nbconvert import HTMLExporter

ROOT=Path(__file__).resolve().parents[1]
cells=[nbformat.v4.new_markdown_cell('''# Online Machine Learning with River and CapyMOA
**Tianyi LIANG · M2 Data Science · 13 November 2026**

This executed notebook reads the saved full experiment outputs. The actual training code is in `src/`; reproduce all runs with `python src/data.py` followed by `python src/run_experiments.py`. The report is `report/report.pdf` and its editable source is `report/main.tex`.

## Steps 1 and 7 — Environment and completed coverage
Seed **42** is fixed and paired across methods and libraries. Every selected real dataset is used in full. Each stream begins with a common 50-instance training-only warm-up, followed by test-then-train evaluation. The clustering experiment uses test-block-then-train-block evaluation.'''),
nbformat.v4.new_code_cell('''from pathlib import Path
import json
import pandas as pd
from IPython.display import display, Image
root = Path.cwd()
if not (root / 'results/tables/runs.csv').exists():
    root = root.parent
runs = pd.read_csv(root / 'results/tables/runs.csv')
coverage = runs.groupby(['phase','library']).size().rename('Completed runs').reset_index()
display(coverage)
print('Total completed experiment runs:', len(runs))
manifest = json.loads((root / 'results/run-manifest.json').read_text())
print('Hardware:', manifest['cpu'], '| RAM:', manifest['ram_gib'], 'GiB')
print('Fixed seeds:', manifest['seeds'])''')]
sections=[
('Steps 2–3 — River regression','main','regression','River',['mae','rmse','r2','latency_us','runtime_s','rss_peak_mib'],['regression-learning']),
('Step 4 — Regression drift and model resets','reset','regression','River',['rmse','post_rmse','latency_us'],['regression-drift']),
('Step 5 — River classification','main','classification','River',['accuracy','f1','kappa','latency_us','runtime_s'],['classification-learning','classification-drift']),
('Step 8 — CapyMOA classification and cross-library comparison','main','classification','CapyMOA',['accuracy','f1','kappa','latency_us','runtime_s','rss_peak_mib'],['cross-library']),
('Step 9 — Streaming clustering','clustering','clustering','CapyMOA',['silhouette','ari','prequential_mse','latency_us','runtime_s'],['cluster-evolution','cluster-snapshots']),
('Step 10 (bonus) — CapyMOA AutoML','automl','classification','CapyMOA',['accuracy','f1','latency_us','runtime_s'],['automl'])]
for title,phase,task,library,metrics,figures in sections:
 cells.append(nbformat.v4.new_markdown_cell('## '+title+'\n\nTables show the actual seed-42 measurements. Complete measurements are in `results/tables/`.'))
 cells.append(nbformat.v4.new_code_cell(f"selected = runs.query(\"phase == {phase!r} and task == {task!r} and library == {library!r}\")\ndisplay(selected.groupby(['dataset', 'model'])[{metrics!r}].mean().round(4))"))
 for fig in figures:cells.append(nbformat.v4.new_code_cell(f"display(Image(filename=str(root / 'results/figures/{fig}.png')))"))
cells.insert(2,nbformat.v4.new_markdown_cell('''### Key comparison: Elec2 classification
The table below gives accuracy as a fraction, per-instance processing latency in microseconds, and evaluation time in seconds. All 45,312 observations are used, with 45,262 scored after warm-up. Native optimizer and implementation differences are detailed in the PDF.'''))
cells.insert(3,nbformat.v4.new_code_cell("display(runs.query(\"phase=='main' and dataset=='Elec2'\").groupby(['library','model'])[['accuracy','latency_us','runtime_s']].mean().round(4))"))
cells.append(nbformat.v4.new_markdown_cell('''## Step 6 — Interpretation and reproducibility
The PDF report analyzes the quality–cost trade-offs, detector alarms and recovery, cross-library differences, cluster evolution, and AutoML allocation. See `docs/protocol.md` for timing boundaries, fixed preprocessing, detector parameters, and the constant placeholder used in clustering transport. `results/tables/detector-alarms.json` preserves exact alarm positions; `.npz` files in `results/raw/` retain prediction/latency traces.

References and dataset attribution are included in the report and `data/README.md`. No email or Moodle submission has been performed.'''))
nb=nbformat.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'Online ML Lab (River + CapyMOA)','language':'python','name':'online-ml-lab'}})
NotebookClient(nb,timeout=180,kernel_name='online-ml-lab',resources={'metadata':{'path':str(ROOT)}}).execute()
path=ROOT/'notebooks/01_lab_results.ipynb';nbformat.write(nb,path)
exporter=HTMLExporter();exporter.exclude_input=True
body,_=exporter.from_notebook_node(nb)
(ROOT/'results/lab_results.html').write_text(body)
print('Executed',path,'and exported results/lab_results.html')
