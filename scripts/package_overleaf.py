"""Package the standalone report source for Overleaf (no external paths)."""
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

root = Path(__file__).resolve().parents[1]
report = root / 'report'
files = [report / 'main.tex']
files += sorted((report / 'tables').glob('*.tex'))
files += sorted(p for p in (report / 'figures').iterdir() if p.suffix in {'.pdf', '.jpg'})
assert files[0].is_file()
assert '../' not in files[0].read_text(), 'Report must not depend on parent directories'
output = report / 'overleaf.zip'
with ZipFile(output, 'w', ZIP_DEFLATED) as archive:
    for path in files:
        archive.write(path, path.relative_to(report))
    archive.writestr('README.txt', '''Online Machine Learning with River and CapyMOA
Tianyi LIANG - ENSTA Paris

Overleaf: New Project > Upload Project > select this ZIP.
Main document: main.tex. Compiler: pdfLaTeX.
All tables, figures and references are included; no Python or external paths needed.
Edit main.tex to revise the report. Compile twice to resolve references.
Local build: pdflatex main.tex (twice).
''')
print(output)
print(f'{len(files) + 1} files; {output.stat().st_size:,} bytes')
