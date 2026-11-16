"""Run PDF text and collision checks on all final DC motor figures."""
from pathlib import Path
import subprocess,sys,json,os

ws=Path(sys.argv[1]).resolve();out=ws/'presentation';qa=out/'qa'
skill=Path('C:/Users/sk3547969879/.agents/skills/nature-figure/scripts')
rows=[]
for pdf in sorted((out/'figures').glob('*.pdf')):
    for kind,script in [('text','audit_pdf_text.py'),('collision','audit_figure_collisions.py')]:
        result=subprocess.run([sys.executable,str(skill/script),str(pdf),'--json'],capture_output=True,text=True,encoding='utf8',env={**os.environ,'PYTHONUTF8':'1','PYTHONIOENCODING':'utf-8'})
        path=qa/f'{pdf.stem}.{kind}.json';path.write_text(result.stdout,encoding='utf8')
        if result.returncode:raise RuntimeError(f'{kind} audit failed for {pdf.name}: {result.stdout} {result.stderr}')
        report=json.loads(result.stdout);rows.append(dict(figure=pdf.name,audit=kind,verdict=report.get('verdict'),status='passed'))
(qa/'audit-summary.json').write_text(json.dumps(rows,indent=2),encoding='utf8')
print(json.dumps(dict(figures=len(rows)//2,checks=len(rows),all_passed=True),indent=2))
