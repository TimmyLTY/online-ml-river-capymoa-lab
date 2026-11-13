"""Check coverage, paired data, provenance and recomputed metrics before submission."""
from pathlib import Path
import sys,json
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from data import ROOT,CFG,load
from run_experiments import specs,key,fingerprint,scores

expected=specs();assert len(expected)==86
current=fingerprint();checked=0
for s in expected:
 p=ROOT/'results/raw'/(key(s)+'.json');r=json.loads(p.read_text())
 assert r['status']=='completed' and r['fingerprint']==current,(p,'provenance mismatch')
 assert r['n_test']==r['n_total']-50
 assert r['runtime_s']>0 and r['latency_us']>0
 if s['phase']!='clustering':
  z=np.load(p.with_suffix('.npz'));assert len(z['y'])==r['n_test'];assert np.isfinite(z['pred']).all()
  for k,v in scores(z['y'],z['pred'],s['task']).items():assert np.isclose(r[k],v), (p,k)
  _,y=load(s['dataset'],s['seed']);assert np.array_equal(z['y'],y[50:])
  assert np.array_equal(z['index'],np.arange(50,len(y)))
 else:
  windows=json.loads(p.with_name(p.stem+'-windows.json').read_text())
  assert sum(w['end']-w['start'] for w in windows)==r['n_test']
 checked+=1
alarms=json.loads((ROOT/'results/tables/detector-alarms.json').read_text());assert len(alarms)==24
for r in alarms:
 assert r['n_alarms']==len(r['positions'])
 if r['detected'] is not None:
  after=[p for p in r['positions'] if p>=5000]
  assert r['delay']==(after[0]-5000 if after else None)
for s in expected:
 if s['phase']=='main' and s['library']=='CapyMOA':
  a=np.load(ROOT/'results/raw'/(key(s)+'.npz'));s2={**s,'library':'River'};b=np.load(ROOT/'results/raw'/(key(s2)+'.npz'))
  assert np.array_equal(a['y'],b['y']) and np.array_equal(a['index'],b['index'])
report={'status':'passed','runs_checked':checked,'detector_replays_checked':len(alarms),'checks':['complete expected matrix','source/config/data fingerprint','all scored observations retained','metrics recomputed from predictions','cross-library targets and order identical','delay/missing detection consistency'],'fingerprint':current}
(ROOT/'results/validation.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
