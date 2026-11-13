"""Check coverage, paired data, provenance and recomputed metrics before submission."""
from pathlib import Path
import sys,json,hashlib
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from data import ROOT,CFG,load
from run_experiments import specs,key,fingerprint,scores

expected=specs();assert len(expected)==86
current=fingerprint();checked=0
manifest=json.loads((ROOT/'results/data-manifest.json').read_text())
assert len(manifest)==10 and CFG['seeds']==[42]
counts={}
for record in manifest:
 with np.load(ROOT/'data'/f"{record['dataset']}-{record['seed']}.npz") as values:
  X,y=values['X'],values['y']
  assert hashlib.sha256(X.tobytes()+y.tobytes()).hexdigest()==record['sha256']
  assert X.shape==(record['n'],record['n_features']) and len(y)==record['n']
  counts[record['dataset']]=record['n']
assert counts==dict(TrumpApproval=1001,ChickWeights=578,BikeDaily=731,Elec2=45312,Phishing=1250,BreastCancer=569,Iris=150,FriedmanShift=10000,SEAReversal=10000,MovingBlobs=6000)
for s in expected:
 p=ROOT/'results/raw'/(key(s)+'.json');r=json.loads(p.read_text())
 assert r['status']=='completed' and r['fingerprint']==current,(p,'provenance mismatch')
 assert r['n_total']==counts[s['dataset']] and r['seed']==42
 assert r['n_test']==r['n_total']-CFG['warmup']
 assert r['runtime_s']>0 and r['latency_us']>0
 if s['phase']!='clustering':
  z=np.load(p.with_suffix('.npz'));assert len(z['y'])==r['n_test'];assert np.isfinite(z['pred']).all()
  assert np.isclose(z['latency_us'].mean(),r['latency_us'])
  assert np.isclose(np.percentile(z['latency_us'],95),r['p95_us'])
  for k,v in scores(z['y'],z['pred'],s['task']).items():assert np.isclose(r[k],v), (p,k)
  _,y=load(s['dataset'],s['seed']);assert np.array_equal(z['y'],y[50:])
  assert np.array_equal(z['index'],np.arange(50,len(y)))
 else:
  windows=json.loads(p.with_name(p.stem+'-windows.json').read_text())
  assert sum(w['end']-w['start'] for w in windows)==r['n_test']
  assert windows[0]['start']==CFG['warmup'] and windows[-1]['end']==r['n_total']
  assert all(a['end']==b['start'] for a,b in zip(windows,windows[1:]))
  for metric in ['silhouette','ari','prequential_mse']:
   valid=[w for w in windows if w[metric] is not None]
   assert np.isclose(np.average([w[metric] for w in valid],weights=[w['end']-w['start'] for w in valid]),r[metric])
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
report={'status':'passed','runs_checked':checked,'detector_replays_checked':len(alarms),'checks':['complete expected matrix','source/config/data fingerprint','all scored observations retained','metrics recomputed from predictions','cross-library targets and order identical','delay/missing detection consistency','full dataset counts and array checksums','latency aggregates from stored traces','weighted clustering metrics from held-out blocks'],'fingerprint':current}
(ROOT/'results/validation.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
