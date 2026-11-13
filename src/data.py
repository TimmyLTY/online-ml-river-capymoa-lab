"""Acquire complete datasets and persist identical numeric streams for both libraries."""
from pathlib import Path
import hashlib, io, json, urllib.request, zipfile
import numpy as np
import pandas as pd
from river import datasets
from sklearn.datasets import load_breast_cancer, load_iris

ROOT = Path(__file__).resolve().parents[1]
CFG = json.loads((ROOT/'configs/experiment.json').read_text())
DATA = ROOT/'data'


def river_arrays(ds, transform=None):
    rows=list(ds)
    if transform: rows=[(transform(x), y) for x,y in rows]
    keys=sorted(rows[0][0])
    return np.array([[float(x[k]) for k in keys] for x,y in rows]), np.array([y for x,y in rows]),keys


def prepare():
    DATA.mkdir(exist_ok=True)
    raw={}
    raw['TrumpApproval']=river_arrays(datasets.TrumpApproval())
    raw['ChickWeights']=river_arrays(datasets.ChickWeights(), lambda x: {'time':x['time'],**{f'diet_{i}':float(x['diet']==i) for i in range(1,5)}})
    url='https://archive.ics.uci.edu/static/public/275/bike+sharing+dataset.zip'
    path=DATA/'bike-sharing.zip'
    if not path.exists(): urllib.request.urlretrieve(url,path)
    with zipfile.ZipFile(path) as z: frame=pd.read_csv(z.open('day.csv'))
    # casual + registered exactly equals target cnt: never expose either to a learner.
    columns=['yr','holiday','workingday','temp','atemp','hum','windspeed']
    features=frame[columns].copy()
    for col,values in [('season',range(1,5)),('mnth',range(1,13)),('weekday',range(7)),('weathersit',range(1,5))]:
        for v in values: features[f'{col}_{v}']=(frame[col]==v).astype(float)
    raw['BikeDaily']=(features.to_numpy(float),frame['cnt'].to_numpy(float),list(features))
    raw['Elec2']=river_arrays(datasets.Elec2())
    raw['Phishing']=river_arrays(datasets.Phishing())
    b=load_breast_cancer(); raw['BreastCancer']=(b.data,b.target,list(b.feature_names))
    iris=load_iris();raw['Iris']=(iris.data,iris.target,list(iris.feature_names))
    sources={
      'TrumpApproval':'https://riverml.xyz/latest/api/datasets/TrumpApproval/',
      'ChickWeights':'https://riverml.xyz/latest/api/datasets/ChickWeights/',
      'BikeDaily':url,'Elec2':'https://riverml.xyz/latest/api/datasets/Elec2/',
      'Phishing':'https://riverml.xyz/latest/api/datasets/Phishing/',
      'BreastCancer':'https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html',
      'Iris':'https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_iris.html'}
    manifest=[]
    for seed in CFG['seeds']:
        rng=np.random.default_rng(seed)
        current=dict(raw)
        n=CFG['synthetic_n']; t=CFG['drift_index']
        X=rng.uniform(0,1,(n,10)); y=10*np.sin(np.pi*X[:,0]*X[:,1])+20*(X[:,2]-.5)**2+10*X[:,3]+5*X[:,4]+rng.normal(0,1,n)
        y[t:]=30-y[t:]
        current['FriedmanShift']=(X,y,[f'x{i}' for i in range(10)])
        X=rng.uniform(0,10,(n,3)); y=(X[:,0]+X[:,1]>8).astype(int); y[t:]=1-y[t:]
        noise=rng.random(n)<.05; y[noise]=1-y[noise]
        current['SEAReversal']=(X,y,['x0','x1','x2'])
        ncluster=6000; yc=rng.integers(0,3,ncluster); centers=np.array([[-3,-2],[3,-2],[0,3]])
        X=centers[yc]+rng.normal(0,.55,(ncluster,2)); X[3000:]+=np.array([3.,1.5])
        current['MovingBlobs']=(X,yc,['x0','x1'])
        for name,(X,y,features) in current.items():
            order='source chronological order' if name in ['TrumpApproval','ChickWeights','BikeDaily','Elec2'] else 'generated order'
            if name in ['Phishing','BreastCancer','Iris']:
                order='seeded permutation of complete static dataset'; perm=np.random.default_rng(seed).permutation(len(y));X,y=X[perm],y[perm]
            X=np.asarray(X,dtype=np.float64); y=np.asarray(y,dtype=np.float64 if name in CFG['regression'] else np.int64)
            assert np.isfinite(X).all() and np.isfinite(y).all()
            file=DATA/f'{name}-{seed}.npz'; np.savez_compressed(file,X=X,y=y)
            checksum=hashlib.sha256(X.tobytes()+y.tobytes()).hexdigest()
            manifest.append(dict(dataset=name,seed=seed,n=len(y),features=features,n_features=X.shape[1],order=order,sha256=checksum,source=sources.get(name,'src/data.py'),drift_index=t if name in ['FriedmanShift','SEAReversal'] else 3000 if name=='MovingBlobs' else None))
    (ROOT/'results/data-manifest.json').write_text(json.dumps(manifest,indent=2))
    print('Prepared',len(manifest),'complete streams',flush=True)


def load(name,seed):
    with np.load(DATA/f'{name}-{seed}.npz') as d: X,y=d['X'],d['y']
    w=CFG['warmup']; mean=X[:w].mean(0);sd=X[:w].std(0); sd[sd<1e-12]=1
    return (X-mean)/sd,y

if __name__=='__main__':prepare()
