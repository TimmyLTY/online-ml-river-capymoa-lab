"""Sequential isolated workers: one process per learner/dataset/seed/configuration."""
import os
os.environ.update({x:'1' for x in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']})
from pathlib import Path
import sys,json,time,subprocess,hashlib,random,platform,argparse,traceback
import numpy as np
import psutil
from data import CFG,ROOT,load
from models import river_model,capy_model,schema_for,detector
os.environ['JAVA_HOME']=str(Path(sys.prefix)/'lib/jvm')
os.environ['CAPYMOA_JVM_ARGS']=CFG['jvm_args']
RAW=ROOT/'results/raw'


def scores(y,p,task):
    from sklearn.metrics import accuracy_score,f1_score,cohen_kappa_score,mean_absolute_error,mean_squared_error,r2_score
    if task=='regression':return dict(mae=mean_absolute_error(y,p),rmse=mean_squared_error(y,p)**.5,r2=r2_score(y,p))
    return dict(accuracy=accuracy_score(y,p),f1=f1_score(y,p,average='macro',zero_division=0),kappa=cohen_kappa_score(y,p))


def key(s):return '-'.join(str(s[k]) for k in ['phase','library','dataset','model','seed'])


def supervised(s):
    random.seed(s['seed']);np.random.seed(s['seed'])
    X,y=load(s['dataset'],s['seed']);task=s['task'];w=CFG['warmup'];n=len(y)
    regression=task=='regression'
    ym=float(y[:w].mean()) if regression else 0.;ys=float(y[:w].std()) if regression else 1.;ys=max(ys,1e-12)
    target=(y-ym)/ys if regression else y
    capy=s['library']=='CapyMOA'; schema=schema_for(X.shape[1]) if capy else None
    if capy:
        from capymoa.core import LabeledInstance
    def make():return capy_model(s['model'],schema,s['seed']) if capy else river_model(task,'HT' if s['phase']=='reset' else s['model'],s['seed'])
    start=time.perf_counter();model=make();init_s=time.perf_counter()-start
    det=detector(s['model'],s['seed']) if s['phase']=='reset' else None
    def input_at(i):return LabeledInstance.from_array(schema,X[i],int(y[i])) if capy else dict(enumerate(X[i]))
    for i in range(w):
        x=input_at(i)
        if capy:model.train(x)
        else:model.learn_one(x,float(target[i]) if regression else int(target[i]))
    p=[];lat=[];alarms=[];proc=psutil.Process();rss0=proc.memory_info().rss;rsspeak=rss0
    started=time.perf_counter()
    for i in range(w,n):
        begin=time.perf_counter_ns();x=input_at(i)
        pred=model.predict(x) if capy else model.predict_one(x)
        if pred is None:pred=0
        pred=float(pred)*ys+ym if regression else int(pred)
        if not np.isfinite(pred):raise ValueError(f'non-finite prediction at {i}')
        if det is not None:
            loss=min(abs(pred-y[i])/(3*ys),1.) if regression else int(pred!=y[i])
            det.update(loss)
            if det.drift_detected:
                alarms.append(i);model=make();det=detector(s['model'],s['seed'])
        if capy:model.train(x)
        else:model.learn_one(x,float(target[i]) if regression else int(target[i]))
        lat.append((time.perf_counter_ns()-begin)/1000.);p.append(pred)
        if i%250==0:rsspeak=max(rsspeak,proc.memory_info().rss)
    runtime=time.perf_counter()-started;rsspeak=max(rsspeak,proc.memory_info().rss)
    p=np.array(p);lat=np.array(lat);truth=y[w:]
    np.savez_compressed(RAW/(key(s)+'.npz'),y=truth,pred=p,latency_us=lat,index=np.arange(w,n))
    r={**s,**scores(truth,p,task), 'n_total':n,'n_test':len(p),'warmup':w,'init_s':init_s,'runtime_s':runtime,'latency_us':float(lat.mean()),'p95_us':float(np.percentile(lat,95)),'rss_baseline_mib':rss0/2**20,'rss_peak_mib':rsspeak/2**20,'rss_growth_mib':(rsspeak-rss0)/2**20,'alarms':alarms}
    if s['dataset'] in ['FriedmanShift','SEAReversal']:
        split=CFG['drift_index']-w;r.update({'pre_'+k:v for k,v in scores(truth[:split],p[:split],task).items()});r.update({'post_'+k:v for k,v in scores(truth[split:],p[split:],task).items()})
    if s['phase']=='automl':
        r['selection_info']=str(model.get_model_info()) if hasattr(model,'get_model_info') else str(model.best_model)
    return r


def clustering(s):
    random.seed(s['seed']);np.random.seed(s['seed'])
    from capymoa.cluster import Clustream_with_kmeans,ClusTree,Denstream_with_dbscan
    from capymoa.core import LabeledInstance
    from sklearn.metrics import silhouette_score,adjusted_rand_score
    from sklearn.metrics import pairwise_distances_argmin_min
    X,y=load(s['dataset'],s['seed']);schema=schema_for(X.shape[1]);w=CFG['warmup']
    factories={'CluStream':lambda:Clustream_with_kmeans(schema,max_num_kernels=30,k_option=3,time_window=1000),'ClusTree':lambda:ClusTree(schema,horizon=1000,max_height=3),'DenStream':lambda:Denstream_with_dbscan(schema,horizon=1000,epsilon=.6,beta=.2,mu=2,init_points=50,offline_option=2,lambda_option=.01,speed=100)}
    start=time.perf_counter();model=factories[s['model']]();init_s=time.perf_counter()-start
    if hasattr(model.moa_learner,'setRandomSeed'):model.moa_learner.setRandomSeed(s['seed']);model.moa_learner.resetLearning()
    for x in X[:w]:model.train(LabeledInstance.from_array(schema,x,0))
    rows=[];snapshots=[];proc=psutil.Process();rss0=proc.memory_info().rss;peak=rss0;begin=time.perf_counter();operations=0.
    # Test each block against a frozen model BEFORE training any observation in it.
    block=25 if s['dataset']=='Iris' else 250
    for start in range(w,len(X),block):
        end=min(start+block,len(X));xb=X[start:end];op=time.perf_counter()
        result=model.get_clustering_result() if model.implements_macro_clusters() else model.get_micro_clustering_result()
        centers=np.asarray(result.get_centers())
        valid=centers.ndim==2 and len(centers)>0
        if valid:
            labels,dist=pairwise_distances_argmin_min(xb,centers)
            error=float(np.mean(dist**2));k=len(np.unique(labels));ari=float(adjusted_rand_score(y[start:end],labels));sil=float(silhouette_score(xb,labels)) if 1<k<len(xb) else None
        else:error=ari=sil=None;k=0
        operations+=time.perf_counter()-op
        rows.append(dict(start=start,end=end,n_centers=len(centers),occupied=k,prequential_mse=error,ari=ari,silhouette=sil))
        if s['dataset']=='MovingBlobs' and (start in [50,2800,3050,5550]):snapshots.append(dict(start=start,end=end,centers=centers.tolist(),points=xb.tolist(),labels=y[start:end].tolist()))
        op=time.perf_counter()
        for x in xb:model.train(LabeledInstance.from_array(schema,x,0))
        operations+=time.perf_counter()-op;peak=max(peak,proc.memory_info().rss)
    runtime=time.perf_counter()-begin
    (RAW/(key(s)+'-windows.json')).write_text(json.dumps(rows,indent=2));(RAW/(key(s)+'-snapshots.json')).write_text(json.dumps(snapshots))
    def weighted(field):
        good=[r for r in rows if r[field] is not None]
        return float(np.average([r[field] for r in good],weights=[r['end']-r['start'] for r in good])) if good else None
    return {**s,'n_total':len(X),'n_test':len(X)-w,'init_s':init_s,'runtime_s':runtime,'latency_us':operations/(len(X)-w)*1e6,'rss_baseline_mib':rss0/2**20,'rss_peak_mib':peak/2**20,'rss_growth_mib':(peak-rss0)/2**20,'silhouette':weighted('silhouette'),'ari':weighted('ari'),'prequential_mse':weighted('prequential_mse'),'valid_blocks':sum(r['ari'] is not None for r in rows),'total_blocks':len(rows)}


def specs():
    out=[]
    for seed in CFG['seeds']:
        for task in ['regression','classification']:
            for ds in CFG[task]:
                for lib in (['River'] if task=='regression' else ['River','CapyMOA']):
                    for name in CFG['regressors' if task=='regression' else 'classifiers']:
                        out.append(dict(phase='main',task=task,library=lib,dataset=ds,model=name,seed=seed))
                for det in CFG[task+'_detectors']:out.append(dict(phase='reset',task=task,library='River',dataset=ds,model=det,seed=seed))
        for ds in CFG['classification']:
            for name in CFG['automl']:out.append(dict(phase='automl',task='classification',library='CapyMOA',dataset=ds,model=name,seed=seed))
        for ds in CFG['cluster_datasets']:
            for name in CFG['clusterers']:out.append(dict(phase='clustering',task='clustering',library='CapyMOA',dataset=ds,model=name,seed=seed))
    return out


def fingerprint():
    paths=[ROOT/'configs/experiment.json',ROOT/'results/data-manifest.json']+sorted((ROOT/'src').glob('*.py'))
    return hashlib.sha256(b''.join(p.read_bytes() for p in paths)).hexdigest()


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--worker');ap.add_argument('--phase');args=ap.parse_args();RAW.mkdir(exist_ok=True,parents=True)
    if args.worker:
        s=json.loads(args.worker);r=clustering(s) if s['phase']=='clustering' else supervised(s);r['status']='completed';r['fingerprint']=s.pop('fingerprint',None)
        (RAW/(key(s)+'.json')).write_text(json.dumps(r,indent=2,allow_nan=False));print(key(s), 'done',flush=True);return
    manifest=dict(platform=platform.platform(),cpu=subprocess.check_output(['sysctl','-n','machdep.cpu.brand_string'],text=True).strip(),logical_cpus=os.cpu_count(),ram_gib=psutil.virtual_memory().total/2**30,python=platform.python_version(),jvm_args=CFG['jvm_args'],threads=1,seeds=CFG['seeds'],started_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'),fingerprint=fingerprint())
    (ROOT/'results/run-manifest.json').write_text(json.dumps(manifest,indent=2))
    jobs=[s for s in specs() if not args.phase or s['phase']==args.phase]
    for i,s in enumerate(jobs,1):
        s['fingerprint']=manifest['fingerprint'];p=RAW/(key(s)+'.json')
        if p.exists() and json.loads(p.read_text()).get('fingerprint')==s['fingerprint']:continue
        print(f'[{i}/{len(jobs)}] {key(s)}',flush=True)
        log=RAW/(key(s)+'.log')
        with log.open('w') as f:subprocess.run([sys.executable,__file__,'--worker',json.dumps(s)],check=True,stdout=f,stderr=subprocess.STDOUT,cwd=ROOT)
    manifest['completed_at']=time.strftime('%Y-%m-%dT%H:%M:%S%z');manifest['jobs_completed']=len(jobs)
    (ROOT/'results/run-manifest.json').write_text(json.dumps(manifest,indent=2));print('ALL RUNS COMPLETE',len(jobs),flush=True)

if __name__=='__main__':main()
