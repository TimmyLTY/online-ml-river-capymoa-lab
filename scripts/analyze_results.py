"""Derive all tables/figures from completed runs, retaining missing detections."""
from pathlib import Path
import sys,json
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from data import ROOT,CFG,load
from models import detector
from run_experiments import specs,key
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

RAW=ROOT/'results/raw';FIG=ROOT/'results/generated/figures';TAB=ROOT/'results/tables';TEX=ROOT/'results/generated/tables'
for folder in [FIG,TAB,TEX]:folder.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'font.size':9,'axes.titlesize':10,'axes.labelsize':9,'legend.fontsize':8,'pdf.fonttype':42,'ps.fonttype':42,'savefig.facecolor':'white','axes.spines.top':False,'axes.spines.right':False})
COLORS=['#0072B2','#D55E00','#009E73','#CC79A7'];LINES=['-','--','-.',':']


def save(fig,name):
    fig.savefig(FIG/(name+'.pdf'));plt.close(fig)


def fmt(v,d=2):return '--' if pd.isna(v) else f'{v:.{d}f}'

def pm(g,m,d=2,factor=1):
    a=g[m].astype(float)*factor
    if not a.notna().any(): return '--'
    return f'{a.mean():.{d}f}'

def latex_table(name,headers,rows,align=None):
    out='\\begin{tabular}{'+(align or 'l'+'r'*(len(headers)-1))+'}\n\\toprule\n'+' & '.join(headers)+' \\\\\n\\midrule\n'
    out+='\n'.join(' & '.join(str(v) for v in row)+' \\\\' for row in rows)
    out+='\n\\bottomrule\n\\end{tabular}\n';(TEX/(name+'.tex')).write_text(out)


def main():
    expected=specs();records=[]
    for s in expected:
        p=RAW/(key(s)+'.json')
        if not p.exists():raise RuntimeError(f'Missing run {p.name}')
        r=json.loads(p.read_text());assert r['status']=='completed';records.append(r)
    df=pd.DataFrame(records);df.drop(columns=['alarms'],errors='ignore').to_csv(TAB/'runs.csv',index=False)
    detection=[]
    for s in expected:
        if s['phase']!='main' or s['library']!='River' or s['model']!='HT':continue
        z=np.load(RAW/(key(s)+'.npz'));y,p,indices=z['y'],z['pred'],z['index'];_,all_y=load(s['dataset'],s['seed'])
        scale=max(np.std(all_y[:CFG['warmup']]),1e-12)
        errors=np.minimum(np.abs(y-p)/(3*scale),1) if s['task']=='regression' else (y!=p).astype(int)
        for name in CFG[s['task']+'_detectors']:
            d=detector(name,s['seed']);alarms=[]
            for index,error in zip(indices,errors):
                d.update(float(error))
                if d.drift_detected:alarms.append(int(index));d=detector(name,s['seed'])
            synthetic=s['dataset'] in ['FriedmanShift','SEAReversal'];after=[a for a in alarms if a>=CFG['drift_index']]
            detection.append(dict(task=s['task'],dataset=s['dataset'],seed=s['seed'],detector=name,n_alarms=len(alarms),prechange_alarms=sum(a<CFG['drift_index'] for a in alarms) if synthetic else None,delay=after[0]-CFG['drift_index'] if synthetic and after else None,detected=bool(after) if synthetic else None,positions=alarms))
    (TAB/'detector-alarms.json').write_text(json.dumps(detection,indent=2));dd=pd.DataFrame(detection);dd.drop(columns='positions').to_csv(TAB/'detectors.csv',index=False)
    for task in ['regression','classification']:
        models=CFG['regressors' if task=='regression' else 'classifiers'];rows=[]
        for ds in CFG[task]:
            for m in models:
                g=df.query('phase=="main" and library=="River" and dataset==@ds and model==@m')
                metrics=[pm(g,v,2,100 if v in ['accuracy','f1'] else 1) for v in (['mae','rmse','r2'] if task=='regression' else ['accuracy','f1','kappa'])]
                rows.append([ds,m,*metrics,fmt(g.latency_us.mean(),1),fmt(g.runtime_s.mean(),3),fmt(g.rss_peak_mib.mean(),0)])
        latex_table(task,['Dataset','Model']+(['MAE','RMSE','$R^2$'] if task=='regression' else ['Acc. (\\%)','Macro-F1 (\\%)','$\\kappa$'])+['$\\mu$s/ex.','Time (s)','RSS (MiB)'],rows,'llrrrrrr')
        # Cumulative curves use the configured fixed seed; optional bands apply only if more seeds are configured.
        fig,axes=plt.subplots(2,2,figsize=(7,3.35),layout='constrained')
        for ax,ds in zip(axes.flat,CFG[task]):
            for mi,m in enumerate(models):
                curves=[]
                for seed in CFG['seeds']:
                    z=np.load(RAW/f'main-River-{ds}-{m}-{seed}.npz');loss=(z['pred']-z['y'])**2 if task=='regression' else (z['pred']==z['y'])
                    c=np.cumsum(loss)/np.arange(1,len(loss)+1);curves.append(np.sqrt(c) if task=='regression' else c*100)
                a=np.asarray(curves);stride=max(1,a.shape[1]//200);x=z['index'][::stride];mean=a.mean(0)[::stride];sd=a.std(0,ddof=0)[::stride]
                ax.plot(x,mean,color=COLORS[mi],ls=LINES[mi],label=m,lw=1.1);ax.fill_between(x,mean-sd,mean+sd,color=COLORS[mi],alpha=.1) if len(curves)>1 else None
            if ds in ['FriedmanShift','SEAReversal']:ax.axvline(5000,color='0.4',ls=':',lw=1)
            ax.set(title=ds,xlabel='Stream index',ylabel='Cumulative RMSE' if task=='regression' else 'Cumulative accuracy (%)');ax.grid(alpha=.15)
        axes[0,0].legend(ncol=2);save(fig,task+'-learning')
        # Drift counts and reset comparison.
        rows=[]
        for ds in CFG[task]:
            base=df.query('phase=="main" and library=="River" and dataset==@ds and model=="HT"').set_index('seed')
            for detname in CFG[task+'_detectors']:
                g=dd.query('dataset==@ds and detector==@detname');reset=df.query('phase=="reset" and dataset==@ds and model==@detname').set_index('seed')
                metric='rmse' if task=='regression' else 'accuracy'
                change=(base[metric]-reset[metric])/base[metric]*100 if task=='regression' else (reset[metric]-base[metric])*100
                post='post_'+metric
                postval=fmt(reset[post].mean(),2) if ds in ['FriedmanShift','SEAReversal'] else '--'
                if task=='classification' and postval!='--':postval=fmt(reset[post].mean()*100,2)
                positions=g.iloc[0].positions
                extent='--' if not positions else str(positions[0]) if len(positions)==1 else f'{positions[0]}--{positions[-1]}'
                rows.append([ds,detname,pm(g,'n_alarms',0),extent,fmt(g.prechange_alarms.mean(),0),('Yes' if g.detected.any() else 'No') if g.detected.notna().any() else '--',fmt(g.delay.mean(),0),fmt(change.mean(),2),postval])
        latex_table(task+'-drift',['Dataset','Detector','Alarms','First/last','Pre','Hit','Delay','$\\Delta$ RMSE (\\%)' if task=='regression' else '$\\Delta$ acc. (pp)','Post metric'],rows,'llrlrrrrr')
        ds='FriedmanShift' if task=='regression' else 'SEAReversal';seed=42
        fig,axes=plt.subplots(2,1,figsize=(7,3.6),layout='constrained',height_ratios=[2,1],sharex=True)
        for mi,name in enumerate(['None']+CFG[task+'_detectors']):
            filename=f'main-River-{ds}-HT-{seed}' if name=='None' else f'reset-River-{ds}-{name}-{seed}'
            z=np.load(RAW/(filename+'.npz'));error=(z['pred']-z['y'])**2 if task=='regression' else (z['pred']==z['y']).astype(float)*100
            rolling=pd.Series(error).rolling(250,min_periods=250).mean().to_numpy();rolling=np.sqrt(rolling) if task=='regression' else rolling
            axes[0].plot(z['index'][::25],rolling[::25],label=('No reset' if name=='None' else name),color=COLORS[mi],ls=LINES[mi],lw=1.2)
        for i,detname in enumerate(CFG[task+'_detectors']):
            alarms=dd.query('dataset==@ds and seed==42 and detector==@detname').iloc[0].positions
            axes[1].scatter(alarms,[i]*len(alarms),marker='|',s=90,color=COLORS[i+1])
        axes[1].set(yticks=range(3),yticklabels=CFG[task+'_detectors'],xlabel='Stream index (zero-based)');axes[0].set(ylabel='Rolling RMSE (250)' if task=='regression' else 'Rolling accuracy (%)',title=ds+'; seed 42');axes[0].legend(ncol=4)
        for ax in axes:ax.axvline(5000,color='black',ls=':',lw=1);ax.grid(alpha=.15)
        save(fig,task+'-drift')
    rows=[]
    for ds in CFG['classification']:
        for m in CFG['classifiers']:
            r=df.query('phase=="main" and dataset==@ds and model==@m and library=="River"').set_index('seed');c=df.query('phase=="main" and dataset==@ds and model==@m and library=="CapyMOA"').set_index('seed')
            rows.append([ds,m,pm(r,'accuracy',2,100),pm(c,'accuracy',2,100),fmt((c.accuracy-r.accuracy).mean()*100,2),fmt(r.latency_us.mean(),1),fmt(c.latency_us.mean(),1),fmt(r.latency_us.mean()/c.latency_us.mean(),2),fmt(r.runtime_s.mean(),3),fmt(c.runtime_s.mean(),3)])
    latex_table('cross-library',['Dataset','Model','River acc.','Capy acc.','$\\Delta$ pp','River $\\mu$s','Capy $\\mu$s','R/C','River s','Capy s'],rows,'llrrrrrrrr')
    fig,axes=plt.subplots(2,2,figsize=(7,3.15),layout='constrained')
    for ax,ds in zip(axes.flat,CFG['classification']):
        for mi,m in enumerate(CFG['classifiers']):
            for lib,marker in [('River','o'),('CapyMOA','^')]:
                g=df.query('phase=="main" and dataset==@ds and model==@m and library==@lib')
                ax.scatter(g.latency_us.mean(),100*g.accuracy.mean(),color=COLORS[mi],marker=marker,s=40,label=m+' '+lib)
        ax.set(xscale='log',xlim=(3,700),ylim=(45,100),xticks=[10,100],title=ds,xlabel='Latency (microseconds, log scale)',ylabel='Accuracy (%)');ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter());ax.grid(alpha=.2)
    handles,labels=axes[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='outside lower center',ncol=4,fontsize=7);save(fig,'cross-library')
    rows=[]
    for ds in CFG['cluster_datasets']:
        for name in CFG['clusterers']:
            g=df.query('phase=="clustering" and dataset==@ds and model==@name')
            rows.append([ds,name,pm(g,'silhouette',3),pm(g,'ari',3),pm(g,'prequential_mse',3),fmt(g.runtime_s.mean(),3),fmt(g.latency_us.mean(),1),fmt(g.rss_peak_mib.mean(),0)])
    latex_table('clustering',['Dataset','Method','Silhouette','ARI','Dist.$^2$','Time(s)','$\\mu$s/ex.','RSS'],rows,'llrrrrrr')
    fig,axes=plt.subplots(1,3,figsize=(7,2.05),layout='constrained')
    for mi,name in enumerate(CFG['clusterers']):
        r=pd.read_json(RAW/f'clustering-CapyMOA-MovingBlobs-{name}-42-windows.json')
        for ax,metric,title in zip(axes,['prequential_mse','ari','n_centers'],['Prequential squared distance','ARI (held-out block labels)','Number of centers']):
            ax.plot(r.start,r[metric],label=('No reset' if name=='None' else name),color=COLORS[mi],ls=LINES[mi],marker=['o','s','^'][mi],ms=2,lw=1);ax.axvline(3000,color='0.4',ls=':');ax.set(title=title,xlabel='Block start');ax.grid(alpha=.15)
    axes[0].legend(fontsize=7);save(fig,'cluster-evolution')
    fig,axes=plt.subplots(1,3,figsize=(7,1.9),layout='constrained',sharex=True,sharey=True)
    r=json.loads((RAW/'clustering-CapyMOA-MovingBlobs-CluStream-42-snapshots.json').read_text())
    for ax,point in zip(axes,[r[1],r[2],r[3]]):
        x=np.array(point['points']);c=np.array(point['centers']);labels=np.array(point['labels'])
        for label in range(3):ax.scatter(*x[labels==label].T,s=5,color=COLORS[label],alpha=.55,marker=['o','s','^'][label])
        if c.ndim==2:ax.scatter(*c.T,marker='X',s=65,color='black',edgecolor='white',linewidth=.5)
        ax.set(title=f"Block {point['start']}–{point['end']-1}",xlabel='Standardized x0')
    axes[0].set_ylabel('Standardized x1');save(fig,'cluster-snapshots')
    rows=[]
    for ds in CFG['classification']:
        static=df.query('phase=="main" and library=="CapyMOA" and dataset==@ds');best=static.groupby('model').accuracy.mean().idxmax()
        for name in [best]+CFG['automl']:
            g=df.query('dataset==@ds and library=="CapyMOA" and model==@name')
            rows.append([ds,name+(' (best static)' if name==best else ''),pm(g,'accuracy',2,100),pm(g,'f1',2,100),fmt(g.runtime_s.mean(),3),fmt(g.latency_us.mean(),1),fmt(g.rss_peak_mib.mean(),0)])
    latex_table('automl',['Dataset','Method','Accuracy (\\%)','Macro-F1 (\\%)','Time (s)','$\\mu$s/ex.','RSS'],rows,'llrrrrr')
    fig,axes=plt.subplots(1,2,figsize=(7,2.5),layout='constrained')
    for mi,name in enumerate(['HT','Bagging','Bandit','SuccessiveHalving']):
        phase='main' if mi<2 else 'automl';z=np.load(RAW/f'{phase}-CapyMOA-SEAReversal-{name}-42.npz');a=(z['y']==z['pred']).astype(float)*100
        axes[0].plot(z['index'][::25],pd.Series(a).rolling(250,min_periods=250).mean().to_numpy()[::25],label=name,ls=LINES[mi],color=COLORS[mi],lw=1.1)
    axes[0].axvline(5000,color='0.4',ls=':');axes[0].set(xlabel='Stream index',ylabel='Rolling accuracy (%)',title='SEAReversal; seed 42');axes[0].legend(fontsize=7,ncol=2)
    for mi,name in enumerate(CFG['automl']):
        g=df.query('phase=="automl" and model==@name').groupby('dataset').latency_us.mean().reindex(CFG['classification']);axes[1].plot(range(4),g,marker=['o','s'][mi],color=COLORS[mi],label=name)
    axes[1].set(xticks=range(4),xticklabels=CFG['classification'],ylabel='Mean latency (microseconds)',yscale='log',title='AutoML cost');axes[1].tick_params(axis='x',labelrotation=20);axes[1].legend(fontsize=7)
    save(fig,'automl')
    print('Analyzed',len(df),'runs;',len(dd),'detector replays; figures and tables generated.')

if __name__=='__main__':main()
