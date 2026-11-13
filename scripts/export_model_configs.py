from pathlib import Path
import sys,json,os
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root/'src'))
os.environ['JAVA_HOME']=str(Path(sys.prefix)/'lib/jvm');os.environ['CAPYMOA_JVM_ARGS']='-Xmx2g -Xss10M'
from data import CFG
from models import river_model,capy_model,schema_for,detector
out={}
for task in ['regression','classification']:
 for name in CFG['regressors' if task=='regression' else 'classifiers']:
  m=river_model(task,name,42);out['River/'+task+'/'+name]=str(m._get_params())
for name in CFG['classifiers']+CFG['automl']:
 m=capy_model(name,schema_for(3),42)
 out['CapyMOA/classification/'+name]={'params':str(m.get_params()),'moa_cli':str(m.moa_learner.getOptions().getAsCLIString()) if hasattr(m,'moa_learner') else None}
for name in set(CFG['regression_detectors']+CFG['classification_detectors']):
 m=detector(name,42);out['detector/'+name]=str(m._get_params())
(root/'results/model-configurations.json').write_text(json.dumps(out,indent=2))
