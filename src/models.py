"""Explicit model factories; cross-library differences are documented in protocol.md."""
from data import CFG

def river_model(task,name,seed):
    from river import tree,forest,ensemble,linear_model,optim,neighbors
    opts=dict(grace_period=CFG['tree_grace_period'],delta=CFG['tree_delta'],tau=CFG['tree_tau'])
    if task=='regression':
        if name=='HT':return tree.HoeffdingTreeRegressor(**opts,leaf_prediction='mean')
        if name=='ARF':return forest.ARFRegressor(n_models=CFG['ensemble_size'],seed=seed,leaf_prediction='mean',**opts)
        if name=='LinearAdam':return linear_model.LinearRegression(optimizer=optim.Adam(CFG['linear_learning_rate']),intercept_lr=.01)
        if name=='KNN':return neighbors.KNNRegressor(n_neighbors=CFG['knn_k'],engine=neighbors.LazySearch(window_size=CFG['knn_window']))
    else:
        if name=='HT':return tree.HoeffdingTreeClassifier(**opts,leaf_prediction='mc',merit_preprune=False)
        if name=='Bagging':return ensemble.ADWINBaggingClassifier(model=river_model(task,'HT',seed),n_models=CFG['ensemble_size'],seed=seed)
        if name=='Logistic':return linear_model.LogisticRegression(optimizer=optim.Adam(CFG['linear_learning_rate']),intercept_lr=.01,l2=0)
        if name=='KNN':return neighbors.KNNClassifier(n_neighbors=CFG['knn_k'],engine=neighbors.LazySearch(window_size=CFG['knn_window']),weighted=False)
    raise ValueError((task,name))

def schema_for(d):
    from capymoa.stream import Schema
    return Schema.from_custom(features=[f'x{i}' for i in range(d)]+['target'],target='target',categories={'target':['0','1']})

def capy_model(name,schema,seed):
    from capymoa.classifier import HoeffdingTree,OnlineAdwinBagging,SGDClassifier,KNN
    opts=dict(schema=schema,random_seed=seed,grace_period=CFG['tree_grace_period'],confidence=CFG['tree_delta'],tie_threshold=CFG['tree_tau'],leaf_prediction='MajorityClass',disable_prepruning=True)
    if name=='HT':return HoeffdingTree(**opts)
    if name=='Bagging':return OnlineAdwinBagging(schema=schema,random_seed=seed,base_learner=HoeffdingTree(**opts),ensemble_size=CFG['ensemble_size'],number_of_jobs=1)
    if name=='Logistic':return SGDClassifier(schema=schema,random_seed=seed,loss='log_loss',penalty=None,learning_rate='invscaling',eta0=CFG['linear_learning_rate'],n_jobs=1)
    if name=='KNN':return KNN(schema=schema,random_seed=seed,k=CFG['knn_k'],window_size=CFG['knn_window'])
    if name in ['Bandit','SuccessiveHalving']:
        from capymoa.automl import BanditClassifier,SuccessiveHalvingClassifier,EpsilonGreedy
        pool=[capy_model(n,schema,seed) for n in CFG['classifiers']]
        if name=='Bandit':return BanditClassifier(schema=schema,random_seed=seed,base_classifiers=pool,policy=EpsilonGreedy(epsilon=.1,burn_in=100))
        return SuccessiveHalvingClassifier(schema=schema,random_seed=seed,base_classifiers=pool,budget=12000,eta=2,min_models=1)
    raise ValueError(name)

def detector(name,seed):
    from river import drift
    if name=='ADWIN':return drift.ADWIN(delta=.002)
    if name=='PageHinkley':return drift.PageHinkley(min_instances=30,delta=.005,threshold=50,alpha=.9999,mode='both')
    if name=='KSWIN':return drift.KSWIN(alpha=.005,window_size=100,stat_size=30,seed=seed)
    if name=='DDM':return drift.binary.DDM(warm_start=30,warning_threshold=2,drift_threshold=3)
    if name=='EDDM':return drift.binary.EDDM(warm_start=30,alpha=.95,beta=.9)
    raise ValueError(name)
