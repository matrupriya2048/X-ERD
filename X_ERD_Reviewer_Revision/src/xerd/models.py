from dataclasses import dataclass
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

CONFIGS = {"RF+XGB":[0,1], "RF+LGB":[0,2], "XGB+LGB":[1,2], "X-ERD":[0,1,2]}
SEEDS = [0,1,2,3,4,5,42,123,256,999]

def bases(seed=42, jobs=4):
    return {
        "RF": RandomForestClassifier(n_estimators=100, criterion="gini", min_samples_split=2, random_state=seed, n_jobs=jobs),
        "XGB": XGBClassifier(n_estimators=100, learning_rate=.3, max_depth=6, objective="binary:logistic", reg_lambda=1., random_state=seed, n_jobs=jobs, eval_metric="logloss", verbosity=0),
        "LGB": LGBMClassifier(n_estimators=100, learning_rate=.1, num_leaves=31, boosting_type="gbdt", random_state=seed, n_jobs=jobs, verbose=-1)}

@dataclass
class Ensemble:
    base: dict
    meta: dict
    oof: np.ndarray
    inner_folds: np.ndarray
    seed: int

    def probabilities(self, X):
        z = np.column_stack([m.predict_proba(X)[:,1] for m in self.base.values()])
        result = {n:z[:,i] for i,n in enumerate(self.base)}
        result.update({n:m.predict_proba(z[:,CONFIGS[n]])[:,1] for n,m in self.meta.items()})
        return result

    def coefficients(self):
        return {n:{"coef":m.coef_[0],"intercept":m.intercept_[0]} for n,m in self.meta.items()}

def fit(X, y, seed=42, jobs=4):
    """Original StackingClassifier(cv=5) equivalent; fresh inner OOF on each call.

    Shared base fits across pairwise/full models only save compute: all original
    settings and the original unshuffled stratified inner folds are retained.
    """
    X=np.asarray(X)
    y=np.asarray(y)
    if set(np.unique(y)) != {0,1}: raise ValueError("Both true binary classes required")
    oof=np.full((len(y),3),np.nan)
    folds=np.full(len(y),-1)
    for fold,(a,b) in enumerate(StratifiedKFold(5,shuffle=False).split(X,y)):
        folds[b]=fold
        for j,m in enumerate(bases(seed,jobs).values()):
            m.fit(X[a],y[a]);oof[b,j]=m.predict_proba(X[b])[:,1]
    if not np.isfinite(oof).all() or (folds<0).any(): raise RuntimeError("Incomplete OOF predictions")
    meta={n:LogisticRegression(C=1.,solver="lbfgs",max_iter=1000,random_state=seed).fit(oof[:,cols],y)
          for n,cols in CONFIGS.items()}
    base=bases(seed,jobs)
    for m in base.values():m.fit(X,y)
    return Ensemble(base,meta,oof,folds,seed)
