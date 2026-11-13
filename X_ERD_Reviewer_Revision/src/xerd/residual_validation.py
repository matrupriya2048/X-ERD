"""Exact logit attribution and external transfer for the passthrough extension."""
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from scipy.special import expit
from .io import save
from .models import CONFIGS
from .residual import ResidualEnsemble
from .optimize import base_oof,head,candidates
from .external_dataset import source_data
from .explain import deletion,targeted
from .metrics import metrics

def run(ds,root,out):
    root=Path(root);out=Path(out);dest=out/'residual_validation';dest.mkdir(exist_ok=True)
    model=joblib.load(out/'residual_model.joblib');selection=json.loads((out/'residual.json').read_text())['selection']
    original=pd.read_csv(root/'results/test_predictions.csv')
    # Reuse base explanations only after verifying all three fitted base outputs.
    for name,base in model.base.items():
        if not np.allclose(base.predict_proba(ds.Xt)[:,1],original[name],atol=1e-12,rtol=0):
            raise ValueError('Different base fit: recompute base SHAP before attribution')
    with np.load(root/'results/shap_values.npz') as z:
        values={n:z[n] for n in model.base};expected=z['expected_values'];bg=z['background_ids']
    mean=ds.train.set_index('sample_id').loc[bg,ds.columns].to_numpy().mean(0)
    w=model.meta['X-ERD'].coef_[0];b=float(model.meta['X-ERD'].intercept_[0])
    raw=(ds.Xt-mean)*w[3:];trees=sum(w[i]*values[n] for i,n in enumerate(model.base))
    phi=raw+trees;baseline=b+float(w[:3]@expected+w[3:]@mean)
    probability=model.probabilities(ds.Xt)['X-ERD'];err=float(np.max(abs(expit(baseline+phi.sum(1))-probability)))
    if err>1e-4:raise AssertionError('Passthrough stack reconstruction failed')
    np.savez_compressed(dest/'shap_values.npz',stack_logit=phi,raw_component=raw,tree_component=trees,baseline=baseline,background_ids=bg)
    pd.DataFrame({'feature':ds.names,'mean_abs_total':abs(phi).mean(0),'mean_abs_raw':abs(raw).mean(0),'mean_abs_tree':abs(trees).mean(0)}).sort_values('mean_abs_total',ascending=False).to_csv(dest/'global_shap.csv',index=False)
    save(dest/'explanations.json',{'equation':'Phi_j = beta_j*(x_j-E_background[x_j]) + sum_k w_k*phi_base_kj',
      'scope':'exact additive interventional logit explanation; not full-stack probability-space SHAP or causal attribution',
      'maximum_probability_reconstruction_error':err,'base_weights':w[:3],'raw_feature_weights':w[3:],'intercept':b,'background_sample_ids':bg})
    deletion(ds,model,values,phi,sum(values.values())/3,dest);targeted(ds,model,values['LGB'],phi,dest)
    X,y,names,meta=source_data(root/'data/external_sources');target=pd.read_csv(out/'elderan_transfer_predictions.csv')
    ix=meta.set_index('ID').index.get_indexer(target.source_id);cols=pd.read_csv(out/'elderan_common_apis.csv')
    groups=ds.metadata.set_index('sample_id').loc[ds.train.sample_id,'md5'].str.lower().to_numpy();source=ds.X[:,cols.mlran_index]
    z,folds,base=base_oof(source,ds.y,candidates()[selection['candidate']],groups)
    heads={n:head(np.column_stack([z[:,j],source]),ds.y,selection['C']) for n,j in CONFIGS.items()}
    model=ResidualEnsemble(base,heads);p=model.probabilities(X[ix][:,cols.elderan_index])
    pd.DataFrame({'source_id':target.source_id,'y_true':y[ix],**p}).to_csv(dest/'transfer_predictions.csv',index=False)
    save(dest/'transfer.json',{'models':{n:metrics(y[ix],v) for n,v in p.items()},'scope':'Same source-trained hyperparameters,56 common APIs,898 deduplicated hash-novel target executables; no external tuning'})
