"""Audit the training-selected refit without optimizing external/test scores."""
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from .io import save
from .explain import explain
from .external_dataset import source_data
from .optimize import base_oof,head,candidates
from .models import Ensemble,CONFIGS
from .metrics import metrics

def run(ds,root,out):
    root=Path(root);out=Path(out);selected=json.loads((out/'optimization.json').read_text())['selection']
    dest=out/'optimized_validation';dest.mkdir(parents=True,exist_ok=True)
    model=joblib.load(out/'optimized_model.joblib')
    explain(ds,model,dest)
    # Match exactly the original deduplicated transfer target IDs and API columns.
    X,y,names,meta=source_data(root/'data/external_sources')
    target=pd.read_csv(out/'elderan_transfer_predictions.csv');ix=meta.set_index('ID').index.get_indexer(target.source_id)
    cols=pd.read_csv(out/'elderan_common_apis.csv');groups=ds.metadata.set_index('sample_id').loc[ds.train.sample_id,'md5'].str.lower().to_numpy()
    z,folds,base=base_oof(ds.X[:,cols.mlran_index],ds.y,candidates()[selected['candidate']],groups)
    heads={n:head(z[:,j],ds.y,selected['C']) for n,j in CONFIGS.items()};m=Ensemble(base,heads,z,folds,42)
    p=m.probabilities(X[ix][:,cols.elderan_index]);pd.DataFrame({'source_id':target.source_id,'y_true':y[ix],**p}).to_csv(dest/'transfer_predictions.csv',index=False)
    save(dest/'transfer.json',{'models':{n:metrics(y[ix],v) for n,v in p.items()},'scope':'Same selected hyperparameters,refitted on shared56 source APIs; no external-label tuning; same898 hash-novel target executables as the original comparison'})
