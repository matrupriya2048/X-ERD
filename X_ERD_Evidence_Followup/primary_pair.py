"""RF+LightGBM primary inference; the XGBoost estimator is never invoked."""
from pathlib import Path
import argparse,json
import joblib
import numpy as np
import pandas as pd
from xerd.io import load,digest

def probabilities(artifact,X):
    X=np.asarray(X)
    if X.ndim!=2 or X.shape[1]!=len(artifact['feature_ids']):raise ValueError('Feature width/order mismatch')
    if not np.isfinite(X).all() or not np.isin(X,[0,1]).all():raise ValueError('Expected finite binary indicators')
    z=np.column_stack([artifact['base'][name].predict_proba(X)[:,1] for name in ['RF','LGB']])
    if artifact.get('passthrough',False):z=np.column_stack([z,X])
    return artifact['meta'].predict_proba(z)[:,1]

def export(root,destination):
    root=Path(root);ds=load(root/'data/mlran');source=root/'results/model.joblib';m=joblib.load(source)
    artifact={'model':'RF+LGB','base':{k:m.base[k] for k in ['RF','LGB']},'meta':m.meta['RF+LGB'],
              'feature_ids':ds.columns,'threshold':0.5,'source_model_sha256':digest(source),
              'decision':'Reviewer-requested simpler primary configuration; existing fit retained; no target-driven hyperparameter selection.'}
    joblib.dump(artifact,destination)
    assert np.allclose(probabilities(artifact,ds.Xt),m.probabilities(ds.Xt)['RF+LGB'],rtol=0,atol=1e-12)
    return artifact

if __name__=='__main__':
    parser=argparse.ArgumentParser();sub=parser.add_subparsers(dest='command',required=True)
    e=sub.add_parser('export');e.add_argument('--root',type=Path,required=True);e.add_argument('--output',type=Path,required=True)
    p=sub.add_parser('predict');p.add_argument('--model',type=Path,required=True);p.add_argument('--vectors',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    q=sub.add_parser('predict-report');q.add_argument('--model',type=Path,required=True);q.add_argument('--report',type=Path,required=True);q.add_argument('--feature-names',type=Path,required=True);q.add_argument('--output',type=Path,required=True)
    a=parser.parse_args()
    if a.command=='export':export(a.root,a.output)
    elif a.command=='predict':
        model=joblib.load(a.model);frame=pd.read_csv(a.vectors)
        ids=model['feature_ids'];available=[x for x in frame if x.isdigit()]
        if available!=ids:raise ValueError('CSV feature identifiers/order differ from trained artifact')
        probability=probabilities(model,frame[ids].to_numpy())
        pd.DataFrame({'probability':probability,'detected':probability>=model['threshold']}).to_csv(a.output,index=False)
    else:
        from xerd.parser import vector
        from cape_adapter import adapt
        report=json.loads(a.report.read_text(encoding='utf-8'));report,receipt=adapt(report)
        behavior=report.get('behavior') or {}
        if not any(p.get('calls') for p in behavior.get('processes',[])) and not behavior.get('apistats'):raise ValueError('Unprocessed report: no observed API activity')
        model=joblib.load(a.model);lookup=json.loads(a.feature_names.read_text(encoding='utf-8'))
        names=[lookup[k] for k in model['feature_ids']]
        x=vector(report,names);p=float(probabilities(model,x[None,:])[0])
        a.output.write_text(json.dumps({'probability':p,'detected':p>=model['threshold'],'model':model['model'],'active_feature_n':int(x.sum()),'input_schema':receipt,'scope':'Full-run classifier output, not an early-encryption or forensic technique confirmation.'},indent=2))
