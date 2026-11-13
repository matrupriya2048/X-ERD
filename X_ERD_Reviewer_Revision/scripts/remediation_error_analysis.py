"""Explain remaining false alerts without using them to tune the frozen model."""
from pathlib import Path
import json,warnings
import joblib,shap
import numpy as np
import pandas as pd
from scipy.special import expit
from xerd.io import load,save,digest
from xerd.explain import positive_values

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/remediation'

def main():
    warnings.filterwarnings('ignore',message='X does not have valid feature names')
    ds=load(ROOT/'data/mlran');frozen=json.loads((OUT/'frozen_model.json').read_text());assert digest(OUT/'model.joblib')==frozen['model_sha256']
    m=joblib.load(OUT/'model.joblib');p=pd.read_csv(OUT/'holdout_predictions.csv',float_precision='round_trip');v=pd.read_csv(OUT/'holdout_vectors.csv');audit={r['source_path']:r for r in json.loads((OUT/'holdout_audit.json').read_text())}
    ix=np.where(p['Development-adapted__X-ERD'].to_numpy()>=.5)[0];X=v[ds.columns].to_numpy()[ix];dev=pd.read_csv(OUT/'development_vectors.csv');train=np.vstack([ds.X,dev[ds.columns].to_numpy()])
    with np.load(OUT/'shap_values.npz') as z:background=train[z['background_indices']]
    values={};expected={}
    for name,model in m.base.items():
        ex=shap.TreeExplainer(model,data=background,model_output='probability',feature_perturbation='interventional');values[name],expected[name]=positive_values(ex,X)
    w=m.meta['X-ERD'].coef_[0];phi=sum(w[i]*values[n] for i,n in enumerate(m.base));baseline=float(m.meta['X-ERD'].intercept_[0]+sum(w[i]*expected[n] for i,n in enumerate(m.base)))
    if m.passthrough:phi+=w[3:]*(X-background.mean(0));baseline+=float(w[3:]@background.mean(0))
    error=float(abs(expit(baseline+phi.sum(1))-p['Development-adapted__X-ERD'].to_numpy()[ix]).max());assert error<1e-4
    rows=[]
    for j,k in enumerate(ix):
        path=p.source_path.iloc[k];a=audit[path];order=np.argsort(-phi[j]);active=[int(i) for i in order if X[j,i]>0 and phi[j,i]>0][:8]
        rows.append({'source_path':path,'filename':a.get('filename'),'md5':a.get('md5'),'publisher_label':'benign','adapted_probability':float(p['Development-adapted__X-ERD'].iloc[k]),'previous_probability':float(p['Previous passthrough__X-ERD'].iloc[k]),'active_features':int(X[j].sum()),'positive_active_contributions':[{'feature_id':ds.columns[i],'feature_name':ds.names[i],'logit_contribution':float(phi[j,i])} for i in active]})
    save(OUT/'remaining_false_alerts.json',{'model_sha256':frozen['model_sha256'],'n':len(rows),'maximum_probability_reconstruction_error':error,'scope':'Post-hoc error inspection only. Publisher benign labels are retained without fresh adjudication. Contributions explain this fitted model, not malicious intent or causal error mechanisms. No thresholds, rules, labels or weights are changed after inspecting these cases.','cases':rows})
    print('Explained remaining false alerts',len(rows),'maximum probability error',error)

if __name__=='__main__':main()
