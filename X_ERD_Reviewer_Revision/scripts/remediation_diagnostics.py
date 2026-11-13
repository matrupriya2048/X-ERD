"""Post-evaluation representation sensitivity, without changing the fitted model."""
from pathlib import Path
import json,hashlib
import numpy as np
import pandas as pd
from xerd.io import load,save
from xerd.external_raw import interval

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/remediation'

def fingerprints(X):
    return np.array([hashlib.sha256(np.packbits(row.astype(np.uint8)).tobytes()).hexdigest() for row in X])

def main():
    ds=load(ROOT/'data/mlran');dev=pd.read_csv(OUT/'development_vectors.csv');training=fingerprints(np.vstack([ds.X,dev[ds.columns].to_numpy()]))
    result={}
    for name,endpoint in [('holdout','fpr'),('ransomware','recall')]:
        vectors=pd.read_csv(OUT/(name+'_vectors.csv'));pred=pd.read_csv(OUT/(name+'_predictions.csv'));fp=fingerprints(vectors[ds.columns].to_numpy());novel=~np.isin(fp,training)
        row={'n':len(fp),'distinct_exact_vectors':len(set(fp)),'rows_identical_to_any_fit_vector':int((~novel).sum()),'novel_vector_rows':int(novel.sum()),'novel_vector_models':{}}
        for label in ['Original full stack','Previous passthrough','Development-adapted']:
            p=pred[label+'__X-ERD'].to_numpy();k=int((p[novel]>=.5).sum());n=int(novel.sum())
            row['novel_vector_models'][label]={endpoint:k/n if n else None,'events':k,'n':n,'ci95':interval(k,n)}
        result[name]=row
    save(OUT/'representation_diagnostics.json',{'scope':'Post-evaluation descriptive sensitivity. Primary executable-hash cohort unchanged; no vector-based outcome exclusion or model reselection. Exact-vector novelty does not prove application, family or environment independence. Binary intervals still do not capture all collection dependencies.','results':result})
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
