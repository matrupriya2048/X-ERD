"""Reviewer-directed primary pair and additional measured evidence; no fitting."""
from pathlib import Path
import json,hashlib
import numpy as np
import pandas as pd
import joblib
from scipy.special import expit
from xerd.io import load,save,digest,environment
from xerd.metrics import metrics
from xerd.external_raw import interval

HERE=Path(__file__).resolve().parent;ROOT=HERE.parent/'X_ERD_Reviewer_Revision';OUT=HERE/'closure_results'

def run():
    OUT.mkdir(exist_ok=True);ds=load(ROOT/'data/mlran');m=joblib.load(ROOT/'results/model.joblib')
    head=m.meta['RF+LGB'];z=np.load(ROOT/'results/shap_values.npz');w=head.coef_[0]
    phi=w[0]*z['RF']+w[1]*z['LGB'];base=float(head.intercept_[0]+w@z['expected_values'][[0,2]])
    probs=m.probabilities(ds.Xt)['RF+LGB'];error=float(abs(expit(base+phi.sum(1))-probs).max());assert error<1e-4
    np.savez_compressed(OUT/'primary_pair_shap.npz',logit_contributions=phi,baseline=base)
    ix=np.linspace(0,len(ds.yt)-1,100,dtype=int)
    pd.DataFrame({'feature_id':ds.columns,'feature_name':ds.names,'mean_abs_primary_pair_logit_contribution':abs(phi[ix]).mean(0)}).sort_values('mean_abs_primary_pair_logit_contribution',ascending=False).to_csv(OUT/'primary_pair_global_attribution.csv',index=False)
    pd.DataFrame({'sample_id':ds.test.sample_id,'y_true':ds.yt,'RF+LGB':probs}).to_csv(OUT/'primary_pair_official_predictions.csv',index=False)
    # Transparent lexical proxy, deliberately excluding generic file I/O.
    impact=np.array([any(s in n.lower() for s in ['encrypt','decrypt','cryptstringtobinary','crypt32.dll']) for n in ds.names])
    cohort=(ds.yt==1)&(probs>=.5);positive=np.maximum(phi,0)*(ds.Xt>0)
    mass=positive[:,impact].sum(1)/np.maximum(positive.sum(1),1e-15)
    Xm=ds.Xt.copy();Xm[:,impact]=0;masked=m.probabilities(Xm)['RF+LGB'];has=(ds.Xt[:,impact]>0).any(1)
    per=pd.DataFrame({'sample_id':ds.test.sample_id,'y_true':ds.yt,'detected':probs>=.5,'proxy_active':has,
                       'positive_active_logit_mass_fraction':mass,'original_probability':probs,'proxy_masked_probability':masked})
    per.to_csv(OUT/'impact_proxy_per_case.csv',index=False)
    proxy={'model':'original RF+LGB primary pair','features':[n for n,k in zip(ds.names,impact) if k],
           'definition':'Predeclared lexical encryption/decryption/cryptographic-library indicators. Ordinary NtWriteFile and generic file operations are excluded because they do not establish mass modification or encryption.',
           'ransomware_detected_n':int(cohort.sum()),'detected_with_active_proxy_n':int((cohort&has).sum()),
           'detected_with_active_proxy_fraction':float(has[cohort].mean()),
           'positive_active_logit_mass_mean':float(mass[cohort].mean()),'positive_active_logit_mass_median':float(np.median(mass[cohort])),
           'detections_lost_when_proxy_zeroed':int((cohort&(masked<.5)).sum()),
           'original_metrics':metrics(ds.yt,probs),'masked_metrics':metrics(ds.yt,masked),
           'scope':'Full-run indicator reliance and feature-space masking only. Strings and library presence are untimed. No feature is asserted to be post-encryption; these numbers do not bound true pre-encryption efficacy. Custom encryption can avoid these tokens.'}
    save(OUT/'impact_proxy.json',proxy)
    # Already held-out per-fold predictions; no retraining or seed selection.
    pre=pd.read_csv(ROOT/'results/extended/sequence_prefix_predictions.csv');rows=[]
    for (model,seed,window),g in pre.groupby(['model','seed','calls']):
        assert not g.source_file.duplicated().any()
        result=metrics(g.y_true.to_numpy(),g.probability.to_numpy());ben=g[g.y_true==0];ran=g[g.y_true==1]
        rows.append({'model':model,'seed':int(seed),'calls':str(window),'metrics':result,
                     'benign_n':len(ben),'benign_fp':int((ben.probability>=.5).sum()),
                     'fpr_ci95':interval(int((ben.probability>=.5).sum()),len(ben)),
                     'ransomware_n':len(ran),'ransomware_tp':int((ran.probability>=.5).sum())})
    save(OUT/'sequence_window_controls.json',{'source_sha256':digest(ROOT/'results/extended/sequence_prefix_predictions.csv'),
         'rows':rows,'scope':'Call-count windows at a fixed 0.5 threshold, same 20 benign and 60 ransomware traces per seed. Seeds are dependent repeated fits, not independent samples. Call counts are not elapsed time or verified onset.'})
    sources={
      'official':('results/test_predictions.csv','RF+LGB'),
      'strict_temporal':('results/temporal_strict_predictions.csv','RF+LGB'),
      'external_original_182':('results/extended/external_raw_predictions.csv','RF+LGB'),
      'new_same_collection_362_original':('results/remediation/holdout_predictions.csv','Original full stack__RF+LGB'),
      'new_same_collection_362_adapted':('results/remediation/holdout_predictions.csv','Development-adapted__RF+LGB'),
      'zirari_58_original':('results/remediation/ransomware_predictions.csv','Original full stack__RF+LGB'),
      'zirari_58_adapted':('results/remediation/ransomware_predictions.csv','Development-adapted__RF+LGB'),
      'external_shared_api':('results/extended/elderan_transfer_predictions.csv','RF+LGB')}
    results={}
    for label,(path,col) in sources.items():
        frame=pd.read_csv(ROOT/path);results[label]={'metrics':metrics(frame.y_true.to_numpy(),frame[col].to_numpy()),'source':path,'source_sha256':digest(ROOT/path),'column':col}
    # Fixed reviewer-requested pair, also scored on the already-inspected MTM cohort.
    X=pd.read_csv(HERE/'vectors.csv');pp=m.probabilities(X[ds.columns].to_numpy())['RF+LGB']
    results['mixed_file_142']={'metrics':metrics(np.zeros(len(X),int),pp),'scope':'Previously inspected mixed-file collection; no new prospective evidence.'}
    save(OUT/'primary_pair_results.json',{'primary':'RF+LGB','reason':'Reviewer 1.2 explicitly permits adopting RF+LightGBM; this is a disclosed revision decision, not a pre-evaluation model-selection result.',
       'training_changed':False,'model_sha256':digest(ROOT/'results/model.joblib'),'head_coefficients':w,'head_intercept':head.intercept_[0],
       'maximum_probability_reconstruction_error':error,'results':results,'environment':environment(),
       'source_sha256':digest(Path(__file__))})
    print(json.dumps({'primary_official':results['official']['metrics'],'proxy':proxy,'windows':len(rows),'reconstruction_error':error},indent=2))

if __name__=='__main__':run()
