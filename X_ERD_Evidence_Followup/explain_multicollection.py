from pathlib import Path
import json,joblib
import numpy as np
import pandas as pd
import shap
from scipy.special import expit
from xerd.io import load,save,digest
from xerd.explain import positive_values
from xerd.mapping import map_features
from primary_pair import probabilities
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent/'X_ERD_Reviewer_Revision';OUT=HERE/'multicollection'
ds=load(ROOT/'data/mlran');m=joblib.load(OUT/'model.joblib');protocol=json.loads((OUT/'protocol.json').read_text())
en=pd.read_csv(HERE/'closure_results/en2025_translated_vectors.csv').set_index('source_path');dev=pd.read_csv(ROOT/'results/remediation/development_vectors.csv')
train=np.vstack([ds.X,dev[ds.columns].to_numpy(),en.loc[protocol['train_paths'],ds.columns].to_numpy()])
rng=np.random.default_rng(42);idx=rng.choice(len(train),100,replace=False);bg=train[idx];bgmean=bg.mean(0)
target=en.loc[protocol['test_paths'],ds.columns].to_numpy();X=np.vstack([ds.Xt,target]);vals=[];expected=[]
for name in ['RF','LGB']:
    print('Explaining final pair',name,flush=True)
    ex=shap.TreeExplainer(m['base'][name],data=bg,model_output='probability',feature_perturbation='interventional')
    v,b=positive_values(ex,X);assert abs(b+v.sum(1)-m['base'][name].predict_proba(X)[:,1]).max()<1e-4
    vals.append(v);expected.append(b)
coef=m['meta'].coef_[0];phi=sum(coef[j]*v for j,v in enumerate(vals))+(X-bgmean)*coef[2:]
base=float(m['meta'].intercept_[0]+coef[:2]@np.asarray(expected)+coef[2:]@bgmean)
p=probabilities(m,X);error=float(abs(expit(base+phi.sum(1))-p).max());assert error<1e-4
np.savez_compressed(OUT/'explanation_values.npz',contributions=phi,baseline=base,background_indices=idx)
sample=np.linspace(0,974,100,dtype=int)
globaldf=pd.DataFrame({'feature_id':ds.columns,'feature_name':ds.names,'mean_abs_logit_contribution':abs(phi[sample]).mean(0)}).sort_values('mean_abs_logit_contribution',ascending=False)
globaldf.to_csv(OUT/'global_attribution.csv',index=False)
mapping=map_features(ds.names,json.loads((ROOT/'config/mapping_schema.json').read_text()))
mapped=np.array([mapping[n]['technique_id'] is not None for n in ds.names]);positive=np.maximum(phi[:975],0)*(ds.Xt>0);sel=(ds.yt==1)&(p[:975]>=.5)
mass=positive[sel][:,mapped].sum(1)/np.maximum(positive[sel].sum(1),1e-15)
save(OUT/'explanations.json',{'model_sha256':digest(OUT/'model.joblib'),'scope':'Final multi-collection pair logit; probability-scale common-background TreeSHAP plus exact centered raw linear terms. Not causal, forensic, or human-verified.',
    'source_test_n':975,'en_holdout_n':len(target),'max_probability_reconstruction_error':error,'background_n':100,
    'source_detected_ransomware_n':int(sel.sum()),'mapped_positive_active_logit_mass_mean':float(mass.mean()),
    'mapped_feature_n':int(mapped.sum()),'calibrated_mapping_accuracy':False,'script_sha256':digest(Path(__file__))})
rows=[]
for i in np.flatnonzero(p[975:]>=.5):
    order=np.argsort(-phi[975+i])[:15]
    for j in order:rows.append({'source_path':protocol['test_paths'][i],'probability':float(p[975+i]),'feature_id':ds.columns[j],'feature_name':ds.names[j],'active':int(target[i,j]),'logit_contribution':float(phi[975+i,j])})
pd.DataFrame(rows).to_csv(OUT/'holdout_false_alarm_attributions.csv',index=False)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
d=globaldf.head(14).iloc[::-1];f,a=plt.subplots(figsize=(9,4.7));a.barh(d.feature_id.astype(str),d.mean_abs_logit_contribution,color='#26664c');a.set_xlabel('Mean absolute final-pair logit contribution');a.set_ylabel('Feature ID (full names in accompanying CSV)');a.set_title('Multi-collection pair: fixed 100-record source sample');f.tight_layout();f.savefig(OUT/'global_attribution.png',dpi=220);plt.close(f)
print('Final-pair explanation identity maximum error',error,flush=True)
