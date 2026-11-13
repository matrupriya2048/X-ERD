"""Exploratory paired external endpoint comparisons; report all declared pairs."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.stats import binomtest
from xerd.io import save
from xerd.metrics import holm
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/extended'

def main():
    new=pd.read_csv(OUT/'residual_validation/transfer_predictions.csv').set_index('source_id')
    baseline=pd.read_csv(OUT/'elderan_transfer_predictions.csv').set_index('source_id')
    pub=pd.read_csv(OUT/'external_comparators_predictions.csv').set_index('source_id').loc[new.index]
    rivals={'Original full':baseline.loc[new.index,'X-ERD'],**{c:pub[c] for c in pub if c!='y_true'}}
    rows=[];rng=np.random.default_rng(42)
    for name,old in rivals.items():
        for cls,endpoint in [(0,'FPR'),(1,'Recall')]:
            ix=new.y_true.to_numpy()==cls;a=old.to_numpy()[ix]>=.5;b=new['X-ERD'].to_numpy()[ix]>=.5
            discord=[int((a&~b).sum()),int((~a&b).sum())];delta=[]
            for _ in range(2000):
                z=rng.integers(0,len(a),len(a));delta.append(float(b[z].mean()-a[z].mean()))
            rows.append({'comparator':name,'endpoint':endpoint,'n':len(a),'new_minus_comparator':float(b.mean()-a.mean()),
              'paired_bootstrap_ci95':np.quantile(delta,[.025,.975]),'discordant_old_only_new_only':discord,
              'exact_p':float(binomtest(discord[0],sum(discord),.5).pvalue) if sum(discord) else 1.})
    for endpoint in ['FPR','Recall']:
        group=[r for r in rows if r['endpoint']==endpoint]
        for r,p in zip(group,holm([r['exact_p'] for r in group])):r['holm_p']=float(p)
    meta=pd.read_csv(ROOT/'data/external_sources/elderan_ids.txt',sep=';').set_index('ID').loc[new.index]
    save(OUT/'external_paired_comparisons.json',{'rows':rows,'target_ransomware_family_counts':meta.loc[new.y_true==1,'Ransomware_Family'].value_counts().to_dict(),
      'scope':'Exploratory post-hoc paired comparisons on reused external records; five comparisons per endpoint family,Holm correction. Binary-distinct records can still share application or collection dependencies. Recall n30 is limited.'})
    import json
    raw=json.loads((OUT/'external_raw.json').read_text())
    prevalence=pd.read_csv(ROOT/'results/feature_prevalence.csv',float_precision='round_trip')
    shifts=[]
    for group,v in raw['categories'].items():
        selected=prevalence.feature_name.str.startswith(group+':')
        shifts.append({'category':group,'feature_count':v['feature_n'],
          'source_benign_mean_active':float(prevalence.loc[selected,'train_class0_prevalence'].sum()),
          'external_benign_mean_active':v['active_values']/raw['evaluated_unique_n']})
    save(OUT/'raw_category_shift.json',{'scope':'Descriptive comparison of active selected indicators; aggregate modality differences do not identify a causal failure mechanism. No target tuning is performed.','rows':shifts})
    print('Saved10 paired endpoint comparisons, family coverage and raw category shift')

if __name__=='__main__':main()
