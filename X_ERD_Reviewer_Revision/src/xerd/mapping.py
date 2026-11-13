"""Versioned, evidence-scoped ATT&CK suggestions, not confirmed forensic facts."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from .io import save

def map_features(names,schema):
    out={}
    for name in names:
        rule=next((r for r in schema['rules'] if r.get('exact')==name or (r.get('prefix') and name.startswith(r['prefix']))),None)
        out[name]=rule or {'technique_id':None,'confidence_tier':'Footprint','rule_specificity_score':0,'rationale':'No context-supported technique inferred from this feature alone.'}
    return out

def evaluate(ds,m,out,schema_path,annotations=None):
    schema=json.loads(Path(schema_path).read_text(encoding='utf-8'));mapping=map_features(ds.names,schema)
    p=m.probabilities(ds.Xt)['X-ERD'];sel=(ds.yt==1)&(p>=.5)
    with np.load(Path(out)/'shap_values.npz') as z:phi=z['stack_logit'][sel]
    active=np.maximum(phi,0)*(ds.Xt[sel]>0);mapped=np.array([mapping[n]['technique_id'] is not None for n in ds.names])
    mass=active[:,mapped].sum(1)/np.maximum(1e-15,active.sum(1))
    result={'schema_version':schema['version'],'feature_count':len(mapping),'mapped_features':int(mapped.sum()),
       'distinct_techniques':sorted({r['technique_id'] for r in mapping.values() if r['technique_id']}),
       'all_feature_mappings':mapping,'attribution_sample_ids':ds.test.sample_id.to_numpy()[sel],
       'attribution_output':'trained stack logit, active positive mass', 'mass_mapped_per_sample':mass,
       'mass_mean':float(mass.mean()),'mass_median':float(np.median(mass)),
       'independent_accuracy':'NOT EVALUATED: no independent human/external reference labels supplied',
       'confidence_scope':'qualitative rule specificity, not calibrated probabilities or a forensic ground truth'}
    # A reproducible blinded worksheet: assessors label without seeing our label.
    rng=np.random.default_rng(42);audit=[]
    for flag,n in [(True,20),(False,10)]:
        pool=np.where(mapped==flag)[0]
        for j in rng.choice(pool,min(n,len(pool)),replace=False):
            audit.append({'feature_id':ds.columns[j],'feature_name':ds.names[j],'reference_technique':'','assessor_id':'','provenance_url':'','assessment_notes':''})
    pd.DataFrame(audit).to_csv(Path(out)/'independent_mapping_audit_blank.csv',index=False)
    if annotations:
        ref=pd.read_csv(annotations,keep_default_na=False)
        required={'feature_name','reference_technique','assessor_id','provenance_url'}
        if not required<=set(ref):raise ValueError('Independent annotation columns missing')
        if len(ref)<30 or (ref[list(required)]=='').any().any():raise ValueError('At least30 complete independent records required')
        if not set(ref.feature_name)<=set(mapping):raise ValueError('Unknown audit feature')
        if ref.feature_name.duplicated().any():raise ValueError('Duplicate audit item')
        hit=[(mapping[n]['technique_id'] or 'NONE')==t for n,t in zip(ref.feature_name,ref.reference_technique)]
        result['reference_agreement']={'n':len(hit),'exact_agreement':float(np.mean(hit)),
           'caveat':'Agreement conditional on supplied references; assessor independence must be documented, not inferred by software'}
    save(Path(out)/'mapping.json',result)
