"""Reconstruct the disclosed development set; audit schemas without relabeling."""
from pathlib import Path
import json,collections
import numpy as np
import pandas as pd
import joblib
from xerd.io import load,save,digest
from xerd.parser import vector

def main():
    root=Path.cwd();out=root/'results/remediation';out.mkdir(parents=True,exist_ok=True)
    ds=load(root/'data/mlran');manifest=json.loads((root/'config/benign_report_sources.json').read_text())
    accepted={r['source_path'] for r in json.loads((root/'results/extended/external_raw_audit.json').read_text()) if r['status']=='evaluated'}
    rows=[];X=[];audit=[]
    for i,item in enumerate(manifest['files']):
        if item['path'] not in accepted:continue
        path=root/item['path'];assert digest(path)==item['sha256']
        d=json.loads(path.read_text(encoding='utf-8'));f=d['target']['file'];b=d['behavior'];x=vector(d,ds.names)
        X.append(x);rows.append({'source_path':item['path'],'md5':f['md5'].lower(),'sha1':f.get('sha1'),'sha256':f.get('sha256'),'y_true':0})
        audit.append({'source_path':item['path'],'filename':f.get('name'),'machine':d.get('info',{}).get('machine',{}).get('name'),
          'cuckoo_version':d.get('info',{}).get('version'),'duration':d.get('info',{}).get('duration'),'summary_keys':sorted(b.get('summary',{})),
          'summary_counts':{k:len(v) if hasattr(v,'__len__') else None for k,v in b.get('summary',{}).items()},'feature_counts':{g:int(x[[n.startswith(g+':') for n in ds.names]].sum()) for g in sorted({n.split(':')[0] for n in ds.names})}})
        if len(rows)%25==0:print('Reconstructed development reports',len(rows),flush=True)
    assert len(rows)==182;X=np.asarray(X);ids=pd.DataFrame(rows)
    old=pd.read_csv(root/'results/extended/external_raw_predictions.csv',float_precision='round_trip').set_index('source_path')
    model=joblib.load(root/'results/model.joblib');p=model.probabilities(X)
    for name,v in p.items():assert np.allclose(v,old.loc[ids.source_path,name],atol=1e-14,rtol=0),name
    frame=pd.concat([ids,pd.DataFrame(X,columns=ds.columns)],axis=1);frame.to_csv(out/'development_vectors.csv',index=False)
    save(out/'development_schema_audit.json',audit)
    save(out/'development_preparation.json',{'reports':len(rows),'sha256':digest(out/'development_vectors.csv'),'all_original_predictions_reconstructed':True,
      'schema_versions':dict(collections.Counter(str(r['cuckoo_version']) for r in audit)),'machines':dict(collections.Counter(str(r['machine']) for r in audit)),
      'source_files_verified':True,'labels':'Unchanged publisher clean labels; this set is now development data, not a new test'})
    print('PASS: 182 development vectors reproduce all seven original model scores',flush=True)

if __name__=='__main__':main()
