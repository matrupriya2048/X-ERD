"""Independent EldeRan collection: strict new split and shared-API transfer."""
from pathlib import Path
import zipfile,json,hashlib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
from .models import fit
from .additional import binary_mi
from .metrics import metrics,bootstrap
from .io import save,digest

def source_data(source):
    source=Path(source)
    with zipfile.ZipFile(source/'elderan_data.txt') as z:
        with z.open('RansomwareData.csv') as f:df=pd.read_csv(f,header=None,dtype=np.int32)
    names=[]
    for line in (source/'elderan_names.txt').read_text(encoding='utf-8').splitlines():names.append(line.split(';',1)[1])
    if len(names)!=df.shape[1]:raise ValueError('Feature-name alignment failed')
    ids=pd.read_csv(source/'elderan_ids.txt',sep=';').set_index('ID')
    if set(df.iloc[:,1])!={0,1}:raise ValueError('Invalid original EldeRan labels')
    meta=ids.loc[df.iloc[:,0]].reset_index()
    if not np.array_equal(meta.Ransomware,df.iloc[:,1]):raise ValueError('Source labels disagree')
    X=df.iloc[:,3:].to_numpy(dtype=np.uint8)
    if not np.isin(X,[0,1]).all():raise ValueError('Expected binary source data')
    return X,df.iloc[:,1].to_numpy(),names[3:],meta

def run(ds,source,out,jobs=4):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    X,y,names,meta=source_data(source)
    original_n=len(y)
    # Executable hashes, not record IDs, define independent sample units.
    # Keep the lowest original source ID before splitting/selection; no outcomes
    # or model scores enter this decision. Identical vectors remain grouped too.
    if meta.assign(hash=meta.MD5.str.lower()).groupby('hash').Ransomware.nunique().max()>1:
        raise ValueError('Conflicting labels for an executable hash')
    keep=meta.sort_values('ID').drop_duplicates('MD5',keep='first').index.to_numpy()
    pd.DataFrame({'source_id':meta.ID,'md5':meta.MD5,'kept':np.isin(np.arange(len(meta)),keep)}).to_csv(out/'elderan_dedup_audit.csv',index=False)
    X=X[keep];y=y[keep];meta=meta.iloc[keep].reset_index(drop=True)
    protocol={'source':'https://github.com/rissgrouphub/ransomwaredataset2016','source_arxiv':'1609.03020',
      'checksums':{n:digest(Path(source)/n) for n in ['elderan_data.txt','elderan_names.txt','elderan_ids.txt']},
      'within_collection':'lowest source ID per executable MD5 kept before split; seed42 StratifiedGroupKFold5 fold0 test; identical full feature vectors grouped; train-only MI2000 + RF100 RFE483; original X-ERD architecture',
      'transfer':'train official MLRan training IDs on exact name-matched API indicators; evaluate independent EldeRan records after MD5 overlap removal; no target tuning',
      'limitations':'external historical WindowsXP corpus; within-collection refit and direct transfer reported separately; API transfer uses restricted common representation, not frozen483 model'}
    save(out/'external_dataset_protocol.json',protocol)
    groups=np.array([hashlib.sha256(row.tobytes()).hexdigest() for row in X])
    a,b=next(StratifiedGroupKFold(5,shuffle=True,random_state=42).split(X,y,groups))
    if set(groups[a])&set(groups[b]):raise AssertionError('Vector duplicate leakage')
    # Same binary input across baselines. All supervised selection excludes b.
    mi=binary_mi(X[a],y[a]);rank=np.lexsort((np.arange(X.shape[1]),-mi))[:2000]
    selector=RFE(RandomForestClassifier(n_estimators=100,random_state=42,n_jobs=jobs),n_features_to_select=483,step=.2).fit(X[a][:,rank],y[a])
    sel=rank[selector.support_];m=fit(X[a][:,sel],y[a],jobs=jobs);p=m.probabilities(X[b][:,sel])
    pd.DataFrame({'source_id':meta.ID.iloc[b].to_numpy(),'y_true':y[b],**p}).to_csv(out/'elderan_holdout_predictions.csv',index=False)
    pd.DataFrame({'source_id':meta.ID,'vector_group':groups,'split':np.where(np.isin(np.arange(len(y)),b),'test','train')}).to_csv(out/'elderan_split.csv',index=False)
    pd.DataFrame({'source_feature_index':sel,'name':np.array(names)[sel]}).to_csv(out/'elderan_selected_features.csv',index=False)
    results={'source_records':original_n,'duplicate_executables_removed':original_n-len(y),'n':len(y),'benign':int((y==0).sum()),'ransomware':int(y.sum()),'unique_vectors':len(set(groups)),
      'within_collection':{'train_n':len(a),'test_n':len(b),'models':{n:metrics(y[b],v) for n,v in p.items()},'full_ci95':bootstrap(y[b],p['X-ERD'])}}
    old={name.lower():i for i,name in enumerate(names) if name.startswith('API:')}
    pairs=[(j,old[n.lower()],n) for j,n in enumerate(ds.names) if n.startswith('API:') and n.lower() in old]
    if not pairs:raise ValueError('No common APIs')
    md5cols=[c for c in ds.metadata if 'md5' in c.lower()]
    if len(md5cols)!=1:raise ValueError('Cannot establish source MD5 overlap')
    seen=set(ds.metadata[md5cols[0]].astype(str).str.lower());keep=~meta.MD5.astype(str).str.lower().isin(seen).to_numpy()
    m=fit(ds.X[:,[x[0] for x in pairs]],ds.y,jobs=jobs)
    p=m.probabilities(X[keep][:,[x[1] for x in pairs]])
    pd.DataFrame({'source_id':meta.ID.to_numpy()[keep],'y_true':y[keep],**p}).to_csv(out/'elderan_transfer_predictions.csv',index=False)
    pd.DataFrame(pairs,columns=['mlran_index','elderan_index','api']).to_csv(out/'elderan_common_apis.csv',index=False)
    results['transfer']={'shared_api_features':len(pairs),'excluded_md5_overlap':int((~keep).sum()),'test_n':int(keep.sum()),'benign':int((y[keep]==0).sum()),'ransomware':int(y[keep].sum()),'models':{n:metrics(y[keep],v) for n,v in p.items()},'full_ci95':bootstrap(y[keep],p['X-ERD'])}
    save(out/'external_dataset.json',results)
    print('EldeRan evaluation complete',results['within_collection']['models']['X-ERD'],results['transfer']['models']['X-ERD'],flush=True)
