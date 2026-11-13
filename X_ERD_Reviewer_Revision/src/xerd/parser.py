"""Conservative exact-token extraction. All ambiguity and missing groups are exposed."""
from pathlib import Path
import ntpath,json
import numpy as np
import pandas as pd
import shap
from scipy.stats import spearmanr
from .models import fit
from .metrics import metrics
from .io import save
from .explain import positive_values

OPS={'REG':{'OPENED':'regkey_opened','READ':'regkey_read','WRITTEN':'regkey_written','DELETED':'regkey_deleted'},
     'FILE':{'EXISTS':'file_exists','OPENED':'file_opened','CREATED':'file_created','RECREATED':'file_recreated','READ':'file_read','WRITTEN':'file_written','FAILED':'file_failed','DELETED':'file_deleted'},
     'DIRECTORY':{'CREATED':'directory_created','ENUMERATED':'directory_enumerated'}}

def normalize(s):return str(s).replace('/','\\').lower().rstrip('\\')
def strings(value):
    if isinstance(value,str):return [value]
    if isinstance(value,list):return [s for x in value for s in strings(x)]
    if isinstance(value,dict):return [s for k,v in value.items() if k in ['string','value','strings'] for s in strings(v)]
    return []

def vector(report,names):
    b=report.get('behavior',{});summary=b.get('summary',{})
    apis={}
    for pid,values in b.get('apistats',{}).items():
        if isinstance(values,dict):apis.update({k.lower():v for k,v in values.items()})
        elif not str(pid).isdigit():apis[str(pid).lower()]=values
    for proc in b.get('processes',[]):
        for c in proc.get('calls',[]):
            if isinstance(c,dict):apis[c.get('api','').lower()]=1
    sigs={str(x.get('name','')).lower().replace(' ','_') for x in report.get('signatures',[])}
    dlls={normalize(x) for x in summary.get('dll_loaded',[])}
    dllbase={ntpath.basename(x) for x in dlls}
    text={x.lower() for x in strings(report.get('strings',[]))}
    exts={ntpath.splitext(str(x))[1].lstrip('.').lower() for x in summary.get('file_created',[])}
    exts|={ntpath.splitext(str(x.get('name','')))[1].lstrip('.').lower() for x in report.get('dropped',[])}
    types={normalize(x.get('type','')).replace(' ','_') for x in report.get('dropped',[])}
    def active(n):
        p=n.split(':',2);group=p[0]
        if group=='API':return p[1].lower() in apis
        if group=='SIGNATURE':return p[1].lower() in sigs
        if group=='STRING':return n.partition(':')[2].lower() in text
        if len(p)<3:return False
        op,value=p[1],p[2]
        if group=='SYSTEM' and op in ['DLL_LOADED','COMMAND_LINE','MUTEX','GUID']:
            # The benchmark uses exact lowercased tokens, not basename aliases.
            # split(':',2) above preserves the drive-letter colon in full paths.
            return value.lower() in {str(x).lower() for x in summary.get(op.lower(),[])}
        if group in OPS and op in OPS[group]:return value.lower() in {str(x).lower() for x in summary.get(OPS[group][op],[])}
        if group=='DROP' and op=='EXTENSION':return value.lower() in exts
        if group=='DROP' and op=='TYPE':return normalize(value) in types
        return False
    return np.array([active(n) for n in names],dtype=np.int8)

def events(report):
    calls=[c for p in report.get('behavior',{}).get('processes',[]) for c in p.get('calls',[]) if isinstance(c,dict) and isinstance(c.get('time'),(int,float))]
    return sorted(calls,key=lambda c:c['time'])

def evaluate(ds,raw,out,jobs=4):
    reports={int(p.stem):json.loads(p.read_text(encoding='utf-8')) for p in Path(raw).glob('*.json')}
    all_df=pd.concat([ds.train,ds.test]).set_index('sample_id',verify_integrity=True)
    ids=[i for i in sorted(reports) if i in all_df.index]
    keep=~ds.train.sample_id.isin(ids).to_numpy()
    if not len(ids):raise ValueError('No matching raw reports')
    # Explicitly exclude every available evaluation report from this fit.
    m=fit(ds.X[keep],ds.y[keep],jobs=jobs)
    Xoff=all_df.loc[ids,ds.columns].to_numpy();Xpar=np.array([vector(reports[i],ds.names) for i in ids]);y=(all_df.loc[ids,'type_label']>0).to_numpy(dtype=int)
    po=m.probabilities(Xoff);pp=m.probabilities(Xpar)
    rows=[]
    for n in po:
        rows.append({'model':n,'official':metrics(y,po[n]),'parsed':metrics(y,pp[n]),
            'label_agreement':float(np.mean((po[n]>=.5)==(pp[n]>=.5))), 'probability_mae':float(abs(po[n]-pp[n]).mean())})
    categories={}
    for group in sorted({n.split(':')[0] for n in ds.names}):
        c=np.array([n.split(':')[0]==group for n in ds.names]);a=Xoff[:,c]>0;b=Xpar[:,c]>0
        tp=int((a&b).sum());fp=int((~a&b).sum());fn=int((a&~b).sum())
        categories[group]={'tp':tp,'fp':fp,'fn':fn,'precision':tp/(tp+fp) if tp+fp else None,'recall':tp/(tp+fn) if tp+fn else None}
    jac=np.sum((Xoff>0)&(Xpar>0),axis=1)/np.maximum(1,np.sum((Xoff>0)|(Xpar>0),axis=1))
    rng=np.random.default_rng(42);bg=ds.X[keep][rng.choice(keep.sum(),100,replace=False)];vo=[];vp=[]
    for model in m.base.values():
        ex=shap.TreeExplainer(model,data=bg,model_output='probability',feature_perturbation='interventional')
        vo.append(positive_values(ex,Xoff)[0]);vp.append(positive_values(ex,Xpar)[0])
    w=m.meta['X-ERD'].coef_[0];so=sum(v*w[i] for i,v in enumerate(vo));sp=sum(v*w[i] for i,v in enumerate(vp))
    stability=[]
    for i,sid in enumerate(ids):
        rho=float(spearmanr(so[i],sp[i]).statistic)
        stability.append({'sample_id':sid,'spearman':rho if np.isfinite(rho) else None,
            'top5_overlap_fraction':len(set(np.argsort(abs(so[i]))[-5:])&set(np.argsort(abs(sp[i]))[-5:]))/5})
    save(Path(out)/'parser.json',{'sample_ids':ids,'evaluation_reports_excluded_from_training':True,'training_n':int(keep.sum()),
      'benign_n':int((y==0).sum()),'ransomware_n':int(y.sum()),'scope':'small available-report holdout; no population generalization or early-encryption claim; benign FPR unavailable if no negatives',
      'models':rows,'all_categories':categories,'jaccard_per_sample':jac,'mean_jaccard':float(jac.mean()),'attribution':'trained stack logit, common probability-scale base SHAP','stability':stability})
    prefixes(ds,reports,ids,keep,out,jobs)

def prefixes(ds,reports,ids,keep,out,jobs):
    # Only API observations are causally timestamped by this extractor. Untimed
    # full-run signatures/strings/summary artifacts are deliberately excluded.
    api=np.array([n.startswith('API:') for n in ds.names]);names=np.array(ds.names)[api]
    m=fit(ds.X[keep][:,api],ds.y[keep],jobs=jobs);rows=[]
    for sid in ids:
        ev=events(reports[sid])
        if not ev:continue
        for seconds in [0,1,5,15,30,60]:
            calls={c['api'].lower() for c in ev if c['time']<=ev[0]['time']+seconds}
            x=np.array([[n.split(':',1)[1].lower() in calls for n in names]],dtype=int)
            rows.append({'sample_id':sid,'seconds':seconds,'api_calls_total':len(ev),'api_prefix_features':int(x.sum()),
                         'probability':float(m.probabilities(x)['X-ERD'][0])})
    pd.DataFrame(rows).to_csv(Path(out)/'api_prefix_predictions.csv',index=False)
    save(Path(out)/'api_prefix_protocol.json',{'features':'only selected API features; separate API-only model; same original ensemble architecture',
      'training_reports_excluded':ids,'windows_seconds':[0,1,5,15,30,60],
      'scope':'exploratory time-prefix sensitivity on available ransomware reports; no encryption-onset labels, no benign FPR, not a latency guarantee'})
