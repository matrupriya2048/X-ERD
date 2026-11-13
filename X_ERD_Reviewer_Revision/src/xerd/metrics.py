import numpy as np
from scipy import stats
from sklearn.metrics import confusion_matrix,roc_auc_score,brier_score_loss

def ece(y,p,bins=10):
    y,p=np.asarray(y),np.asarray(p)
    if len(y)==0:raise ValueError("No examples")
    idx=np.minimum((p*bins).astype(int),bins-1)
    return float(sum((idx==i).mean()*abs(y[idx==i].mean()-p[idx==i].mean()) for i in range(bins) if (idx==i).any()))

def metrics(y,p,theta=.5):
    y,p=np.asarray(y),np.asarray(p)
    if not len(y) or len(y)!=len(p) or not np.isfinite(p).all() or (p<0).any() or (p>1).any():raise ValueError("Invalid probabilities")
    tn,fp,fn,tp=map(int,confusion_matrix(y,p>=theta,labels=[0,1]).ravel())
    div=lambda a,b:float(a/b) if b else None
    return dict(tn=tn,fp=fp,fn=fn,tp=tp,accuracy=(tn+tp)/len(y),precision=div(tp,tp+fp),recall=div(tp,tp+fn),
                fpr=div(fp,tn+fp),f1=div(2*tp,2*tp+fp+fn),auc=float(roc_auc_score(y,p)) if len(np.unique(y))==2 else None,
                brier=float(brier_score_loss(y,p)),ece=ece(y,p),theta=float(theta),n=len(y))

def matched(y,p,target):
    """Descriptive test ROC only. Highest threshold attaining the exact target."""
    positives=np.sort(np.asarray(p)[np.asarray(y)==1])[::-1]
    if not len(positives) or not 0<target<=1:raise ValueError("Invalid recall target")
    threshold=positives[int(np.ceil(target*len(positives)-1e-10))-1]
    return {"target":target,**metrics(y,p,float(threshold))}

def bootstrap(y,p,n=2000,seed=42):
    rng=np.random.default_rng(seed);y=np.asarray(y);p=np.asarray(p)
    groups=[np.where(y==c)[0] for c in [0,1]]
    if any(len(g)==0 for g in groups):raise ValueError("Stratified binary intervals require both classes")
    values={k:[] for k in ["accuracy","precision","recall","fpr","f1"]}
    for _ in range(n):
        ix=np.concatenate([rng.choice(g,len(g),replace=True) for g in groups]);r=metrics(y[ix],p[ix])
        for k in values:values[k].append(r[k])
    return {k:np.quantile(v,[.025,.975]) for k,v in values.items()}

def paired(a,b,ratio=.25):
    a,b=np.asarray(a),np.asarray(b);d=np.round(a-b,14);n=len(d)
    if n<2:raise ValueError("At least two pairs required")
    var=d.var(ddof=1);se=np.sqrt((1/n+ratio)*var)
    t=float(d.mean()/se) if se else 0.
    non=d[d!=0]
    ranks=stats.rankdata(abs(non))
    rb=float(np.sum(np.sign(non)*ranks)/ranks.sum()) if len(non) else 0.
    return dict(n=n,mean_difference=float(d.mean()),sd_difference=float(d.std(ddof=1)),
        lower=int((d<0).sum()),higher=int((d>0).sum()),ties=int((d==0).sum()),
        nb_t=t,nb_p=float(2*stats.t.sf(abs(t),n-1)) if se else (1. if not d.mean() else 0.),
        wilcoxon_p=float(stats.wilcoxon(d,zero_method="wilcox",method="asymptotic").pvalue) if len(non) else 1.,
        paired_rank_biserial=rb,zero_method="wilcox",correction_factor=1/n+ratio)

def holm(ps):
    order=np.argsort(ps);out=np.empty(len(ps));running=0.
    for k,i in enumerate(order):running=max(running,min(1.,ps[i]*(len(ps)-k)));out[i]=running
    return out

def delong(y,a,b):
    y=np.asarray(y);scores=np.array([a,b]);m=int(y.sum());n=len(y)-m
    if min(m,n)<2:raise ValueError("Two examples per class required")
    pos,neg=scores[:,y==1],scores[:,y==0]
    tx=np.array([stats.rankdata(x) for x in pos]);ty=np.array([stats.rankdata(x) for x in neg])
    tz=np.array([stats.rankdata(np.r_[x,z]) for x,z in zip(pos,neg)])
    v01=(tz[:,:m]-tx)/n;v10=1-(tz[:,m:]-ty)/m
    auc=v01.mean(1);cov=np.cov(v01)/m+np.cov(v10)/n
    var=float(np.array([1.,-1.])@cov@np.array([1.,-1.]))
    delta=float(auc[0]-auc[1]);z=delta/np.sqrt(var) if var>0 else 0.
    return {"aucs":auc,"delta_auc":delta,"z":z,"p":float(2*stats.norm.sf(abs(z))) if var>0 else (1. if delta==0 else 0.)}
