from pathlib import Path
import numpy as np
import pandas as pd
import shap
from scipy import stats
from scipy.special import expit,logit
from .io import save

def positive_values(explainer,X):
    v=explainer.shap_values(X,check_additivity=False)
    if isinstance(v,list):v=v[1]
    v=np.asarray(v)
    if v.ndim==3:v=v[:,:,1]
    b=np.asarray(explainer.expected_value).reshape(-1)[-1]
    return v,float(b)

def explain(ds,m,out,background_size=100):
    out=Path(out);rng=np.random.default_rng(42)
    ix=rng.choice(len(ds.y),min(background_size,len(ds.y)),replace=False)
    background=ds.X[ix];vals={};expected={};errors={}
    for name,model in m.base.items():
        print('Probability SHAP',name,flush=True)
        explainer=shap.TreeExplainer(model,data=background,model_output='probability',feature_perturbation='interventional')
        v,b=positive_values(explainer,ds.Xt)
        error=float(abs(b+v.sum(1)-model.predict_proba(ds.Xt)[:,1]).max())
        if error>1e-4:raise AssertionError(f'{name} probability additivity error {error}')
        vals[name]=v;expected[name]=b;errors[name]=error
    w=m.meta['X-ERD'].coef_[0];intercept=m.meta['X-ERD'].intercept_[0]
    stack_logit=sum(w[i]*vals[n] for i,n in enumerate(m.base))
    baseline=intercept+sum(w[i]*expected[n] for i,n in enumerate(m.base))
    p=m.probabilities(ds.Xt)['X-ERD']
    error=float(abs(expit(baseline+stack_logit.sum(1))-p).max())
    if error>1e-4:raise AssertionError('Stack logit additivity failed')
    uniform=sum(vals.values())/3
    np.savez_compressed(out/'shap_values.npz',**vals,stack_logit=stack_logit,uniform_probability_surrogate=uniform,
        background_ids=ds.train.sample_id.to_numpy()[ix],expected_values=np.array(list(expected.values())),stack_logit_baseline=baseline)
    # Preserve the original deterministic global sampling rule, report all names.
    global_ix=np.linspace(0,len(ds.yt)-1,100,dtype=int)
    pd.DataFrame({'feature_id':ds.columns,'feature_name':ds.names,'mean_abs_stack_logit_shap':abs(stack_logit[global_ix]).mean(0),
                  'mean_abs_uniform_surrogate_shap':abs(uniform[global_ix]).mean(0)}).sort_values('mean_abs_stack_logit_shap',ascending=False).to_csv(out/'global_shap.csv',index=False)
    save(out/'explanations.json',{'base_scale':'probability with common training background','explained_stack_output':'pre-sigmoid trained stack logit',
        'not_claimed':'Uniform average is a probability-mean surrogate, not the deployed logistic stack. Logit contributions are not probability-space full-stack SHAP.',
        'base_expected_values':expected,'max_base_additivity_errors':errors,'max_stack_probability_reconstruction_error':error,
        'background_sample_ids':ds.train.sample_id.to_numpy()[ix],'global_sample_ids':ds.test.sample_id.to_numpy()[global_ix],
        'meta_coefficients':w,'meta_intercept':intercept})
    deletion(ds,m,vals,stack_logit,uniform,out)
    targeted(ds,m,vals['LGB'],stack_logit,out)

def deletion(ds,m,vals,stack,uniform,out,steps=20):
    rng=np.random.default_rng(42)
    ix=np.r_[rng.choice(np.where(ds.yt==0)[0],150,replace=False),rng.choice(np.where(ds.yt==1)[0],150,replace=False)]
    X=ds.Xt[ix];y=ds.yt[ix];curves={};rows=[];curve_rows=[]
    schemes={'LGB':vals['LGB'][ix], 'uniform_surrogate':uniform[ix],'stack_logit':stack[ix],
             'random':rng.normal(size=X.shape)}
    for name,v in schemes.items():
        # Explain confidence in the true class separately; benign confidence is 1-p.
        score=v*np.where(y==1,1.,-1.)[:,None]
        order=np.argsort(-np.where(X>0,score,-np.inf),axis=1)
        batch=np.repeat(X[:,None,:],steps+1,axis=1)
        for k in range(1,steps+1):
            for i in range(len(X)):
                active=order[i,:min(k,int(X[i].sum()))];batch[i,k,active]=0
        pp=m.probabilities(batch.reshape(-1,X.shape[1]))['X-ERD'].reshape(len(X),steps+1)
        confidence=np.where(y[:,None]==1,pp,1-pp)
        for cls,k in [('all',np.ones(len(y),bool)),('benign',y==0),('ransomware',y==1)]:
            for step in range(steps+1):
                v=confidence[k,step];curve_rows.append({'method':name,'class':cls,'deleted':step,
                  'mean_true_class_confidence':float(v.mean()),'se_true_class_confidence':float(v.std(ddof=1)/np.sqrt(len(v)))})
        auc=np.trapezoid(confidence,x=np.arange(steps+1)/steps,axis=1)
        curves[name]=auc
        for i in range(len(X)):rows.append({'sample_id':int(ds.test.sample_id.iloc[ix[i]]),'label':int(y[i]),'method':name,'deletion_auc':float(auc[i])})
    pd.DataFrame(rows).to_csv(Path(out)/'deletion_samples.csv',index=False)
    pd.DataFrame(curve_rows).to_csv(Path(out)/'deletion_curves.csv',index=False)
    summary={}
    for cls,k in [('all',np.ones(len(y),bool)),('benign',y==0),('ransomware',y==1)]:
        summary[cls]={}
        for name,v in curves.items():
            a=v[k];summary[cls][name]={'mean':float(a.mean()),'sample_sd':float(a.std(ddof=1)),
                'mean_ci95_normal':(a.mean()+np.array([-1,1])*1.96*a.std(ddof=1)/np.sqrt(len(a))).tolist()}
        for comp in ['LGB','uniform_surrogate','random']:
            d=curves['stack_logit'][k]-curves[comp][k]
            summary[cls]['stack_vs_'+comp]={'mean_difference':float(d.mean()),
                'paired_t_p':float(stats.ttest_rel(curves['stack_logit'][k],curves[comp][k]).pvalue) if d.std()>0 else 1.,
                'wilcoxon_p':float(stats.wilcoxon(d,zero_method='wilcox',method='asymptotic').pvalue) if np.any(d!=0) else 1.}
    save(Path(out)/'deletion.json',{'steps':steps,'normalization':'x-axis k/20; exact same samples/model/grid for all schemes',
        'ranking':'active indicators, signed attribution toward true class; random order control',
        'limitation':'zeroing may produce out-of-distribution inputs; deletion is empirical sensitivity, not proof of explanation correctness', 'results':summary})

def targeted(ds,m,lgb,stack,out):
    indices=np.where(ds.yt==1)[0];X=ds.Xt[indices]
    rows=[]
    for name,v in [('LGB_surrogate',lgb[indices]),('trained_stack_logit',stack[indices])]:
        order=np.argsort(-np.where(X>0,v,-np.inf),axis=1)
        for k in [0,1,3,5,10,20,40]:
            q=X.copy()
            for i in range(len(X)):q[i,order[i,:min(k,int(X[i].sum()))]]=0
            p=m.probabilities(q)['X-ERD']
            rows.append({'method':name,'budget':k,'n':len(X),'detected':int((p>=.5).sum()),'recall':float((p>=.5).mean())})
    save(Path(out)/'targeted_masking.json',{'scope':'one-shot feature-space masking, no executable evasion or functionality validation','rows':rows})
