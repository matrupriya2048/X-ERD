from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size':12,'axes.labelsize':12,'axes.titlesize':13,'legend.fontsize':10})
from sklearn.metrics import roc_curve

def generate(ds,out):
    out=Path(out);fig=out/'figures';fig.mkdir(exist_ok=True)
    p=pd.read_csv(out/'test_predictions.csv',float_precision='round_trip')
    names=['RF','XGB','LGB','RF+LGB','X-ERD']
    f,ax=plt.subplots(1,2,figsize=(12,4.5))
    for n in names:
        x,y,_=roc_curve(p.y_true,p[n],drop_intermediate=False)
        for a in ax:a.plot(x,y,label=n)
    ax[0].set(xlabel='False positive rate',ylabel='Recall',title='ROC on official held-out split')
    ax[1].set(xlim=(0,.05),ylim=(.9,1),xlabel='False positive rate',ylabel='Recall',title='Low-FPR ROC region')
    for a in ax:a.grid(alpha=.2);a.legend(fontsize=10)
    f.tight_layout();f.savefig(fig/'matched_roc.png',dpi=250);plt.close(f)
    f,ax=plt.subplots(figsize=(7,5))
    for n in names:
        pred=p[n].to_numpy();bins=np.minimum((pred*10).astype(int),9)
        valid=[i for i in range(10) if (bins==i).any()]
        ax.plot([pred[bins==i].mean() for i in valid],[p.y_true[bins==i].mean() for i in valid],'o-',label=n)
    ax.plot([0,1],[0,1],'k--');ax.set(xlabel='Predicted ransomware probability',ylabel='Observed ransomware fraction',title='Reliability;10 equal-width bins');ax.legend();f.tight_layout();f.savefig(fig/'calibration.png',dpi=250);plt.close(f)
    if (out/'threshold_curves.csv').exists():
        df=pd.read_csv(out/'threshold_curves.csv');f,axs=plt.subplots(1,2,figsize=(12,4.5))
        for a,metric in zip(axs,['recall','fpr']):
            for n in ['LGB','RF+LGB','X-ERD']:
                for split,ls in [('validation','-'),('test_descriptive','--')]:
                    d=df[(df.model==n)&(df.split==split)];a.plot(d.theta,d[metric],ls,label=n+' '+split)
            a.set(xlabel='Threshold',ylabel=metric);a.legend(fontsize=6);a.grid(alpha=.2)
        f.tight_layout();f.savefig(fig/'threshold_stability.png',dpi=250);plt.close(f)
    if (out/'global_shap.csv').exists():
        df=pd.read_csv(out/'global_shap.csv').head(14).iloc[::-1];f,ax=plt.subplots(figsize=(8,5))
        # IDs are unambiguous; full feature names are in the companion CSV.
        ax.barh(df.feature_id.astype(str),df.mean_abs_stack_logit_shap);ax.set(xlabel='Mean absolute trained-stack logit SHAP',ylabel='Feature ID; full names in global_shap.csv')
        f.tight_layout();f.savefig(fig/'global_shap.png',dpi=250);plt.close(f)
    if (out/'corruption_summary.csv').exists():
        df=pd.read_csv(out/'corruption_summary.csv');f,axs=plt.subplots(1,2,figsize=(12,4.5))
        for a,key in zip(axs,['recall','fpr']):
            for n in names:
                d=df[(df.model==n)&df.scenario.isin(['drop10','drop20','drop30'])].sort_values('scenario')
                a.errorbar([10,20,30],d[key+'_mean']*100,yerr=d[key+'_sd']*100,marker='o',label=n)
            a.set(xlabel='Per-indicator dropout (%)',ylabel=key+' (%)');a.legend(fontsize=8);a.grid(alpha=.2)
        f.tight_layout();f.savefig(fig/'corruption.png',dpi=250);plt.close(f)
    # Tables are exported mechanically from JSON, never transcribed by hand.
    c=json.loads((out/'core.json').read_text());pd.DataFrame(c['metrics']).T.to_csv(out/'table_core.csv')
    if (out/'temporal_fixed483.json').exists():
        d=json.loads((out/'temporal_fixed483.json').read_text());rows=[]
        for yr,v in d['years'].items():rows.append({'year':yr,**v['X-ERD']})
        pd.DataFrame(rows).to_csv(out/'table_temporal_fixed483.csv',index=False)
    additional_figures(ds,out,fig)

def additional_figures(ds,out,fig):
    from scipy.stats import t
    from matplotlib.patches import FancyBboxPatch
    f,ax=plt.subplots(figsize=(11,4));ax.set(xlim=(0,13),ylim=(0,4));ax.axis('off')
    boxes=[(.15,2.5,'Raw Cuckoo JSON\nParser evaluation:20 holdouts'),(3,2.5,'Official483 indicators\nRF / XGBoost / LightGBM'),
           (6.3,2.5,'Training-only inner OOF\nL2 logistic stacking'),(9.6,2.5,'Ransomware probability\nThresholded classification'),
           (3,.4,'Common training background\nProbability-scale Tree SHAP'),(6.3,.4,'Unnormalized meta weights\nExact additive stack logit'),
           (9.6,.4,'Versioned ATT&CK rules\nCandidate suggestions')]
    for x,y,text in boxes:
        ax.add_patch(FancyBboxPatch((x,y),2.9,1,boxstyle='round,pad=0.05',facecolor='#e9f2f7',edgecolor='#244b63'))
        ax.text(x+1.45,y+.5,text,ha='center',va='center',fontsize=11)
    for a,b in [((3,3),(3.1,3)),((5.9,3),(6.25,3)),((9.2,3),(9.55,3)),((4.45,2.45),(4.45,1.45)),((5.95,.9),(6.25,.9)),((9.25,.9),(9.55,.9))]:
        ax.annotate('',xy=b,xytext=a,arrowprops={'arrowstyle':'->','lw':1.5})
    f.tight_layout();f.savefig(fig/'architecture.png',dpi=250);plt.close(f)
    df=pd.concat([ds.train,ds.test]).merge(ds.metadata[['sample_id','first_submission_date_year']],on='sample_id')
    years=sorted(df.first_submission_date_year.unique());f,ax=plt.subplots(figsize=(10,4))
    ax.bar(years,[((df.first_submission_date_year==yr)&(df.sample_type==0)).sum() for yr in years],label='Goodware')
    ax.bar(years,[((df.first_submission_date_year==yr)&(df.sample_type==1)).sum() for yr in years],
        bottom=[((df.first_submission_date_year==yr)&(df.sample_type==0)).sum() for yr in years],label='Ransomware')
    ax.set(xlabel='First-submission year',ylabel='Samples',title=f'All samples:{min(years)}-{max(years)}; ransomware:{df.loc[df.sample_type==1,"first_submission_date_year"].min()}-{df.loc[df.sample_type==1,"first_submission_date_year"].max()}')
    ax.legend();f.tight_layout();f.savefig(fig/'temporal_distribution.png',dpi=250);plt.close(f)
    if (out/'cv_summary.json').exists():
        d=json.loads((out/'cv_summary.json').read_text());v=d['fpr']['comparisons'];f,ax=plt.subplots(figsize=(8,4))
        for i,(n,r) in enumerate(v.items()):
            error=t.ppf(.975,49)*np.sqrt(.27)*r['sd_difference']*100
            ax.errorbar(r['mean_difference']*100,i,xerr=error,fmt='o',color='#245e7c')
        ax.axvline(0,color='black',ls='--');ax.set(yticks=range(4),yticklabels=list(v),xlabel='X-ERD minus comparator FPR (percentage points)',title='Approximate95% corrected repeated-CV intervals')
        f.tight_layout();f.savefig(fig/'cv_corrected.png',dpi=250);plt.close(f)
    if (out/'temporal_strict.json').exists():
        d=json.loads((out/'temporal_strict.json').read_text());f,axs=plt.subplots(1,2,figsize=(11,4))
        for ax,key in zip(axs,['recall','fpr']):
            for n in ['LGB','RF+LGB','X-ERD']:
                ax.plot(list(d['years']),[r[n][key]*100 for r in d['years'].values()],'o-',label=n)
            ax.set(xlabel='Held-out first-submission year',ylabel=key+' (%)');ax.legend();ax.grid(alpha=.2)
        f.tight_layout();f.savefig(fig/'temporal_strict.png',dpi=250);plt.close(f)
    if (out/'deletion_curves.csv').exists():
        d=pd.read_csv(out/'deletion_curves.csv');f,axs=plt.subplots(1,2,figsize=(11,4))
        for ax,cls in zip(axs,['benign','ransomware']):
            for n in d.method.unique():
                r=d[(d['class']==cls)&(d.method==n)];ax.plot(r.deleted,r.mean_true_class_confidence,label=n)
                ax.fill_between(r.deleted,r.mean_true_class_confidence-1.96*r.se_true_class_confidence,r.mean_true_class_confidence+1.96*r.se_true_class_confidence,alpha=.1)
            ax.set(xlabel='Active indicators set to zero',ylabel='True-class probability',title=cls);ax.legend(fontsize=7)
        f.tight_layout();f.savefig(fig/'deletion.png',dpi=250);plt.close(f)
    if (out/'targeted_masking.json').exists():
        d=json.loads((out/'targeted_masking.json').read_text());r=pd.DataFrame(d['rows']);f,ax=plt.subplots(figsize=(7,4))
        for name in r.method.unique():
            z=r[r.method==name];ax.plot(z.budget,z.recall*100,'o-',label=name)
        ax.set(xlabel='Maximum active indicators zeroed',ylabel='Ransomware recall (%)',title='Feature-space masking; functionality untested');ax.legend(fontsize=8);f.tight_layout();f.savefig(fig/'targeted.png',dpi=250);plt.close(f)
    if (out/'parser.json').exists():
        d=json.loads((out/'parser.json').read_text());f,axs=plt.subplots(1,2,figsize=(11,4))
        axs[0].bar(range(20),d['jaccard_per_sample']);axs[0].set(xlabel='Raw report index',ylabel='Feature Jaccard')
        st=pd.DataFrame(d['stability']);axs[1].scatter(st.spearman,st.top5_overlap_fraction);axs[1].set(xlabel='Signed attribution Spearman correlation',ylabel='Top5 absolute-attribution overlap fraction')
        f.tight_layout();f.savefig(fig/'parser_stability.png',dpi=250);plt.close(f)
