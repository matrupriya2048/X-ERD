"""Render updated architecture and measured external comparisons from exports."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from scipy.stats import beta

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/extended';FIG=ROOT/'results/figures'

def ci(k,n):
    return [beta.ppf(.025,k,n-k+1) if k else 0,beta.ppf(.975,k+1,n-k) if k<n else 1]

def main():
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.labelsize':11})
    f,ax=plt.subplots(figsize=(10.5,4.7));ax.set(xlim=(0,13.2),ylim=(0,5));ax.axis('off')
    boxes=[(.15,2.7,'Raw Cuckoo reports\nSeparate parser\nassessment'),(3.4,2.7,'483 indicator inputs\nRF / XGBoost / LightGBM'),
           (6.7,2.7,'Training-only OOF scores\nRegularized logistic head'),(10,2.7,'Ransomware probability\nThresholded classification'),
           (3.4,.5,'Shared training\nbackground\nBase probability SHAP'),(6.7,.5,'Fitted linear coefficients\nExact logit attribution'),
           (10,.5,'Versioned ATT&CK rules\nCandidate suggestions')]
    for x,y,t in boxes:
        ax.add_patch(FancyBboxPatch((x,y),2.9,1.05,boxstyle='round,pad=0.04',facecolor='#edf3f6',edgecolor='#25485e'))
        ax.text(x+1.45,y+.525,t,ha='center',va='center',fontsize=12)
    for a,b in [((3.08,3.22),(3.34,3.22)),((6.35,3.22),(6.64,3.22)),((9.65,3.22),(9.94,3.22)),
                ((4.85,2.65),(4.85,1.61)),((6.35,1.02),(6.64,1.02)),((9.65,1.02),(9.94,1.02))]:
        ax.annotate('',xy=b,xytext=a,arrowprops={'arrowstyle':'->','lw':1.4,'color':'#25485e'})
    ax.text(1.6,1.83,'Original vectors and parsed\nvectors are evaluated\nunder separate protocols.',ha='center',va='center',fontsize=9)
    ax.plot([3.52,3.52,8.15],[3.82,4.35,4.35],ls='--',color='#9b4d16',lw=1.4)
    ax.annotate('',xy=(8.15,3.81),xytext=(8.15,4.35),arrowprops={'arrowstyle':'->','lw':1.4,'linestyle':'--','color':'#9b4d16'})
    ax.text(5.83,4.57,'Optional raw-feature passthrough',ha='center',fontsize=12,color='#753a12')
    ax.text(6.6,.08,'Adapted variant: add 182 disclosed benign development reports; new holdouts never enter fitting.',ha='center',va='bottom',fontsize=11,color='#753a12')
    f.tight_layout(pad=.7);f.savefig(FIG/'architecture_revision.png',dpi=300);plt.close(f)
    groups=[('Original full',pd.read_csv(OUT/'elderan_transfer_predictions.csv',float_precision='round_trip'),'X-ERD'),
            ('Fixed stack tuned',pd.read_csv(OUT/'optimized_validation/transfer_predictions.csv',float_precision='round_trip'),'X-ERD'),
            ('Passthrough full',pd.read_csv(OUT/'residual_validation/transfer_predictions.csv',float_precision='round_trip'),'X-ERD')]
    published=pd.read_csv(OUT/'external_comparators_predictions.csv',float_precision='round_trip')
    residual=groups[2][1]
    groups += [('Passthrough '+n,residual,n) for n in ['RF+XGB','RF+LGB','XGB+LGB']]
    groups += [(n.replace('-adapted',''),published,n) for n in published.columns if n not in ['source_id','y_true']]
    f,axs=plt.subplots(1,2,figsize=(12,6.3),sharey=True)
    rows=[]
    for i,(name,p,col) in enumerate(groups):
        for ax,label,truth in zip(axs,['FPR','Recall'],[0,1]):
            q=p.loc[p.y_true==truth,col]>=.5;n=len(q);k=int(q.sum());lo,hi=ci(k,n);v=k/n
            ax.errorbar(v*100,i,xerr=[[100*(v-lo)],[100*(hi-v)]],fmt='o',capsize=3,color='#a54a12' if name=='Passthrough full' else '#245e7c')
            rows.append({'model':name,'endpoint':label,'numerator':k,'denominator':n,'rate':v,'ci_low':lo,'ci_high':hi})
    axs[0].set(yticks=range(len(groups)),yticklabels=[g[0] for g in groups],xlabel='False positive rate (%)',title='868 benign executables',xlim=(0,45))
    axs[1].set(xlabel='Ransomware recall (%)',title='30 ransomware executables',xlim=(0,100))
    for ax in axs:ax.grid(axis='x',alpha=.2)
    axs[0].invert_yaxis();f.tight_layout();f.savefig(FIG/'external_transfer_intervals.png',dpi=300);plt.close(f)
    pd.DataFrame(rows).to_csv(OUT/'external_transfer_figure_data.csv',index=False)
    print('Rendered updated architecture and external transfer intervals')

if __name__=='__main__':main()
