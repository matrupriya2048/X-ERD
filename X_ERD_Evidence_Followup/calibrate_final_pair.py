from pathlib import Path
import joblib,numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from xerd.io import load,save,digest
from xerd.metrics import metrics
from primary_pair import probabilities
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent/'X_ERD_Reviewer_Revision';OUT=HERE/'multicollection'
ds=load(ROOT/'data/mlran');m=joblib.load(OUT/'model.joblib')
pred={n:b.predict_proba(ds.Xt)[:,1] for n,b in m['base'].items()};pred['Final pair']=probabilities(m,ds.Xt)
assert np.allclose(pred['Final pair'],pd.read_csv(OUT/'official_reused_regression_predictions.csv').probability,rtol=0,atol=1e-12)
fig,axes=plt.subplots(1,2,figsize=(9,3.6))
for name,p in pred.items():
    observed,mean=calibration_curve(ds.yt,p,n_bins=10,strategy='uniform');axes[0].plot(mean,observed,'o-',label=name)
axes[0].plot([0,1],[0,1],'--',color='gray');axes[0].set(xlabel='Mean predicted probability',ylabel='Observed ransomware fraction',title='10 equal-width reliability bins');axes[0].legend()
for label,name in [(0,'Benign'),(1,'Ransomware')]:axes[1].hist(pred['Final pair'][ds.yt==label],bins=np.linspace(0,1,11),alpha=.65,label=name)
axes[1].set(xlabel='Final-pair probability',ylabel='Records',title='Reused official regression test');axes[1].legend();fig.tight_layout();fig.savefig(OUT/'calibration.png',dpi=220);plt.close(fig)
save(OUT/'calibration.json',{'model_sha256':digest(OUT/'model.joblib'),'n':975,'metrics':{n:metrics(ds.yt,p) for n,p in pred.items()},'scope':'Reused official regression test, not deployment calibration or an additional independent validation. EN benign-only cohorts cannot establish full class-prevalence calibration.','script_sha256':digest(Path(__file__))})
print('Final RF/LightGBM/pair calibration metrics and reliability plot completed.')
