from pathlib import Path
import json,time,platform
import numpy as np
import pandas as pd
from .io import save

def costs(out):
    out=Path(out);core=json.loads((out/'core.json').read_text());thresholds=json.loads((out/'thresholds.json').read_text())
    rows=[]
    for name,values in thresholds.items():
        for policy,key in [('default','test_default'),('validation_constrained','test_val_constrained'),('validation_f1max','test_val_f1max')]:
            r=values[key]
            for prevalence in [.001,.01,465/975]:
                for ratio in [.1,1,5,10,20,50,100,500,1000]:
                    fp=10000*(1-prevalence)*r['fpr'];fn=10000*prevalence*(1-r['recall'])
                    rows.append({'model':name,'policy':policy,'threshold':r['theta'],'prevalence':prevalence,
                        'fn_to_fp_cost_ratio':ratio,'expected_fp_per10000':fp,'expected_fn_per10000':fn,
                        'cost_per10000_in_fp_cost_units':fp+ratio*fn})
    pd.DataFrame(rows).to_csv(out/'operational_costs.csv',index=False)
    save(out/'operational_cost_protocol.json',{'formula':'10000*((1-prevalence)*FPR+cost_ratio*prevalence*FNR)',
       'scope':'sensitivity over stated cost ratios/prevalences and candidate policies; rates estimated on benchmark, not an enterprise deployment measurement or universal optimum',
       'thresholds':'only training-validation policies are evaluated; test data is not used to refit a threshold'})

def latency(ds,m,out):
    # Computational latency only. No sandbox execution or encryption onset is timed.
    rows=[]
    for n in [1,975]:
        X=ds.Xt[:n]
        for _ in range(3):m.probabilities(X)
        durations=[]
        for _ in range(30):
            start=time.perf_counter();m.probabilities(X);durations.append(time.perf_counter()-start)
        rows.append({'batch_n':n,'median_batch_ms':float(np.median(durations)*1000),
          'p95_batch_ms':float(np.quantile(durations,.95)*1000),'median_per_record_ms':float(np.median(durations)*1000/n)})
    save(Path(out)/'computational_latency.json',{'operation':'three base predictions plus four prefit logistic heads',
       'repetitions':30,'warmup':3,'processor':platform.processor(),'platform':platform.platform(),'batches':rows,
       'scope':'container computational inference only, excludes sandbox execution, report parsing, SHAP and encryption timing; not a prevention deadline'})
