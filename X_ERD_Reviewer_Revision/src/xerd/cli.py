from pathlib import Path
import argparse,json,warnings,time
import joblib
from .io import load,save,environment
from . import experiments

def main():
    p=argparse.ArgumentParser(description='X-ERD reviewer revision; empirical results, no target-score fitting')
    p.add_argument('stage',choices=['core','cv','thresholds','corruption','temporal','strict_temporal','ablation','neural','explain','parser','mapping','operational','figures','all'])
    p.add_argument('--data',type=Path,default=Path('data/mlran'))
    p.add_argument('--raw',type=Path,default=Path('data/raw_reports'))
    p.add_argument('--raw-features',type=Path,default=Path('data/raw_features'))
    p.add_argument('--out',type=Path,default=Path('results'))
    p.add_argument('--schema',type=Path,default=Path('config/mapping_schema.json'))
    p.add_argument('--jobs',type=int,default=4)
    p.add_argument('--repetitions',type=int,default=10)
    p.add_argument('--annotations',type=Path)
    args=p.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    warnings.filterwarnings('ignore',message='X does not have valid feature names')
    ds=load(args.data);save(args.out/'environment.json',environment())
    stages=['core','cv','thresholds','corruption','temporal','strict_temporal','ablation','neural','explain','parser','mapping','operational','figures'] if args.stage=='all' else [args.stage]
    for stage in stages:
        start=time.time();print('START',stage,flush=True)
        if stage=='core':experiments.core(ds,args.out,args.jobs)
        elif stage=='cv':experiments.cv(ds,args.out,args.jobs,args.repetitions)
        elif stage=='temporal':experiments.temporal(ds,args.out,args.jobs)
        elif stage=='strict_temporal':
            from .additional import strict_temporal
            strict_temporal(ds,args.raw_features,args.out,args.jobs)
        elif stage=='neural':
            from .additional import neural_control
            neural_control(ds,args.out)
        else:
            m=joblib.load(args.out/'model.joblib')
            if stage=='thresholds':experiments.thresholds(ds,m,args.out,args.jobs)
            elif stage=='ablation':
                from .additional import ablation
                ablation(ds,m,args.out,args.jobs)
            elif stage=='corruption':experiments.corruption(ds,m,args.out)
            elif stage=='explain':
                from .explain import explain
                explain(ds,m,args.out)
            elif stage=='parser':
                from .parser import evaluate
                evaluate(ds,args.raw,args.out,args.jobs)
            elif stage=='mapping':
                from .mapping import evaluate
                evaluate(ds,m,args.out,args.schema,args.annotations)
            elif stage=='operational':
                from .operational import costs,latency
                costs(args.out);latency(ds,m,args.out)
            elif stage=='figures':
                from .figures import generate
                generate(ds,args.out)
        save(args.out/(stage+'_execution.json'),{'stage':stage,'status':'complete','elapsed_seconds':time.time()-start})
        print('DONE',stage,round(time.time()-start,1),'seconds',flush=True)

if __name__=='__main__':main()
