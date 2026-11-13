"""Additional experiments requested during the second reviewer revision."""
from pathlib import Path
import argparse,time,warnings
from .io import load,save,environment

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('stage',choices=['published','external_dataset','external_mapping','sequences','external_raw','optimization','group_cv','optimized_validation','residual','residual_validation','external_comparators'])
    parser.add_argument('--root',type=Path,default=Path('.'))
    args=parser.parse_args();root=args.root;out=root/'results'/'extended';out.mkdir(parents=True,exist_ok=True)
    warnings.filterwarnings('ignore',message='X does not have valid feature names')
    ds=load(root/'data'/'mlran');start=time.time()
    save(out/(args.stage+'_environment.json'),environment())
    print('START',args.stage,flush=True)
    if args.stage=='published':
        from .published import run
        run(ds,out)
    elif args.stage=='external_dataset':
        from .external_dataset import run
        run(ds,root/'data'/'external_sources',out)
    elif args.stage=='external_mapping':
        from .external_mapping import run
        run(ds,root,out)
    elif args.stage=='sequences':
        from .sequences import run
        run(ds,root,out)
    elif args.stage=='external_raw':
        from .external_raw import run
        run(ds,root,out)
    elif args.stage=='optimization':
        from .optimize import run
        run(ds,root,out)
    elif args.stage=='group_cv':
        from .group_cv import run
        run(ds,root,out)
    elif args.stage=='optimized_validation':
        from .optimized_validation import run
        run(ds,root,out)
    elif args.stage=='residual':
        from .residual import run
        run(ds,root,out)
    elif args.stage=='residual_validation':
        from .residual_validation import run
        run(ds,root,out)
    elif args.stage=='external_comparators':
        from .external_comparators import run
        run(ds,root,out)
    save(out/(args.stage+'_execution.json'),{'status':'complete','seconds':time.time()-start})
    print('DONE',args.stage,round(time.time()-start,1),flush=True)
if __name__=='__main__':main()
