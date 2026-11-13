"""Replay training in a new output directory without overwriting frozen evidence."""
from pathlib import Path
import argparse,json,shutil,warnings
import numpy as np
import pandas as pd
from xerd.remediation import train,evaluate
from xerd.io import digest,save

ROOT=Path(__file__).resolve().parents[1]

def main():
    warnings.filterwarnings('ignore',message='X does not have valid feature names')
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,default=ROOT/'results/remediation_reproduction');ap.add_argument('--resume-evaluation',action='store_true');a=ap.parse_args();out=a.output.resolve()
    reference=ROOT/'results/remediation'
    if out==reference.resolve() or (out.exists() and not a.resume_evaluation):raise ValueError('Choose a new output directory; frozen reference evidence is preserved')
    for p in [ROOT/'results/model.joblib',ROOT/'results/extended/residual_model.joblib']:
        if not p.exists():raise FileNotFoundError('Reproduce the original and passthrough fits first: '+str(p))
    frozen=json.loads((reference/'frozen_model.json').read_text())
    if digest(reference/'development_vectors.csv')!=frozen['development_vectors_sha256']:raise ValueError('Development reference hash differs')
    if digest(ROOT/'config/raw_remediation_protocol.json')!=frozen['protocol_sha256']:raise ValueError('Protocol differs')
    if not a.resume_evaluation:
        out.mkdir(parents=True);shutil.copy2(reference/'development_vectors.csv',out/'development_vectors.csv');train(ROOT,out)
    else:
        if not (out/'frozen_model.json').exists():raise FileNotFoundError('No completed replay fit to resume')
    evaluate(ROOT,out)
    for name in ['training_oof_predictions','official_test_predictions','holdout_predictions']:
        old=pd.read_csv(reference/(name+'.csv'),float_precision='round_trip');new=pd.read_csv(out/(name+'.csv'),float_precision='round_trip')
        assert list(old)==list(new) and len(old)==len(new)
        for col in old:
            if pd.api.types.is_numeric_dtype(old[col]):assert np.allclose(old[col],new[col],atol=1e-12,rtol=0,equal_nan=True),col
            else:assert old[col].equals(new[col]),col
    save(out/'reproduction.json',{'status':'passed','scope':'Replay of published frozen protocol; not a newly independent prospective study','reference_model_sha256':frozen['model_sha256'],'verified_predictions':['training_oof_predictions','official_test_predictions','holdout_predictions']})
    print('Deterministic remediation replay verified',flush=True)

if __name__=='__main__':main()
