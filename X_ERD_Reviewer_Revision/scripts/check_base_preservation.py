"""Trace necessary reviewer corrections to the supplied paper's core components."""
from pathlib import Path
import hashlib,json,zipfile

ROOT=Path(__file__).resolve().parents[1]

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    source=ROOT/'manuscript/supplied_reference_only.tex';text=source.read_text(encoding='utf-8')
    components=[
      ('Dataset and feature representation','Official MLRan 3905/975 split; 483 indicators retained for the main benchmark.','Additional past-only selection and external cohorts are separate experiments.',['R1.6','R1.10','R4.3']),
      ('Stacking classifier','Random Forest, XGBoost and LightGBM base families and logistic meta-learner retained; all pairwise heads reported.','OOF/group isolation corrected; previously evaluated passthrough and benign-domain weighting are explicitly separate variants.',['R1.2','R2.5','R3.2']),
      ('Dynamic ingestion','Cuckoo JSON to the same fixed feature dictionary.','Exact-token/parser corrections and larger raw-report validation; observed telemetry is never fabricated.',['R1.3','R1.m2','R2.12','R4.1']),
      ('Explanation','SHAP-based behavioral attribution retained.','Common probability scale and logistic coefficients replace the invalid uniform-surrogate explanation of the deployed score.',['R2.8','R2.m3']),
      ('ATT&CK knowledge engine','Decoupled versioned feature-to-technique rules and general footprint category retained.','Candidates distinguished from execution-level evidence; source concordance is not presented as independent human verification.',['R1.4','R1.g4','R2.10','R4.4']),
      ('Statistical and threshold evaluation','Matched operating points, calibration, repeated folds and threshold sensitivity retained.','Corrected inference, honest uncertainty and group sensitivity replace unsupported superiority claims.',['R1.7','R1.9','R2.1','R2.2','R2.3','R3.3']),
      ('Robustness and case illustration','Telemetry masking, deletion analysis, latency and the illustrative case remain part of the study.','Feature-space perturbation is distinguished from an executable attack; prevention is not inferred without encryption-onset labels.',['R1.14','R1.16','R2.11','R4.2','R4.5']),
      ('Literature and comparison','Behavioral ransomware detection and explainability remain the research problem.','Requested published-detector adaptations, actual sequence models and external assessments are added as benchmarks, not substituted for X-ERD.',['R1.1','R2.13','R3.1','R4.6','R4.9'])]
    for marker in ['Random Forest','XGBoost','LightGBM','SHAP','MLRan','483','MITRE']:
        assert marker in text,marker
    original=['src/xerd/models.py','src/xerd/parser.py','results/core.json','results/test_predictions.csv']
    old=ROOT/'releases/X_ERD_Reviewer_Revision_v0.2.0_Source.zip'
    preserved={}
    if old.exists():
        with zipfile.ZipFile(old) as z:
            for name in original:
                before=hashlib.sha256(z.read(name)).hexdigest();after=sha(ROOT/name);assert before==after,name
                preserved[name]=after
    else:
        expected=json.loads((ROOT/'docs/base_paper_change_map.json').read_text())['unchanged_since_prior_verified_revision']
        for name in original:
            after=sha(ROOT/name);assert after==expected[name],name;preserved[name]=after
    inputs={str(p):sha(p) for p in [source,ROOT.parent/'main_scirep.tex',ROOT.parent/'main_scirep.pdf',Path('C:/Users/hp/Downloads/main_scirep.pdf'),Path('C:/Users/hp/Downloads/Comments.docx')] if p.exists()}
    result={'scope':'Core-method preservation and reviewer-motivated change trace; original source and original-setting predictions are preserved. New variants do not replace the original results.','inputs_sha256':inputs,'unchanged_since_prior_verified_revision':preserved,'components':[dict(component=c,preserved=p,necessary_change=n,reviewer_items=ids) for c,p,n,ids in components]}
    (ROOT/'docs/base_paper_change_map.json').write_text(json.dumps(result,indent=2))
    lines=['BASE PAPER PRESERVATION AND REVIEWER CHANGE MAP','',result['scope'],'']
    for c,p,n,ids in components:lines.extend([c,'Preserved: '+p,'Reviewer correction/addition: '+n,'Linked items: '+', '.join(ids),''])
    lines+=['Original base-model code and original official-test scores/predictions remain byte-identical to the earlier verified revision.','The supplied manuscript files outside X_ERD_Reviewer_Revision are not edited.','The original182 external benign reports are disclosed development data only for the new adaptation; the adverse zero-shot results are retained.']
    (ROOT/'docs/BASE_PAPER_CHANGE_MAP.txt').write_text('\n'.join(lines),encoding='utf-8');print('Verified core preservation:',len(components),'components; original code/results unchanged')

if __name__=='__main__':main()
