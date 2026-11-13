from pathlib import Path
import hashlib,html,json,re
HERE=Path(__file__).resolve().parent
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def run():
    result=json.loads((HERE/'multicollection/results.json').read_text())['metrics'];v=json.loads((HERE/'multicollection/verification.json').read_text());qa=json.loads((HERE/'docs/document_qa.json').read_text())
    for name in ['verification.json','final_image_verification.json']:
        assert json.loads((HERE/'multicollection'/name).read_text())['status']=='passed'
    assert v['fresh_retraining_identical_model_sha256'] and v['en_holdout_vector_matches_to_any_fit_row']==0
    assert 'Ran 19 tests' in (HERE/'final_tests.txt').read_text()
    parameters=json.loads((HERE/'multicollection/model_parameters.json').read_text());assert len(parameters['meta_coefficients'])==485
    text=(HERE/'manuscript/main_scirep_revised.tex').read_text()
    abstract=re.search(r'\\begin\{abstract\}(.*?)\\end\{abstract\}',text,re.S).group(1)
    body=text.split(r'\section*{Introduction}',1)[1].split(r'\section*{Methods}',1)[0]
    body=re.sub(r'\\begin\{(figure\*?|table\*?)\}.*?\\end\{\1\}','',body,flags=re.S);body=re.sub(r'\\(?:cite|ref|label)\{[^}]*\}','',body)
    words=len(re.sub(r'\\[A-Za-z]+|[{}$]',' ',body).split());assert words<=4500
    final={'version':'0.4.2','unit_tests_passed':19,'all_reviewer_items':69,'main_words_excluding_methods_displays':words,
           'abstract_words':len(re.sub(r'\\[A-Za-z]+|[{}$]',' ',abstract).split()),'pages':{k:r['pages'] for k,r in qa.items()},
           'frozen_final_model_sha256':digest(HERE/'multicollection/model.joblib'),
           'retrospective_holdout_n':283,'fresh_retraining_exact_hash':True,'source_only_primary':'RF+LightGBM',
           'source_baseline_and_prior_extensions_preserved':True,'metrics':result,
           'remaining':['Independent human adjudication excluded from this work round.','Verified encryption-onset references unavailable; no pre-encryption success claim.','Author-controlled GitHub/Zenodo publication and final co-author review.']}
    (HERE/'FINAL_VERIFICATION.json').write_text(json.dumps(final,indent=2))
    (HERE/'MODEL_CHECKSUMS.json').write_text(json.dumps({str(p.relative_to(HERE)):digest(p) for p in [HERE/'closure_results/primary_pair.joblib',HERE/'multicollection/model.joblib']},indent=2))
    readme='''X-ERD REVIEWER REVISION 0.4.2 — CURRENT WORKING PACKAGE

LATEST FILES
Use this folder's manuscript/main_scirep_revised.pdf,
manuscript/supplementary_information.pdf and docs/response_letter.pdf.
The companion X_ERD_Reviewer_Revision folder preserves the verified v0.3 code,
historical model/result identities and older documents. Its PDFs are superseded
by the current follow-up PDFs. Base code and the supplied original files remain
unchanged.

WHAT IS COMPLETED
- Reviewer-directed RF+LightGBM primary analysis and two-base-only inference.
- Explicit CAPE event/static-string adapter with operation-provenance tests.
- EN2025: pinned published benign archive, 918 non-pilot inputs, one prior-target
  overlap excluded, 917 unique PE targets audited. Initial transport failure
  is retained, including the schema-translated 100% original-model alert rate.
- Fixed-parameter multi-collection pair repair: 3905 source training records,
  182 prior Nunes development controls and 634 CAPE development controls.
  The 283-case vector-group holdout is RETROSPECTIVE, not a fresh collection.
- On that holdout: 1/283 alerts (0.35%, exact95% interval0.009–1.953%).
  Earlier Nunes holdout: 4/362 alerts (1.10%). Separate raw-ransomware targets:
  55/58 detections. Reused official test: 98.46% accuracy,98.71% recall,1.76% FPR.
- Old source-only primary pair detects56/58 raw-ransomware targets, so the new
  pair has one extra miss. No recall non-inferiority is asserted.
- Provider benign labels conflict with50 CAPE Malicious and270 Suspicious
  statuses. All targets remain; no favorable relabeling/filtering is performed.
  Holdout Clean-status sensitivity is1/172 alerts, not independent human gold.
- Own-model probability SHAP and centered raw terms reconstruct1258 outputs
  within1.01e-08. Current calibration metrics, figures and parameters exported.
- Crypto-indicator reliance and all24 sequence-window/seed control groups.
- 19 focused tests; model/prediction replay; identical model SHA under fresh
  retraining; no development/holdout vector overlap; final raw-report Docker CLI.
- Every one of69 tracked comments has a response and evidence links.

CURRENT SCOPE
FINAL_COMMENT_LEDGER.html records every item. Independent human adjudication
is excluded from this round by the author. Verified first destructive encrypted-
write references are unavailable; timestamps/crypto API hints do not establish
pre-encryption efficacy. GitHub/Zenodo publication and author sign-off remain
the final author-controlled actions. Nothing has been uploaded or submitted.

REPRODUCTION ORDER
Keep the two source folders side by side. First reproduce the v0.3 companion
using its instructions, including pinned source data and frozen benchmark models.
Use its scientific dependency locks/Docker image; acquisition also needs the
companion requirements-acquisition.lock. Then in this folder:
  python acquire_mtm.py
  python evaluate_mtm.py
  python evidence_packets.py
  python close_available_work.py
  python primary_pair.py export --root ../X_ERD_Reviewer_Revision --output closure_results/primary_pair.joblib
  python acquire_en2025.py
  python evaluate_en2025.py
  python evaluate_cape_adapter.py
  python train_multicollection_pair.py
  python explain_multicollection.py
  python calibrate_final_pair.py
  python -m unittest test_evidence test_primary_pair test_cape_adapter -v
  python verify_closure.py
  python verify_en2025.py
  python verify_translated.py
  python verify_multicollection.py
Training refuses to overwrite an evaluated fitted model. Use a clean checkout
or a separate reproduction output directory; retain the original artifact hashes.
The acquisition protocol and split are supplied; do not reselect target paths.

DOCKER INFERENCE
Build the companion xerd-reviewer-revision:0.3.0 image, then:
  docker build -f Dockerfile.primary -t xerd-primary-pair:0.4.2 .
The final image supports predict and predict-report commands. Use
multicollection/model.joblib for the final adapted pair. The source-only
closure_results/primary_pair.joblib is a preserved benchmark comparator.
For raw reports, supply --feature-names pointing to the original 483-ID dictionary.
Mount input/output files and run with --network none. The CLI translates declared
CAPE inputs and leaves original Cuckoo inputs unchanged.

DOCUMENT REPRODUCTION (bundled pypdf/Pillow; MiKTeX; Poppler)
  python build_documents.py papers
  python apply_closure_documents.py papers
  python apply_en2025_documents.py papers
  python apply_multicollection_documents.py papers
  python compile_documents.py main
  python compile_documents.py supplement
  python build_documents.py response
  python apply_closure_documents.py response
  python apply_en2025_documents.py response
  python apply_multicollection_documents.py response
  python compile_documents.py response
  python finish_ledger.py
  python document_qa.py
Inspect the rendered pages before marking visual QA complete.

RELEASE CONTENTS
The archive contains code, protocols, results, figures, PDFs and checksums.
Raw reports, local downloaded archives, model binaries and reproduction copies
are excluded. MODEL_CHECKSUMS.json identifies the tested local model artifacts.
Data-provider licenses differ; consult their source records before redistribution.
No malware executable was acquired or run for this work; datasets contain reports.
'''
    (HERE/'README.txt').write_text(readme,encoding='utf-8')
    body='<h1>Verified X-ERD revision 0.4.2</h1><p>Primary pair, CAPE parser compatibility, larger raw controls and a measured multi-collection repair are implemented. The original architecture and all historical comparisons remain traceable.</p>'
    body+='<table><tr><th>Evaluation</th><th>Result</th><th>Interpretation</th></tr>'
    for a,b,c in [('CAPE holdout','1/283 alerts; 0.35% FPR','Retrospective vector-group holdout; 634 development targets'),('Earlier Nunes holdout','4/362 alerts; 1.10% FPR','Previously evaluated source; no fitting of these rows'),('Separate raw ransomware','55/58 detections; 94.83% recall','Publisher labels; one additional miss versus original primary pair'),('Official benchmark','98.46% accuracy; 98.71% recall; 1.76% FPR','Reused regression test')]:body+='<tr><td>'+a+'</td><td>'+b+'</td><td>'+c+'</td></tr>'
    body+='</table><p>The same 283 CAPE targets triggered283 alerts in each original pair/full stack and142 in the earlier three-model adaptation. The repair gives1, with fixed parameters declared before its fitting. The whole collection had already been inspected, so this is not prospective external validation.</p><p>All initial failures, label contradictions and uncertainty remain in the paper. Human adjudication and verified encryption onset are not asserted. See <a href="FINAL_COMMENT_LEDGER.html">all69 reviewer responses and statuses</a>.</p>'
    (HERE/'EVIDENCE_FOLLOWUP_REPORT.html').write_text('<!doctype html><meta charset="utf-8"><title>X-ERD final evidence</title><style>body{font:17px/1.6 system-ui;max-width:1050px;margin:40px auto;padding:0 20px}table{border-collapse:collapse}td,th{border:1px solid #bcc7d0;padding:10px;text-align:left}</style>'+body,encoding='utf-8')
    print(json.dumps({k:final[k] for k in ['version','unit_tests_passed','all_reviewer_items','main_words_excluding_methods_displays','abstract_words','pages']},indent=2))
if __name__=='__main__':run()
