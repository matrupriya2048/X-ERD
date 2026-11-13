"""Assemble the author handoff only after real container and document checks."""
from pathlib import Path
import hashlib,html,json,re
from pypdf import PdfReader

ROOT=Path(__file__).resolve().parents[1]

def read(p):return json.loads((ROOT/p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    qa=read('docs/manuscript_qa.json');rows=read('docs/reviewer_revision_status.json');h=read('results/remediation/holdout.json');r=read('results/remediation/ransomware.json');o=read('results/remediation/official_test.json')['models']['X-ERD']
    for key,name in [('main_scirep_revised','main_scirep_revised'),('supplementary_information','supplementary_information')]:
        assert qa[key]['pdf_sha256']==sha(ROOT/'manuscript'/(name+'.pdf'))
        assert qa[key]['visual_inspection'].startswith('Completed')
    response=ROOT/'docs/response_letter.pdf';response_log=(ROOT/'docs/response_letter.log').read_text(errors='replace')
    assert not re.search(r'Overfull|undefined references|LaTeX Error',response_log)
    response_qa=read('docs/response_letter_qa.json');assert response_qa['pdf_sha256']==sha(response) and response_qa['visual_inspection'].startswith('Completed')
    verified={k:read(p) for k,p in {'source':'results/packaged_source_verification.json','core':'results/release_verification.json','extended':'results/extended/verification.json','remediation':'results/remediation/verification.json'}.items()}
    assert all(v['status']=='passed' for v in verified.values()) and verified['source']['version']=='0.3.0'
    no_data=(ROOT/'results/docker_packaged_tests_v03.log').read_text();data=(ROOT/'results/docker_packaged_data_tests_v03.log').read_text()
    assert re.search(r'25 passed, 1 skipped',no_data) and re.search(r'26 passed',data)
    replay=read('results/remediation_reproduction/reproduction.json');assert replay['status']=='passed'
    reference_hash=read('results/remediation/frozen_model.json')['model_sha256']
    assert sha(ROOT/'results/remediation/model.joblib')==reference_hash
    assert sha(ROOT/'results/remediation_reproduction/model.joblib')==reference_hash
    image=read('results/docker_image_v03.json')[0]['Id'];assert len(rows['items'])==69
    n=h['evaluated_unique_n'];b=h['models']['Development-adapted']['X-ERD'];previous=h['models']['Previous passthrough']['X-ERD'];q=r['models']['Development-adapted']['X-ERD']
    result={'status':'passed','version':'0.3.0','image_id':image,'packaged_unit_and_integration_tests':26,'packaged_without_data':{'passed':25,'skipped_data_dependent':1},'source_manifest_files':verified['source']['files_checked'],'core_artifact_checks':verified['core'].get('checks_count',len(verified['core'].get('checks',[]))),'extended_artifact_checks':verified['extended']['checks_count'],'remediation_artifact_checks':verified['remediation']['checks'],'deterministic_retraining':'Identical serialized model hash and predictions; independent replay output preserves reference results','reviewer_items':69,'document_pages':{'main':qa['main_scirep_revised']['pages'],'supplement':qa['supplementary_information']['pages'],'response':len(PdfReader(response).pages)},'document_qa':'docs/manuscript_qa.json','response_qa':'docs/response_letter_qa.json','scope':'Software, recorded experiment and document consistency; not human adjudication, universal superiority, zero-error deployment or reviewer acceptance','source_manifest_sha256':sha(ROOT/'config/source_integrity.json')}
    (ROOT/'docs/final_verification.json').write_text(json.dumps(result,indent=2))
    lines=['FINAL AUTHOR REVIEW - X-ERD REVISION 0.3.0','',
      'Core paper preserved: official benchmark, 483 indicators, RF/XGBoost/LightGBM, logistic stacking, parser, SHAP and candidate ATT&CK mapping. Original-setting code and predictions remain unchanged; extensions are separate. See BASE_PAPER_CHANGE_MAP.txt.',
      '', 'Measured repair of the external false-alarm problem',
      '- The original 182-report result remains: original full stack 95 false alarms (52.20%); previous passthrough 92 (50.55%).',
      '- Those 182 reports became disclosed development data. Eighteen nested executable-group-CV configurations preserve a source-recall constraint and fixed 0.50 threshold.',
      f'- A model frozen before new scoring was evaluated on 400 newly selected paths; {n} eligible unique benign executions remained after all recorded exclusions.',
      f'- On these identical new targets: previous passthrough {previous["fp"]}/{n} false alarms ({100*previous["fpr"]:.2f}%); adapted full stack {b["fp"]}/{n} ({100*b["fpr"]:.2f}%).',
      f'- Adapted FPR exact 95% interval: [{100*b["fpr_ci95"][0]:.2f}, {100*b["fpr_ci95"][1]:.2f}]%. This is same-collection adaptation, not a fresh environment.',
      f'- Separate raw ransomware collection: {r["listed_reports"]} listed, {r["acquired_reports"]} acquired, {r["evaluated_unique_n"]} eligible unique executions; {q["tp"]} detected ({100*q["recall"]:.2f}% recall). The previous passthrough detects 56/58, so this adaptation has one additional miss and no demonstrated recall non-inferiority. Provider access exclusions are disclosed.',
      f'- Reused official test: {100*o["accuracy"]:.2f}% accuracy, {o["tp"]}/465 ransomware detections, {o["fp"]}/510 false alarms. This test is a regression check, not independent prospective confirmation.',
      '- Provenance audit: 134 reports with target hashes contain only 77 distinct hashes; 63/134 recorded scores (26/58 eligible targets) are below the publication\'s stated score threshold. Reasons are unknown; no labels or scores are altered to improve results.',
      '- Post-hoc feature-vector audit: nine of 362 benign vectors equal fitting vectors. Excluding them descriptively gives six false alerts among 353 novel vectors (1.70%); the primary cohort and model remain unchanged.',
      '- The adapted RF+XGB pair has a stronger point estimate (4/362 benign false alerts, 55/58 ransomware detections). The full stack remains the frozen primary model; no test-driven reselection or necessity of the third learner is claimed.',
      '- Unrelated benign and ransomware sources are not pooled into an overall accuracy, precision or AUC estimate.',
      '', 'Verification',
      f'- Final Docker image: {image}. All 26 packaged tests pass with benchmark data; 25 pass and one data-dependent test skips without data.',
      f'- Reconciled {result["source_manifest_files"]} packaged source/configuration files, {result["core_artifact_checks"]} core checks, {result["extended_artifact_checks"]} extended checks and {result["remediation_artifact_checks"]} adaptation checks.',
      '- Fresh retraining reproduces the exact serialized model SHA-256. Old and new models are scored on identical target vectors; all accepted, overlapping, duplicate, unprocessed and unavailable reports are accounted for.',
      '- Refitted SHAP/logit identity, deletion sensitivity, targeted masking and mapping coverage are recomputed. The mapper rules and original reference-concordance evidence remain separate from independent human verification.',
      f'- Main paper {result["document_pages"]["main"]} pages; supplement {result["document_pages"]["supplement"]}; response {result["document_pages"]["response"]}, covering all 69 editor/four-reviewer items. Compiled references and rendered pages were checked.',
      '', 'Scope that remains unresolved',
      '- Independent human ATT&CK adjudication and verified encryption-onset annotations are unavailable. The paper does not claim either was completed.',
      '- Historical, publisher-labeled and provenance-limited collections do not establish arbitrary-environment, unseen-family or enterprise performance. False alarms and missed detections are disclosed as measured.',
      '- Reviewer acceptance and zero future errors cannot be guaranteed from these experiments.',
      '- GitHub publication, a matching versioned Zenodo DOI and journal submission remain the author-controlled final steps. Nothing has been pushed, uploaded or submitted.',
      '', 'Read Complete_Revision_Report.html for every reviewer item and RELEASE_AND_SUBMISSION_STEPS.txt for publishing the verified version.']
    (ROOT/'docs/FINAL_REVIEW_FINDINGS.txt').write_text('\n'.join(lines),encoding='utf-8')
    table_rows=[]
    for label in h['models']:
        bv=h['models'][label]['X-ERD'];rv=r['models'][label]['X-ERD']
        table_rows.append(f'<tr><td>{html.escape(label)}</td><td>{bv["fp"]}/{n} ({100*bv["fpr"]:.2f}%)</td><td>{rv["tp"]}/{r["evaluated_unique_n"]} ({100*rv["recall"]:.2f}%)</td></tr>')
    page='<!doctype html><meta charset="utf-8"><title>X-ERD raw-report repair</title><style>body{font:16px/1.6 system-ui;max-width:1000px;margin:40px auto;padding:0 24px;color:#172c36}h1,h2{color:#145b56}table{border-collapse:collapse;width:100%}th,td{padding:12px;border-bottom:1px solid #ccd7da;text-align:left}th{background:#edf5f4}.note{background:#f4f6f8;padding:18px}pre{white-space:pre-wrap;font:inherit}</style><h1>X-ERD: verified raw-report repair</h1><p>The original detector, parser and explanation idea are preserved. The adaptation is separately documented and tested.</p><table><tr><th>Frozen full model</th><th>New benign false alarms / FPR</th><th>Separate ransomware detections / recall</th></tr>'+''.join(table_rows)+'</table><p class="note">These are two different collections. Rates must not be pooled for overall accuracy or precision. The benign collection is familiar after disclosed adaptation; the ransomware corpus has documented curation discrepancies.</p><pre>'+html.escape('\n'.join(lines))+'</pre>'
    (ROOT/'docs/RAW_FPR_REMEDIATION_REPORT.html').write_text(page,encoding='utf-8')
    guide=ROOT/'docs/RELEASE_AND_SUBMISSION_STEPS.txt';text=guide.read_text(encoding='utf-8').replace('â€”','-')
    anchor='3. The software now includes'
    insertion=f'   The separately frozen version 0.3 adaptation now yields {b["fp"]}/{n} false alarms ({100*b["fpr"]:.2f}%) on new eligible same-collection reports, compared with {previous["fp"]}/{n} ({100*previous["fpr"]:.2f}%) for the previous variant. Separate published raw-ransomware sensitivity is {q["tp"]}/{r["evaluated_unique_n"]}. Preserve all protocols and uncertainty in the submission.\n\n'
    if insertion not in text:text=text.replace(anchor,insertion+anchor,1)
    guide.write_text(text,encoding='utf-8')
    ledger=read('docs/manuscript_change_ledger.json');entry={'component':'Disclosed raw-report remediation','change':'Retained original results; added constrained benign adaptation, a frozen new same-collection holdout, separately published raw ransomware sensitivity, refitted attribution/component checks and exact retraining replay. Core method preserved.','reviewer_items':['R1.3','R1.6','R2.12','R2.13','R3.4','R4.1','R4.3']}
    if not any(x['component']==entry['component'] for x in ledger['sections']):ledger['sections'].append(entry)
    (ROOT/'docs/manuscript_change_ledger.json').write_text(json.dumps(ledger,indent=2));print(json.dumps(result,indent=2))

if __name__=='__main__':main()
