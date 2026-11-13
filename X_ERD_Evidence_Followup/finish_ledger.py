from pathlib import Path
import collections,csv,hashlib,html,json
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent/'X_ERD_Reviewer_Revision'
def run():
    data=json.loads((HERE/'docs/reviewer_revision_status.json').read_text());index=json.loads((HERE/'docs/response_letter_index.json').read_text())
    assert len(data['items'])==len(index['items'])==69
    assert {r['id'] for r in data['items']}=={r['id'] for r in index['items']}
    en=json.loads((HERE/'closure_results/en2025_results.json').read_text())
    assert json.loads((HERE/'closure_results/en2025_verification.json').read_text())['status']=='passed'
    for row in data['items']:
        for evidence in row['evidence']:assert (ROOT/evidence).exists() or (HERE/evidence).exists(),evidence
        if row['id'] in {'R1.2','R2.5','R3.2'}:row['status']='Implemented: reviewer-requested primary RF+LightGBM pair'
        if row['id']=='R4.4':row['status']='Implemented: many-to-one schema and numeric rule-specificity scoring'
        if row['id'] in {'R1.3','R1.m2','R1.g3','R2.12','R4.1','R3.1'}:
            row['additional_evidence']=['closure_results/en2025_results.json','closure_results/en2025_audit.json','closure_results/en2025_verification.json']
        if row['id'] in {'R1.4','R1.g4','R2.10','R3.4'}:row['current_task_note']='Human adjudication excluded from this work round by the author; automated and external-reference results remain available.'
    states={r['id']:r['status'] for r in data['items']}
    for row in index['items']:row['status']=states[row['id']]
    data['counts']=dict(collections.Counter(states.values()))
    data['verdict']='All 69 comments tracked. Primary-model alternative, indicator reliance, window controls and additional independent benign collection completed. Human adjudication deferred; verified encryption onset unavailable; author-controlled release remains final.'
    data['remaining']=['Independent human adjudication: excluded from this round by the author.',
                       'Verified encryption-onset ground truth is still unavailable. The new reliance and window analyses do not establish pre-encryption intervention.',
                       'GitHub/Zenodo publication and final co-author sign-off remain author-controlled final actions.']
    (HERE/'docs/reviewer_revision_status.json').write_text(json.dumps(data,indent=2));(HERE/'docs/response_letter_index.json').write_text(json.dumps(index,indent=2))
    seqfile=ROOT/'results/extended/sequence_oof_predictions.csv';seq=[r for r in csv.DictReader(seqfile.open()) if r['model']=='RF+LGB']
    assert len(seq)==80
    fp=sum(r['y_true']=='0' and float(r['probability'])>=.5 for r in seq);tp=sum(r['y_true']=='1' and float(r['probability'])>=.5 for r in seq)
    (HERE/'closure_results/primary_sequence_summary.json').write_text(json.dumps({'n':80,'fp':fp,'tp':tp,'fn':60-tp,'tn':20-fp,'accuracy':(20-fp+tp)/80,'source_sha256':hashlib.sha256(seqfile.read_bytes()).hexdigest()},indent=2))
    body='<h1>Comment-by-comment completion ledger</h1><p>'+html.escape(data['verdict'])+'</p><p>Remaining dependencies: '+html.escape(' '.join(data['remaining']))+'</p><table><tr><th>Comment</th><th>Request</th><th>Current status</th></tr>'
    for row in data['items']:body+='<tr><td>'+html.escape(row['id'])+'</td><td>'+html.escape(row['comment_topic'])+'</td><td>'+html.escape(row['status'])+'</td></tr>'
    body+='</table>'
    (HERE/'FINAL_COMMENT_LEDGER.html').write_text('<!doctype html><meta charset="utf-8"><title>X-ERD final comment ledger</title><style>body{font:16px/1.5 system-ui;max-width:1100px;margin:35px auto;padding:0 20px}table{border-collapse:collapse;width:100%}td,th{border:1px solid #bbc4d0;padding:9px;text-align:left;vertical-align:top}th{background:#e8eef5}</style>'+body,encoding='utf-8')
    (HERE/'REMAINING_REVIEWER_WORK.html').write_text('<!doctype html><meta charset="utf-8"><title>Current remaining work</title><p>The previous open-work list has been superseded by the new primary-pair, reliance, window-control and EN2025 experiments.</p><p>See <a href="FINAL_COMMENT_LEDGER.html">the current 69-comment ledger</a>.</p><ul>'+''.join('<li>'+html.escape(x)+'</li>' for x in data['remaining'])+'</ul>',encoding='utf-8')
    (HERE/'REMAINING_REVIEWER_WORK.json').write_text(json.dumps({'superseded_by':'docs/reviewer_revision_status.json','remaining':data['remaining']},indent=2))
    print('Reconciled all 69 comments and verified every listed evidence path.')
if __name__=='__main__':run()
