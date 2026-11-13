"""Expose available evidence without inventing assessors or event ground truth."""
from pathlib import Path
import collections, csv, hashlib, html, json, math

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent/'X_ERD_Reviewer_Revision'

def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1048576),b''):
            h.update(block)
    return h.hexdigest()

def save(name,data):
    (HERE/name).write_text(json.dumps(data,indent=2,allow_nan=False),encoding='utf-8')

def write_csv(name,fields,rows):
    with (HERE/name).open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(rows)

def inspect_onset(report):
    is_sequence=isinstance(report,list)
    b=(report.get('behavior') or {}) if isinstance(report,dict) else {'processes':[{'calls':report}]} if is_sequence else {}
    numeric=[];other_time_n=0;crypto=[];writes=[]
    for pi,proc in enumerate(b.get('processes') or []):
        for ci,call in enumerate(proc.get('calls') or []):
            if not isinstance(call,dict):continue
            timestamp=call.get('time');api=str(call.get('api',''))
            usable=isinstance(timestamp,(int,float)) and not isinstance(timestamp,bool) and math.isfinite(timestamp)
            if usable:numeric.append(timestamp)
            elif timestamp is not None:other_time_n+=1
            evidence={'json_pointer':f'/{ci}' if is_sequence else f'/behavior/processes/{pi}/calls/{ci}','api':api,'pid':proc.get('pid'),'time':timestamp if usable else None,'status':call.get('status')}
            if api.lower() in {'cryptencrypt','bcryptencrypt','ncryptencrypt'}:crypto.append(evidence)
            if api.lower() in {'ntwritefile','writefile','writefileex'}:writes.append(evidence)
    hints=sorted([c for c in crypto if c['time'] is not None],key=lambda x:x['time'])
    return {'input_format':'report_object' if isinstance(report,dict) else 'sequence_only_array',
            'numeric_timestamped_calls':len(numeric),'other_timestamp_format_calls':other_time_n,
            'first_numeric_time':min(numeric) if numeric else None,'last_numeric_time':max(numeric) if numeric else None,
            'crypto_api_call_n':len(crypto),'write_api_call_n':len(writes),
            'first_timestamped_crypto_hint':hints[0] if hints else None,
            'ground_truth_status':'unavailable_in_evaluated_annotation_inputs',
            'hint_is_verified_encryption_onset':False,
            'reason':'No matched, independently sourced first destructive encrypted-write annotation or before/after file-content verification is supplied for this execution. API calls are candidate evidence only.'}

def run():
    mapping=json.loads((ROOT/'results/extended/mapping_external.json').read_text())['cuckoo']['reports']
    source_by_id={str(r['sample_id']):r for r in mapping}
    packets=HERE/'audit_packets';packets.mkdir(exist_ok=True)
    rows=[];key=[];onsets=[]
    paths=sorted((ROOT/'data/raw_reports').glob('*.json'))+sorted((ROOT/'data/external_reports').glob('*.json'))
    for i,path in enumerate(paths):
        d=json.loads(path.read_text(encoding='utf-8'));sha=digest(path)
        onsets.append({'source_path':str(path.relative_to(ROOT)).replace('\\','/'),'report_sha256':sha,**inspect_onset(d)})
        if path.parent.name!='raw_reports':continue
        uid='EXEC-'+hashlib.sha256(sha.encode()).hexdigest()[:12]
        b=d.get('behavior') or {}
        sigs=[]
        for si,s in enumerate(d.get('signatures') or []):
            sigs.append({'json_pointer':f'/signatures/{si}','name':s.get('name'),'description':s.get('description'),
                         'marks':s.get('marks',[])[:8],'marks_included':min(8,len(s.get('marks') or [])),
                         'total_marks':len(s.get('marks') or [])})
        calls=[]
        for pi,p in enumerate(b.get('processes') or []):
            for ci,c in enumerate(p.get('calls') or []):
                if isinstance(c,dict) and str(c.get('api','')).lower() in {'ntsetcontextthread','createremotethread','ntcreatethreadex','ldrgetprocedureaddress','cryptencrypt','bcryptencrypt'}:
                    calls.append({'json_pointer':f'/behavior/processes/{pi}/calls/{ci}','pid':p.get('pid'),**c})
        payload={'execution_id':uid,'report_sha256':sha,'full_report_relative_path':'../../X_ERD_Reviewer_Revision/'+str(path.relative_to(ROOT)).replace('\\','/'),
                 'summary':b.get('summary',{}),'apistats':b.get('apistats',{}),'signature_evidence':sigs,
                 'selected_calls':calls[:80],'matching_call_n':len(calls),
                 'scope':'Triage extract only; consult full raw report for complete context. Native sandbox descriptions may expose vendor judgments. X-ERD predictions are omitted.',
                 'verified_human_annotation':False}
        save('audit_packets/'+uid+'.json',payload)
        rows.append({'execution_id':uid,'evidence_file':'audit_packets/'+uid+'.json','human_assessor_id':'',
                     'human_assessor_affiliation':'','independence_declaration':'','attack_version':'',
                     'supported_techniques':'','unsupported_candidates':'','insufficient_context':'',
                     'evidence_pointers':'','rationale':'','reviewed_utc':''})
        key.append({'execution_id':uid,'sample_id':path.stem,'report_sha256':sha,
                    'frozen_mapper_output':source_by_id.get(path.stem,{}).get('predicted',[]),
                    'sandbox_reference_output':source_by_id.get(path.stem,{}).get('external_annotations',[]),
                    'not_for_blinded_assessor':True})
        if (i+1)%20==0:print('Evidence inspected',i+1,'of',len(paths),flush=True)
    write_csv('human_review_blank.csv',list(rows[0]),rows)
    save('scoring_key_not_for_assessor.json',key)
    save('onset_evidence_audit.json',onsets)
    summary={'report_n':len(onsets),'mapping_packet_n':len(rows),'completed_human_reviews':0,
             'reports_with_numeric_timestamps':sum(r['numeric_timestamped_calls']>0 for r in onsets),
             'reports_with_crypto_api_calls':sum(r['crypto_api_call_n']>0 for r in onsets),
             'reports_with_timestamped_crypto_hints':sum(r['first_timestamped_crypto_hint'] is not None for r in onsets),
             'reports_with_write_api_calls':sum(r['write_api_call_n']>0 for r in onsets),
             'verified_onset_annotations':0,
             'input_formats':dict(collections.Counter(r['input_format'] for r in onsets)),
             'interpretation':'Data-readiness audit of existing 20 benchmark reports and 80 external JSON inputs. Sequence-only arrays are not treated as full Cuckoo reports. Zero supplied verified annotations is not evidence that encryption did not occur.',
             'source_sha256':digest(Path(__file__))}
    save('annotation_readiness.json',summary)
    print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__':run()
