"""External-reference concordance; never described as a human expert audit."""
from pathlib import Path
import json,re
import numpy as np
import pandas as pd
from .mapping import map_features
from .parser import vector
from .io import save,digest

def technique_id(obj):
    return next((x.get('external_id') for x in obj.get('external_references',[]) if x.get('source_name')=='mitre-attack'),None)

class AttackReference:
    def __init__(self,path):
        self.data=json.loads(Path(path).read_text(encoding='utf-8'))
        self.objects={o['id']:o for o in self.data['objects']}
        self.techniques={technique_id(o):o for o in self.objects.values() if o['type']=='attack-pattern' and technique_id(o)}
        self.revocations={o['source_ref']:o['target_ref'] for o in self.objects.values() if o['type']=='relationship' and o.get('relationship_type')=='revoked-by'}
    def resolve(self,tid):
        obj=self.techniques.get(tid);seen=set()
        while obj and obj.get('revoked') and obj['id'] not in seen:
            seen.add(obj['id']);obj=self.objects.get(self.revocations.get(obj['id']))
        if obj and not obj.get('revoked') and not obj.get('x_mitre_deprecated'):return technique_id(obj)
        return None
    def profile(self,name):
        key=re.sub(r'[^a-z0-9]','',str(name).lower())
        found=[]
        for o in self.objects.values():
            if o['type'] not in ['malware','tool']:continue
            aliases=[o.get('name','')]+o.get('x_mitre_aliases',[])
            if key and key in [re.sub(r'[^a-z0-9]','',a.lower()) for a in aliases]:found.append(o)
        ids={o['id'] for o in found};tech=set()
        for o in self.objects.values():
            if o['type']=='relationship' and o.get('relationship_type')=='uses' and o.get('source_ref') in ids:
                target=self.objects.get(o['target_ref'],{})
                tid=technique_id(target)
                if tid and self.resolve(tid):tech.add(self.resolve(tid))
        return found,tech

def agreement(pred,reference):
    pred=set(pred);reference=set(reference);hit=pred&reference
    return {'intersection':sorted(hit),'prediction_only':sorted(pred-reference),'reference_only':sorted(reference-pred),
      'intersection_n':len(hit),'predicted_n':len(pred),'reference_n':len(reference),
      'precision_like_agreement':len(hit)/len(pred) if pred else None,
      'reference_coverage':len(hit)/len(reference) if reference else None,
      'jaccard':len(hit)/len(pred|reference) if pred|reference else None}

def summarize(rows):
    valid=[r for r in rows if r['reference_n']>0]
    tp=sum(r['intersection_n'] for r in valid);pred=sum(r['predicted_n'] for r in valid);ref=sum(r['reference_n'] for r in valid)
    return {'reference_annotated_n':len(valid),'intersection_total':tp,'predicted_total':pred,'reference_total':ref,
      'micro_precision_like_agreement':tp/pred if pred else None,'micro_reference_coverage':tp/ref if ref else None,
      'mean_jaccard':float(np.mean([r['jaccard'] for r in valid])) if valid else None}

def run(ds,root,out):
    root=Path(root);out=Path(out)
    source=root/'data/external_sources/attack_stix.json';attack=AttackReference(source)
    schema=json.loads((root/'config/mapping_schema.json').read_text());mapping=map_features(ds.names,schema)
    protocol={'evaluation':'External machine-generated annotations and curated family-profile concordance',
      'human_expert_review':False,'assessor':'automated deterministic comparison implemented with AI assistance',
      'schema_frozen':schema['version'],'schema_sha256':digest(root/'config/mapping_schema.json'),'reference_sha256':digest(source),
      'reference_url':'https://github.com/mitre-attack/attack-stix-data',
      'normalization':'official STIX revoked-by links only; unresolved deprecated IDs recorded and omitted; parent-level sensitivity comparison separate',
      'reference_limitations':'Cuckoo signatures are incomplete automated silver labels, partly shared with input features; family profiles describe known capabilities, not every execution. Agreement is not gold-standard accuracy or precision/recall of malicious intent.',
      'no_test_tuning':'mapper is not modified after inspecting concordance'}
    save(out/'mapping_external_protocol.json',protocol)
    reports=[];legacy={}
    for p in sorted((root/'data/raw_reports').glob('*.json')):
        d=json.loads(p.read_text(encoding='utf-8'));x=vector(d,ds.names)
        pred={mapping[n]['technique_id'] for n,a in zip(ds.names,x) if a and mapping[n]['technique_id']}
        refs=set()
        for sig in d.get('signatures',[]):
            raw=sig.get('ttp',{})
            for t in raw if isinstance(raw,(dict,list)) else []:
                tid=t if isinstance(t,str) else t.get('id','')
                if re.fullmatch(r'T\d{4}(?:\.\d{3})?',tid):
                    mapped=attack.resolve(tid);legacy[tid]=mapped
                    if mapped:refs.add(mapped)
        row={'sample_id':int(p.stem),'predicted':sorted(pred),'external_annotations':sorted(refs),**agreement(pred,refs)}
        row['parent_level']=agreement({t.split('.')[0] for t in pred},{t.split('.')[0] for t in refs})
        reports.append(row)
    # An independent published source for known family capabilities; absence is
    # not a negative label. Aggregate each family's predictions, not inflated n.
    frame=pd.concat([ds.train,ds.test],ignore_index=True).merge(ds.metadata[['sample_id','ransomware_family']],on='sample_id',validate='one_to_one')
    families=[]
    for family,g in frame[frame.sample_type==1].groupby('ransomware_family'):
        objects,refs=attack.profile(family)
        if not refs:continue
        x=(g[ds.columns].to_numpy()>0).any(0)
        pred={mapping[n]['technique_id'] for n,a in zip(ds.names,x) if a and mapping[n]['technique_id']}
        families.append({'family':family,'sample_n':len(g),'reference_software':[{'name':o['name'],'id':technique_id(o)} for o in objects],
          'predicted':sorted(pred),'external_annotations':sorted(refs),**agreement(pred,refs)})
    result={'human_expert_review':False,'cuckoo':{'all_report_n':len(reports),'summary':summarize(reports),'reports':reports,'legacy_resolution':legacy,
      'parent_summary':summarize([r['parent_level'] for r in reports])},'family_profiles':{'summary':summarize(families),'families':families}}
    save(out/'mapping_external.json',result)
    # Rule-level source check covers every rule, not an invented random assessor.
    ruleaudit=[]
    for r in schema['rules']:
        tid=r['technique_id'];o=attack.techniques.get(tid,{})
        ruleaudit.append({'rule':r.get('exact',r.get('prefix')),'technique_id':tid,'technique_current':bool(o and attack.resolve(tid)==tid),
          'source_title':o.get('name'),'source_url':next((x['url'] for x in o.get('external_references',[]) if x.get('source_name')=='mitre-attack' and 'url' in x),None),
          'claim':'candidate behavior only; benign use and missing context preclude malicious-technique confirmation','review_type':'AI-assisted source check; not independent human adjudication'})
    save(out/'mapping_rule_source_audit.json',ruleaudit)
    print('External annotation concordance',result['cuckoo']['summary'],'family profiles',len(families),flush=True)
