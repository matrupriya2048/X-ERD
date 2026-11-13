"""Explicit CAPE enhanced-event translation; no class labels or scores consulted."""
from collections import defaultdict

OPERATIONS={('read','registry'):('regkey_read','regkey'),('write','registry'):('regkey_written','regkey'),
            ('delete','registry'):('regkey_deleted','regkey'),('read','file'):('file_read','file'),
            ('write','file'):('file_written','file'),('create','file'):('file_created','file'),
            ('delete','file'):('file_deleted','file'),('load','library'):('dll_loaded','file'),
            ('create','directory'):('directory_created','file')}

def canonical_path(value):
    # CAPE enhanced paths use one extra escaping layer. Decode separator pairs
    # once; preserve UNC prefixes, case and all non-separator characters.
    return value.replace('\\\\','\\')

def adapt(report):
    if not isinstance(report,dict):raise ValueError('Report object required')
    if 'CAPE' not in str(report.get('info',{}).get('version','')).upper():
        return report,{'adapter_applied':False,'reason':'Not declared CAPE input'}
    b=report.get('behavior') or {};summary={k:list(v) if isinstance(v,list) else v for k,v in (b.get('summary') or {}).items()}
    mapped=defaultdict(list);unknown=defaultdict(int);count=0
    for event in b.get('enhanced') or []:
        rule=OPERATIONS.get((event.get('event'),event.get('object')))
        if rule is None:unknown[str((event.get('event'),event.get('object')))]+=1;continue
        dest,field=rule;value=(event.get('data') or {}).get(field)
        if isinstance(value,str):mapped[dest].append(canonical_path(value));count+=1
    for key,values in mapped.items():summary[key]=list(dict.fromkeys([*(summary.get(key) or []),*values]))
    result={**report,'behavior':{**b,'summary':summary}}
    nested=report.get('target',{}).get('file',{}).get('strings')
    if 'strings' not in report and isinstance(nested,list):result['strings']=nested
    return result,{'adapter_applied':True,'mapped_event_n':count,'unsupported_event_counts':dict(unknown),
                   'strings_source':'root' if 'strings' in report else 'target.file.strings' if isinstance(nested,list) else 'unavailable',
                   'mapped_summary_fields':sorted(mapped),'scope':'Observed typed events only. Aggregate files/keys are not relabeled as operation-specific evidence.'}
