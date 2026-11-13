"""Acquire inert published JSON reports, preserving a pre-scoring cohort freeze."""
from pathlib import Path
import concurrent.futures, datetime, hashlib, json, random, urllib.request

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent / 'X_ERD_Reviewer_Revision'
COMMIT = 'c449c4e44ef9d2178fbb40beef7ef12741a894a9'

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            h.update(block)
    return h.hexdigest()

def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    temp.replace(path)

def freeze():
    path = HERE / 'protocol.json'
    if path.exists():
        return json.loads(path.read_text())
    tree_path = ROOT / 'data/external_sources/mtm_tree.json'
    tree = json.loads(tree_path.read_text())
    assert tree['sha'] == COMMIT and not tree['truncated']
    pilot = ['2284694_0.json', '2284695_1.json', '2284696_0.json']
    pool = sorted([x for x in tree['tree'] if x['path'].endswith('_0.json') and x['path'] not in pilot], key=lambda x:x['path'])
    files = random.Random(2026111302).sample(pool, 200)
    models = {'original':'results/model.joblib', 'previous':'results/extended/residual_model.joblib', 'adapted':'results/remediation/model.joblib'}
    p = {'frozen_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
         'scope':'Additional collection stress test using publisher score-derived benign labels; not independent human-adjudicated clean software.',
         'source':'https://github.com/mtm41/dataset', 'commit':COMMIT,
         'paper':'https://rua.ua.es/dspace/bitstream/10045/137642/3/Torres_etal_2023_IEEEAccess.pdf',
         'label_rule':'Filename suffix _0 plus numeric Cuckoo score in [0,3], following publisher score-based benign definition. No inference that low score proves clean software.',
         'eligibility':'Windows file execution, target executable hash, behavior object and API observations. Exclude invalid or mismatched labels/platforms; no replacements.',
         'independence':'Exclude executable hashes in original metadata, adaptation development, previous raw cohorts and earlier external reports; deduplicate all available target hash types. Audit vector matches separately.',
         'threshold':0.5, 'fit_or_selection_on_this_collection':False,
         'seed':2026111302, 'pool_n':len(pool), 'selected_n':len(files),
         'pilot_excluded':pilot, 'models':{k:{'path':v,'sha256':sha(ROOT/v)} for k,v in models.items()},
         'parser_sha256':sha(ROOT/'src/xerd/parser.py'), 'tree_sha256':sha(tree_path),
         'files':files, 'annotation_claim':'No human adjudication or verified encryption-onset labels are supplied by this evaluation.',
         'redistribution':'Repository license not declared; raw reports are local research inputs and excluded from deliverable archive.'}
    save(path,p)
    return p

def download(item):
    target = HERE / 'raw' / item['path']
    if target.exists() and target.stat().st_size == item['size']:
        data=target.read_bytes()
        if hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()==item['sha']:
            return {'path':item['path'],'status':'acquired','bytes':len(data),'sha256':sha(target),'git_blob_sha1_verified':True}
    url=f"https://raw.githubusercontent.com/mtm41/dataset/{COMMIT}/{item['path']}"
    last=''
    for attempt in range(3):
        try:
            req=urllib.request.Request(url,headers={'User-Agent':'XERD-research-evidence-audit'})
            with urllib.request.urlopen(req,timeout=90) as response:
                data=response.read()
            assert len(data)==item['size'], 'Published length mismatch'
            assert hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()==item['sha'], 'Published Git blob mismatch'
            target.parent.mkdir(parents=True,exist_ok=True)
            target.write_bytes(data)
            return {'path':item['path'],'status':'acquired','url':url,'bytes':len(data),'sha256':sha(target),'git_blob_sha1_verified':True}
        except Exception as error:
            last=f'{type(error).__name__}: {error}'
    return {'path':item['path'],'status':'unavailable','url':url,'reason':last}

if __name__=='__main__':
    protocol=freeze()
    print('Frozen selection:',len(protocol['files']),'reports,',sum(x['size'] for x in protocol['files']),'bytes',flush=True)
    rows=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures=[executor.submit(download,x) for x in protocol['files']]
        for future in concurrent.futures.as_completed(futures):
            rows.append(future.result())
            if len(rows)%20==0:
                save(HERE/'acquisition.json',sorted(rows,key=lambda x:x['path']))
                print('Acquired/audited',len(rows),'of',len(futures),flush=True)
    save(HERE/'acquisition.json',sorted(rows,key=lambda x:x['path']))
