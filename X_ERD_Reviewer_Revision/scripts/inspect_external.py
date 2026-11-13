from pathlib import Path
import json,collections,zipfile
R=Path(__file__).resolve().parents[1]/'data'/'external_sources'
for name in ['sequence_listing','sequence_benign_listing']:
 rows=json.loads((R/(name+'.json')).read_text())
 print(name,len(rows),sum(x.get('content_details',{}).get('size',0) for x in rows))
 for x in rows[:5]:print(json.dumps(x,ensure_ascii=True)[:1300])
tree=json.loads((R/'mtm_tree.json').read_text())
print('MTM commit',tree['sha'],'truncated',tree.get('truncated'))
print(collections.Counter(x['path'].split('/')[0] for x in tree['tree']))
for x in tree['tree'][:18]:print(x['path'],x.get('size'))
print('MTM total bytes',sum(x.get('size',0) for x in tree['tree']))
