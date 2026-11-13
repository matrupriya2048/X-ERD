"""Record every listed report and explicit publisher/provider access exclusions."""
from pathlib import Path
import json

ROOT=Path(__file__).resolve().parents[1]

def main():
    inv=json.loads((ROOT/'config/zirari_raw_inventory.json').read_text());attempts={r['name']:r for r in json.loads((ROOT/'data/zirari_cli_retry.json').read_text())};rows=[]
    for item in inv['files']:
        receipt=ROOT/'data/zirari_raw_receipts'/(item['title']+'.receipt.json')
        row={'name':item['title'],'drive_id':item['id'],'bytes':int(item['size'])}
        if receipt.exists():row.update(status='acquired',**json.loads(receipt.read_text()))
        else:
            error=attempts.get(item['title'],{}).get('diagnostic','')
            if 'Only the owner is allowed to download infected files' not in error:raise RuntimeError('Unresolved acquisition failure: '+item['title'])
            row.update(status='unavailable_provider_restriction',reason='Google Drive reports that only the owner may download this flagged file; no bypass attempted. Publisher JSON label is retained; this is an access exclusion, not a model-based exclusion.')
        rows.append(row)
    (ROOT/'config/zirari_acquisition_status.json').write_text(json.dumps({'listed':len(rows),'acquired':sum(r['status']=='acquired' for r in rows),'files':rows,'scope':'All currently listed reports accounted for before ransomware evaluation; unavailable rows do not enter the sensitivity denominator.'},indent=2))
    print('Acquisition accounted for',len(rows),'listed reports;',sum(r['status']=='acquired' for r in rows),'acquired')

if __name__=='__main__':main()
