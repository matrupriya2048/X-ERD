"""Compare available report metadata with the publication's stated curation."""
from pathlib import Path
import json,collections
from xerd.io import save

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/remediation'

def main():
    rows=json.loads((OUT/'ransomware_audit.json').read_text());scored=[r for r in rows if isinstance(r.get('publisher_score'),(float,int))];accepted=[r for r in scored if r['status']=='evaluated']
    hashes=[r['md5'] for r in rows if r.get('md5')]
    result={'publication':'https://doi.org/10.3991/ijim.v19i10.49245','publication_description':'140 selected ransomware samples; score at least 5; duplicates and nonfunctional samples removed (pp.206-207). This description is not substituted for inspection of the currently downloadable artifacts.',
      'currently_listed_report_files':138,'acquired_report_files':135,'files_with_audited_target_md5':len(hashes),'distinct_audited_target_md5':len(set(hashes)),
      'reports_with_numeric_score':len(scored),'reports_with_numeric_score_below5':sum(r['publisher_score']<5 for r in scored),'accepted_unique_reports_with_numeric_score':len(accepted),'accepted_unique_reports_with_numeric_score_below5':sum(r['publisher_score']<5 for r in accepted),'exclusion_counts':dict(collections.Counter(r['status'] for r in rows)),
      'interpretation':'Current report inventory does not fully substantiate the publication curation description. Report files and selected samples are different units; reasons for the discrepancy are not established. Publisher ransomware labels are retained without fresh adjudication; no local Cuckoo-score filter, relabeling or model change is applied. Sensitivity is conditional on available publisher-labeled inputs and documented exclusions.'}
    save(OUT/'publisher_consistency_audit.json',result)
    metrics=json.loads((OUT/'ransomware.json').read_text());metrics.pop('paper_describes_reports',None);metrics['paper_describes_selected_samples']=140;metrics['provenance_audit']='results/remediation/publisher_consistency_audit.json';save(OUT/'ransomware.json',metrics)
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
