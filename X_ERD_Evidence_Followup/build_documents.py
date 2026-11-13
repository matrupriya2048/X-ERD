"""Minimal, reproducible manuscript and response edits over verified v0.3 text."""
from pathlib import Path
import argparse, hashlib, html, json, re, shutil, unicodedata
from pypdf import PdfReader

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent/'X_ERD_Reviewer_Revision'
MAN=HERE/'manuscript'
DOC=HERE/'docs'

def esc(s):
    replacements={'\\':r'\textbackslash{}','&':r'\&','%':r'\%','_':r'\_','#':r'\#','{':r'\{','}':r'\}','$':r'\$','>':r'\textgreater{}','<':r'\textless{}','^':r'\textasciicircum{}','~':r'\textasciitilde{}'}
    return ''.join(replacements.get(c,c) for c in str(s))

def pieces():
    r=json.loads((HERE/'results.json').read_text());a=json.loads((HERE/'annotation_readiness.json').read_text());v=json.loads((HERE/'verification.json').read_text())
    assert v['status']=='passed'
    m=r['models'];p=m['adapted'];n=p['n'];lo,hi=p['binomial_ci95']
    cohort=(f'We added a frozen-model assessment on the separately collected Torres et al. Cuckoo reports. Before scoring, 200 paths were sampled uniformly from 17,935 uninspected files with the publisher benign-label suffix. All 200 files passed published Git-blob and length checks. The prespecified Windows file-execution and hash checks excluded 42 nonmatching executions and 16 duplicate submitted files, leaving {n} unique eligible targets. No model, feature, threshold or calibration was selected on this collection. At threshold 0.50, the adapted stack produced {p["alert_n"]}/{n} alerts ({100*p["label_relative_fpr"]:.2f}%; exact binomial 95% interval {100*lo:.2f}-{100*hi:.2f}%), while the original and previous stacks each produced 5/{n} (3.52%). Thus adaptation did not improve the point estimate on this collection.')
    labels=('The publisher benign designation is based on Cuckoo scores from 0 to 3, not independently verified clean-software labels. These are label-relative alert rates, not a deployment FPR estimate. Post-hoc target-type auditing identifies 86 HTML, 27 PDF, 11 PE executable and 18 other files. Each model alerts on 2/11 PE executables (18.18%); none alerts on the HTML/PDF targets. Thus the aggregate denominator does not establish benign-application performance. Only 25/142 reports contain numeric antivirus-positive counts, all zero; 117 lack those counts. One target vector matches a fitting vector; excluding it descriptively gives 6/141 alerts (4.26%) for the adapted model. The primary cohort is unchanged. Available target-file hashes do not overlap the original metadata or prior adaptation/raw cohorts, but all 80 earlier sequence-only inputs lack executable hashes, preventing complete disjointness verification against those inputs. No pooled accuracy is computed across collections.')
    annotation=(f'An additional annotation-readiness audit inspected the 20 benchmark raw reports and 80 external sequence-only JSON inputs. All {a["report_n"]} contain numeric call timestamps; {a["reports_with_timestamped_crypto_hints"]} contain a timestamped CryptEncrypt, BCryptEncrypt or NCryptEncrypt call. These are API hints, not verified destructive encryption onset. The supplied evidence contains no matched first-encrypted-write annotations. We therefore report neither pre-encryption detection rates nor prevention lead times. Twenty evidence-linked ATT&CK audit packets and an unfilled assessor worksheet are supplied, but no independent human adjudication has occurred. Automated source checks remain explicitly distinct from human validation.')
    return cohort,labels,annotation

def papers():
    MAN.mkdir(exist_ok=True);DOC.mkdir(exist_ok=True)
    for name in ['wlscirep.cls','naturemag.bst','references_revision.bib','fig_class_category_real.png']:
        shutil.copy2(ROOT/'manuscript'/name,MAN/name)
    figs=MAN/'figures';figs.mkdir(exist_ok=True)
    for f in (ROOT/'results/figures').glob('*.png'):shutil.copy2(f,figs/f.name)
    bib=MAN/'references_revision.bib'
    with bib.open('a',encoding='utf-8') as stream:
        stream.write(r'''
@article{torres2023behavior,
 author={Torres, Manuel and Alvarez, Rafael and Cazorla, Miguel},
 title={A Malware Detection Approach Based on Feature Engineering and Behavior Analysis},
 journal={IEEE Access}, volume={11}, pages={105355--105367}, year={2023},
 doi={10.1109/ACCESS.2023.3319093}
}
''')
    cohort,labels,annotation=pieces()
    source=(ROOT/'manuscript/main_scirep_revised.tex').read_text(encoding='utf-8')
    text=source.replace(r'\graphicspath{{../results/figures/}{./}}',r'\graphicspath{{./figures/}{./}}')
    addition=r'\subsection*{Additional collection with score-derived benign labels}'+'\n'
    short=(r'An exploratory frozen-model assessment used the separately collected Torres et al. reports\cite{torres2023behavior}. Of 200 preselected paths, 142 unique submitted files passed the declared checks. The adapted stack produced 6/142 alerts (4.23\%; exact binomial 95\% interval 1.57--8.97\%), versus 5/142 (3.52\%) for each earlier stack. However, 113 targets are HTML/PDF files and only 11 are PE executables; all three models alert on 2/11 executables (18.18\%). Publisher benign labels derive from sandbox scores, and 117/142 reports lack numeric antivirus-positive counts. This mixed-file collection therefore adds a transport stress test, not independently verified benign-application validation or evidence of improved generalization. No model was retuned. The complete sampling, exclusions, file-type breakdown and uncertainty are reported in Supplementary Table~S22 and its accompanying text.'+'\n\n')
    addition+=short
    assert text.count(r'\section*{Discussion}')==1
    text=text.replace(r'\section*{Discussion}',addition+r'\section*{Discussion}',1)
    methods=r'\subsection*{Additional collection and annotation audit}'+'\n'
    methods+=('The additional local protocol was frozen before target acquisition and scoring (seed 2026111302). The repository snapshot is identified in the evidence manifest; all selected paths and exclusions are retained. Eligibility required score 0--3, Windows file execution, a valid target hash, and observed API events or counts. Missing or incompatible telemetry was not scored as a negative. All available MD5, SHA-1 and SHA-256 identifiers were used for overlap checks and duplicate exclusion; exact vector matches were retained in the primary cohort and audited separately. Models, the 483-feature parser and threshold 0.50 remained fixed. Binomial intervals describe uncertainty conditional on the score-derived labels and may understate dependence among related applications.\n\n')
    methods+=esc(annotation)+'\n\n'
    assert text.count(r'\subsection*{Reproducibility}')==1
    text=text.replace(r'\subsection*{Reproducibility}',methods+r'\subsection*{Reproducibility}',1)
    (MAN/'main_scirep_revised.tex').write_text(text,encoding='utf-8')
    sup=(ROOT/'manuscript/supplementary_information.tex').read_text(encoding='utf-8').replace(r'\graphicspath{{../results/figures/}{./}}',r'\graphicspath{{./figures/}{./}}')
    sup=sup.replace('url,hyperref,float','url,float,hyperref')
    assert len(re.findall(r'\\begin\{table\*?\}',sup))==21
    table=r'''
\clearpage
\section*{Additional frozen-model collection assessment}
\begin{table}[H]\centering\small
\caption{Untuned full-stack assessment on the separately collected Torres et al. score-labeled reports. These are label-relative alerts, not independently established clean-software false positives.}
\label{tab:additional_collection}
\begin{tabular}{lrrrr}\toprule
Configuration & Alerts/N & Rate (\%) & 95\% interval & Novel vectors\\\midrule
Original full stack & 5/142 & 3.52 & 1.15--8.03 & 5/141\\
Previous passthrough & 5/142 & 3.52 & 1.15--8.03 & 5/141\\
Frozen adapted stack & 6/142 & 4.23 & 1.57--8.97 & 6/141\\\bottomrule
\end{tabular}
\end{table}
'''
    table+=esc(cohort)+'\n\n'+esc(labels)+'\n\n'+esc(annotation)+'\n\n'
    # Add after existing references to preserve all existing display numbering.
    sup=sup.replace(r'\end{document}',table+r'\end{document}',1)
    (MAN/'supplementary_information.tex').write_text(sup,encoding='utf-8')
    changes={'base':'Verified v0.3 manuscript and supplement','core_model_changed':False,
             'edits':['Abstract and core claims preserved','Results adds a short exploratory subsection','Methods adds cohort and annotation audit','Supplement adds Table S22 and full limitations; prior displays retained'],
             'base_main_sha256':hashlib.sha256((ROOT/'manuscript/main_scirep_revised.pdf').read_bytes()).hexdigest()}
    (DOC/'change_record.json').write_text(json.dumps(changes,indent=2))

def response():
    cohort,labels,annotation=pieces()
    pages=[p.extract_text() or '' for p in PdfReader(MAN/'main_scirep_revised.pdf').pages]
    normalize=lambda s:re.sub('[^a-z0-9]','',unicodedata.normalize('NFKD',s).lower())
    def locate(head):
        hits=[];needle=normalize(head)
        for i,page in enumerate(pages):
            lines=page.splitlines()
            if any(normalize(''.join(lines[j:j+n]))==needle for j in range(len(lines)) for n in [1,2,3]):hits.append(i+1)
        if not hits:raise ValueError('Missing final heading '+head)
        return head+' (p. '+', '.join(map(str,hits))+')'
    rows=json.loads((ROOT/'docs/reviewer_revision_status.json').read_text())['items']
    index=json.loads((ROOT/'docs/response_letter_index.json').read_text())['items']
    base=(ROOT/'docs/response_letter.tex').read_text(encoding='utf-8')
    raw_ids={'R1.3','R1.m2','R1.g3','R2.12','R4.1','R1.6','R4.3','R2.13','R3.1'}
    annotation_ids={'R1.4','R1.g4','R2.10','R3.4','R1.16','R4.5'}
    by_id={x['id']:x for x in rows}
    for item in index:
        heads=[re.sub(r' \(p\. [0-9, ]+\)$','',x) for x in item['locations']]
        extra=[]
        if item['id'] in raw_ids:extra.append('Additional collection with score-derived benign labels')
        if item['id'] in annotation_ids:extra.append('Additional collection and annotation audit')
        newloc=[locate(h) for h in heads+extra]
        oldline=r'\textit{Location:} '+esc('; '.join(item['locations']))+'.'
        newline=r'\textit{Location:} '+esc('; '.join(newloc))+'.'
        addition=''
        if item['id'] in raw_ids:
            addition=(cohort+' '+labels) if item['id'] in {'R1.3','R4.1'} else ('A separate frozen-model collection test gives 6/142 alerts for the adapted model and 5/142 for each earlier model. Only 11 targets are PE executables, with 2/11 alerts for every model; most targets are HTML/PDF. Score-derived labels do not establish verified-clean benign applications or improved transport. Response R1.3 and the additional-collection Results give the full sampling, exclusions and provenance limits.')
        if item['id'] in annotation_ids:addition+=((' ' if addition else '')+annotation)
        if addition:
            newline=r'\textbf{Additional evidence.} '+esc(addition)+'\n\n'+newline
            by_id[item['id']]['response']+=' '+addition
            by_id[item['id']]['followup_evidence']=['results.json','verification.json','annotation_readiness.json']
        # Several comments share a location line. Replace within the specific subsection only.
        start=base.index(r'\subsection*{'+esc(item['id'])+' ')
        end=base.find(r'\subsection*{',start+1)
        if end<0:end=len(base)
        block=base[start:end]
        assert oldline in block,item['id']
        base=base[:start]+block.replace(oldline,newline,1)+base[end:]
        item['locations']=newloc
    intro=('A further collection assessment, frozen before acquisition, is now included: 6/142 score-labeled benign targets trigger the adapted model, compared with 5/142 for each earlier model. This does not establish a cross-collection advantage. The report-level evidence packets facilitate later independent review but do not constitute completed human adjudication. In the evidence paths below, pre-existing artifacts refer to the accompanying v0.3 codebase; the additional experiment and annotation audit are in X_ERD_Evidence_Followup.\n\n')
    base=base.replace(r'\section*{Editor}',esc(intro)+r'\section*{Editor}',1)
    (DOC/'response_letter.tex').write_text(base,encoding='utf-8')
    (DOC/'response_letter_index.json').write_text(json.dumps({'covered_items':len(index),'items':index},indent=2))
    (DOC/'reviewer_revision_status.json').write_text(json.dumps({'items':rows,'new_annotation_claims':'No human or verified onset completion asserted'},indent=2))
    body='<h1>Additional evidence and remaining submission risks</h1>'
    for heading,p in [('Independent collection test',cohort),('Label and independence limits',labels),('Annotation audit',annotation)]:body+='<h2>'+heading+'</h2><p>'+html.escape(p)+'</p>'
    body+='<h2>Decision</h2><p>The new experiment reduces the gap in collection diversity. It does not close independent human adjudication, verified onset, or independently verified benign ground truth. The revised manuscript retains these limits. Reviewer acceptance cannot be guaranteed.</p>'
    body+='<h2>Reviewer coverage</h2><p>All 69 tracked comments remain in the response letter. The affected responses contain the new measurements and updated manuscript locations.</p>'
    (HERE/'EVIDENCE_FOLLOWUP_REPORT.html').write_text('<!doctype html><html lang="en"><meta charset="utf-8"><title>X-ERD additional evidence</title><style>body{font:17px/1.6 system-ui;max-width:950px;margin:45px auto;padding:0 24px;color:#172337}h1,h2{line-height:1.25}h2{margin-top:32px}</style>'+body+'</html>',encoding='utf-8')

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['papers','response']);args=parser.parse_args()
    papers() if args.stage=='papers' else response()
