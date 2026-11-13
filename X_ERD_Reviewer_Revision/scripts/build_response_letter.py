"""Typeset all 69 responses with locations verified against the final manuscript."""
from pathlib import Path
import json,re,unicodedata
from pypdf import PdfReader
from build_revision_report import clean_prose

ROOT=Path(__file__).resolve().parents[1];DOC=ROOT/'docs';MAN=ROOT/'manuscript'

def esc(s):
    s=str(s).replace('\u2013','--').replace('\u2014','--').replace('\u2019',"'")
    out=''
    for ch in s:
        out += {'\\':r'\textbackslash{}','&':r'\&','%':r'\%','_':r'\_','#':r'\#',
                '{':r'\{','}':r'\}','$':r'\$','>':r'\textgreater{}','<':r'\textless{}',
                '^':r'\textasciicircum{}','~':r'\textasciitilde{}'}.get(ch,ch)
    return out.replace(r'1\textasciicircum{}T Sigma 1/K\textasciicircum{}2',r'$\mathbf{1}^{\top}\Sigma\mathbf{1}/K^2$')

def main():
    rows=json.loads((DOC/'reviewer_revision_status.json').read_text())['items']
    if len(rows)!=69 or len({r['id'] for r in rows})!=69:raise AssertionError('Reviewer coverage')
    for p in [ROOT/'results/release_verification.json',ROOT/'results/extended/verification.json']:
        if json.loads(p.read_text())['status']!='passed':raise AssertionError('Unverified results')
    def normalize(s):return re.sub(r'[^a-z0-9]','',unicodedata.normalize('NFKD',s).lower())
    pages=[p.extract_text() for p in PdfReader(MAN/'main_scirep_revised.pdf').pages]
    def locate(head):
        needle=normalize(head)
        # Match a complete heading over at most three lines, not incidental
        # mentions of the heading phrase in abstracts or figure captions.
        hits=[]
        for i,page in enumerate(pages):
            lines=page.splitlines()
            if any(normalize(''.join(lines[j:j+n]))==needle for j in range(len(lines)) for n in (1,2,3)):
                hits.append(i+1)
        if not hits:raise ValueError('Section not located in final PDF: '+head)
        return f'{head} (p. '+', '.join(map(str,hits))+')'
    locations={}
    def link(ids,*heads):
        for i in ids.split():locations[i]=list(heads)
    link('E1 R1.g1 R1.g2 R1.g5 R2.4 R3.2','Discussion','Held-out classification and calibration')
    link('E2','Discussion','Supplementary Information')
    link('E3 R4.9 R1.m5 R2.m7 R2.m8 R2.m9 R4.8','Related Work','Positioning and comparison scope')
    link('E4 R1.5 R2.14','Code Availability','Reproducibility')
    link('R1.1','Published-method comparisons on the official split','Feature-passthrough extension')
    link('R1.2 R2.5','Held-out classification and calibration','Feature-passthrough extension')
    link('R1.3 R1.m2 R1.g3 R2.12 R4.1','Raw-report parser, attributions and time prefixes','Larger external raw-report benign controls')
    link('R1.4 R1.g4 R2.10 R3.4','ATT&CK suggestions and evaluation boundary','External ATT&CK reference concordance')
    link('R1.6 R4.3','Chronological evaluation and selection scope','External EldeRan collection and hash-disjoint transfer')
    link('R1.7 R2.1 R2.7 R3.3 R1.m1 R2.m5','Matched recall and threshold selection','Evaluation and inference')
    link('R1.8 R1.13','Held-out classification and calibration')
    link('R1.9 R1.m7 R2.2 R2.3 R3.5','Repeated cross-validation and corrected inference','Training-only optimization and duplicate-group sensitivity')
    link('R1.10','Benchmark dataset and retained architecture','Parser and temporal isolation')
    link('R1.11 R1.12','Data schema and model fitting','Training-only optimization and duplicate-group sensitivity')
    link('R1.14 R1.m4 R2.11 R4.2','Corruption and targeted feature masking')
    link('R1.15 R2.6 R2.m6 R4.7','Correlated probabilities and variance identity')
    link('R1.16 R4.5','Raw-report parser, attributions and time prefixes','API-sequence comparison')
    link('R1.m3 R1.m6 R2.m1 R2.m2','Benchmark dataset and retained architecture')
    link('R2.8 R2.m3','Probability-consistent attribution and deletion sensitivity','Attribution and mapping')
    link('R2.9','Corruption and targeted feature masking','Discussion')
    link('R2.13 R3.1','Chronological evaluation and selection scope','Published-method comparisons on the official split','External EldeRan collection and hash-disjoint transfer')
    link('R2.m4','Corruption and targeted feature masking')
    link('R4.4','ATT&CK suggestions and evaluation boundary','Attribution and mapping')
    link('R4.6','API-sequence comparison','Extended experimental protocols')
    for item in ['R1.3','R1.m2','R1.g3','R2.12','R4.1','R1.6','R4.3','R2.13','R3.1','R3.4']:
        locations[item].append('Benign-domain adaptation and frozen raw holdouts')
    for item in ['R1.3','R4.1']:
        locations[item].append('Disclosed domain-adaptation protocol')
    if set(locations)!={r['id'] for r in rows}:raise AssertionError('Missing locations: '+str({r['id'] for r in rows}-set(locations)))
    text=r'''\documentclass[11pt,a4paper]{article}
\usepackage[margin=23mm]{geometry}
\usepackage[T1]{fontenc}\usepackage[utf8]{inputenc}
\usepackage{lmodern,microtype,url,hyperref,amsmath,xurl}
\hypersetup{hidelinks}\urlstyle{same}
\setlength{\parindent}{0pt}\setlength{\parskip}{6pt}
\emergencystretch=2em
\begin{document}
\begin{center}{\Large\bfseries Response to the Editor and Reviewers}\par
\vspace{5pt}Scientific Reports\par
Explainable ransomware detection with stacked ensembles and rule-based MITRE ATT\&CK suggestions
\end{center}
Dear Editor and Reviewers,

We thank you for the detailed assessment. The revision retains the RF/XGBoost/LightGBM stacking framework, official MLRan benchmark and the parser, attribution and behavioral-mapping components. We have corrected the statistical and explanatory formulations, expanded the experimental evidence, and restricted conclusions to the measured settings. A bounded feature-passthrough extension is reported separately from the original configuration.

The responses below cover all 69 tracked editor and reviewer items. Numbered requests retain their original order; minor comments use ``m'' and Reviewer 1's wording guidance uses ``g''. Comment headings summarize the requests. Each response identifies the corresponding sections in the revised manuscript and the supporting local release files. Page numbers refer to the accompanying revision, including its title page.

The additional evaluations include published-method adaptations, strict temporal feature selection, executable-group sensitivity, real API-sequence models, an external collection with duplicate/overlap checks, and external benign raw-report controls. Independent human ATT\&CK adjudication and encryption-onset annotations remain unavailable; the associated accuracy and prevention claims are restricted accordingly. Public software archiving is the remaining author-controlled release step, and no completed DOI deposit is asserted in this version.

The expanded raw-report evaluation exposes a substantial transport limitation: on 182 usable, source-disjoint publisher-clean executions, the original full-stack FPR is 52.20\% and the passthrough FPR is 50.55\%. These outcomes are included in the manuscript and preclude an end-to-end deployment-readiness claim. Improvements on the official split and the separate shared-API transfer experiment are reported within their respective protocols.
'''
    previous=None;index=[]
    h=json.loads((ROOT/'results/remediation/holdout.json').read_text());r=json.loads((ROOT/'results/remediation/ransomware.json').read_text())
    new=h['models']['Development-adapted']['X-ERD'];old=h['models']['Previous passthrough']['X-ERD'];q=r['models']['Development-adapted']['X-ERD']
    text+=esc(clean_prose(f'We additionally used those182 reports as explicitly disclosed development data and froze a constrained adaptation before scoring400 new selected benign report paths. On{h["evaluated_unique_n"]} eligible unique executions,the adapted stack gives{new["fp"]} false alarms({100*new["fpr"]:.2f}% FPR),compared with{old["fp"]}({100*old["fpr"]:.2f}%) for the previous passthrough model on the same targets. Separate publisher-labeled ransomware reports with documented curation discrepancies give{q["tp"]}/{r["evaluated_unique_n"]} detections. The previous passthrough detects56/58,so the adapted full stack has one additional miss;recall non-inferiority is not established. These are separate specificity and sensitivity assessments;they are not pooled into an overall accuracy estimate. The original adverse results remain disclosed.'))+'\n\n'
    for row in rows:
        group=row['reviewer']
        if group!=previous:text+='\n'+r'\section*{'+esc(group)+'}\n';previous=group
        text+=r'\subsection*{'+esc(row['id'])+' '+esc(clean_prose(row['comment_topic']))+'}\n'
        text+=r'\textbf{Response.} '+esc(row['response'])+'\n\n'
        located=[locate(h) for h in locations[row['id']]]
        text+=r'\textit{Location:} '+esc('; '.join(located))+'.\n\n'
        text+=r'{\small\raggedright\textit{Evidence:} '+ '; '.join(r'\path{'+p+'}' for p in row['evidence'])+r'\par}'+'\n\n'
        for p in row['evidence']:
            if not (ROOT/p).exists():raise FileNotFoundError('Evidence missing: '+p)
        index.append({'id':row['id'],'status':row['status'],'locations':located,'evidence':row['evidence']})
    text+='\n'+r'\end{document}'+'\n'
    (DOC/'response_letter.tex').write_text(text,encoding='utf-8')
    (DOC/'response_letter_index.json').write_text(json.dumps({'covered_items':69,'items':index},indent=2))
    print('Prepared69 response items with verified manuscript locations')

if __name__=='__main__':main()
