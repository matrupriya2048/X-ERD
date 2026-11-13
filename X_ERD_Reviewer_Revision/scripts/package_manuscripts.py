"""Keep eight main display items; preserve every other item in one supplement."""
import re,json
from pathlib import Path

MAIN_LABELS={'fig:architecture','tab:performance','tab:matched','tab:temporal',
             'tab:published','tab:external_dataset','tab:sequences','tab:raw_benign'}

def partition(text,root):
    root=Path(root);counts={'figure':0,'table':0};moved={};floats=[]
    pattern=r'\\begin\{(figure\*?|table\*?)\}.*?\\end\{\1\}'
    def choose(match):
        block=match.group(0);kind='figure' if match.group(1).startswith('figure') else 'table'
        labels=re.findall(r'\\label\{([^}]+)\}',block)
        if len(labels)!=1:raise ValueError('Every display requires one unique label')
        label=labels[0]
        if label in MAIN_LABELS:return block
        counts[kind]+=1;moved[label]=(kind,counts[kind]);floats.append((kind,label,block))
        return '\n'
    main=re.sub(pattern,choose,text,flags=re.S)
    # References to moved items remain explicit and readable without cross-PDF links.
    def refs(value):
        for label,(kind,number) in moved.items():
            ref=r'\ref{'+label+'}'
            value=value.replace(kind.title()+'~'+ref,'Supplementary '+kind.title()+'~S'+str(number))
            value=value.replace(ref,'S'+str(number))
        return value
    main=refs(main)
    retained=set(re.findall(r'\\label\{((?:fig|tab):[^}]+)\}',main))
    if retained!=MAIN_LABELS:raise ValueError('Main display inventory mismatch: '+str(retained^MAIN_LABELS))
    # Keep detailed newly added protocol paragraphs in Methods, where readers
    # can reproduce them without interrupting the main findings.
    starts=['All adapted detectors use the same','The original EldeRan collection has',
            'The independently published Iwan','Two-layer LSTM and CNN controls',
            'A fixed random sample(seed','The versioned mapper is frozen before',
            'A subsequent fixed search retains','The strong linear baseline motivates']
    methods=[]
    for start in starts:
        # Number-spacing happens before partitioning; allow whitespace at boundaries.
        prefix=start.replace('sample(seed','sample\\(seed') if '(' in start else re.escape(start)
        match=re.search(r'(?m)^'+prefix+r'.*?(?=\n\s*\n|\n\\begin\{table)',main,flags=re.S)
        if match:
            methods.append(match.group(0));main=main[:match.start()]+main[match.end():]
    if methods:
        main=main.replace(r'\subsection*{Reproducibility}',r'\subsection*{Extended experimental protocols}'+'\n\n'+'\n\n'.join(methods)+'\n\n'+r'\subsection*{Reproducibility}',1)
    # Cite each display in the narrative, including the items moved to the SI.
    callouts={
      'Benchmark dataset and retained architecture':r'Figure~\ref{fig:architecture} summarizes the retained pipeline and optional extension.',
      'Held-out classification and calibration':r'Table~\ref{tab:performance} gives the official-split results; Supplementary Table~S2 and Figure~S3 report calibration.',
      'Matched recall and threshold selection':r'Table~\ref{tab:matched} and Supplementary Figure~S4 compare matched operating points. Supplementary Table~S3 and Figure~S5 separate validation-selected policies from descriptive test curves.',
      'Repeated cross-validation and corrected inference':r'Supplementary Table~S4 and Figure~S6 give the corrected paired comparisons.',
      'Chronological evaluation and selection scope':r'Table~\ref{tab:temporal} reports the strict temporal holdout; Supplementary Table~S5 and Figure~S7 give its year-specific results.',
      'Corruption and targeted feature masking':r'Supplementary Table~S10 and Figure~S10 report common-mask corruption; Figure~S11 shows targeted deletion.',
      'Raw-report parser, attributions and time prefixes':r'Supplementary Tables~S11--S12 and Figure~S12 summarize reconstruction and downstream agreement; Table~S13 reports timestamped prefixes.',
      'Published-method comparisons on the official split':r'Table~\ref{tab:published} compares the executed published-method adaptations.',
      'External EldeRan collection and hash-disjoint transfer':r'Table~\ref{tab:external_dataset} separates within-collection refitting from frozen collection transfer.',
      'API-sequence comparison':r'Table~\ref{tab:sequences} compares ordered neural inputs and binary presence on identical folds.',
      'Larger external raw-report benign controls':r'Table~\ref{tab:raw_benign} reports false alerts on the accepted external benign executions.',
      'External ATT&CK reference concordance':r'Supplementary Table~S15 reports exact and parent-level reference concordance.',
      'Training-only optimization and duplicate-group sensitivity':r'Supplementary Table~S16 gives all configurations of the selected fixed-stack refit.',
      'Feature-passthrough extension':r'Supplementary Table~S17 gives official-split ablations; Table~S18 gives additional shared-API transfer comparisons.'}
    for head,prose in callouts.items():
        anchor=r'\subsection*{'+head.replace('&',r'\&')+'}'
        if main.count(anchor)!=1:raise ValueError('Missing or ambiguous display-callout section: '+head)
        main=main.replace(anchor,anchor+'\n'+prose+'\n',1)
    for anchor,prose in [
      ('Against the LGB ranking,','Supplementary Table~S7 and Figure~S9 summarize deletion sensitivity. '),
      ('PE-stub strings,','Supplementary Table~S8 reports the artifact ablations. ')]:
        if main.count(anchor)!=1:raise ValueError('Missing display-callout paragraph: '+anchor)
        main=main.replace(anchor,prose+anchor,1)
    main=main.replace(r'\section*{Data Availability}',r'\section*{Supplementary Information}'+'\n'+
      'A single Supplementary Information PDF contains all additional tables and figures, retaining the complete ablations, calibration, attribution, parser, mapping and optimization evidence. Machine-readable predictions, data manifests and executable protocols accompany the source release.\n\n'+r'\section*{Data Availability}',1)
    # Standalone supplement; source figures and all numerical tables are preserved.
    supplement=r'''\documentclass[10pt]{article}
\usepackage[a4paper,margin=20mm]{geometry}
\usepackage[T1]{fontenc}\usepackage[utf8]{inputenc}
\usepackage{graphicx,booktabs,amsmath,amssymb,url,hyperref,float}
\hypersetup{hidelinks}
\graphicspath{{../results/figures/}{./}}
\newsavebox{\xerdtablebox}
\renewcommand{\thetable}{S\arabic{table}}
\renewcommand{\thefigure}{S\arabic{figure}}
\renewcommand{\refname}{Supplementary references}
\title{Supplementary Information\\Explainable ransomware detection with stacked ensembles and rule-based MITRE ATT\&CK suggestions}
\author{Matrupriya Dibyanshu Panda, Ravi Shankar and Sri Aurobindo Panda}
\date{}
\begin{document}\maketitle
This supplement preserves additional evidence for the revised X-ERD study. Tables and figures refer to the original configuration unless explicitly labeled as training-selected or external experiments. The main manuscript defines sampling limitations and distinguishes automated reference concordance from independent human validation. Code and per-record exports provide the corresponding numerical detail.
'''
    order={'figure':0,'table':0}
    inventory=[]
    for kind,label,block in floats:
        order[kind]+=1;number='S'+str(order[kind]);inventory.append({'label':label,'kind':kind,'number':number})
        block=block.replace('figure*','figure').replace('table*','table').replace('[t]','[H]')
        # Nonfloating blocks flow naturally and cannot split individual tables.
        supplement+='\n'+refs(block)+'\n'
    supplement+='\n'+r'\bibliographystyle{naturemag}\bibliography{references_revision}\end{document}'+'\n'
    # The few supplement captions referring to main figures use numbered labels.
    main_order={'figure':0,'table':0};main_numbers={}
    for match in re.finditer(pattern,main,flags=re.S):
        kind='figure' if match.group(1).startswith('figure') else 'table';main_order[kind]+=1
        main_numbers[re.search(r'\\label\{([^}]+)\}',match.group(0)).group(1)]=str(main_order[kind])
    for label,number in main_numbers.items():supplement=supplement.replace(r'\ref{'+label+'}',number)
    (root/'manuscript/supplementary_information.tex').write_text(supplement,encoding='utf-8')
    (root/'docs/display_inventory.json').write_text(json.dumps({'main':main_numbers,'supplement':inventory,'journal_guidance':'https://www.nature.com/srep/author-instructions/submission-guidelines','main_display_count':len(main_numbers)},indent=2))
    return main
