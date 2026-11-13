"""Actual four-reviewer response and author-facing report; no false completions."""
from pathlib import Path
import json,html,collections,re
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results';DOC=ROOT/'docs'
def read(name):return json.loads((OUT/(name+'.json')).read_text(encoding='utf-8'))

def clean_prose(text):
    protected=[]
    def hold(match):
        value=match.group(0)
        before=' ' if re.fullmatch(r'\d+(?:\.\d+)?e[-+]\d+',value) and match.start() and text[match.start()-1].isalpha() else ''
        protected.append(value);return before+chr(0xE000+len(protected)-1)
    text=re.sub(r'https?://\S+|\b(?:F1|L1|L2|MD5|SHA256)\b|T\d{4}(?:\.\d{3})?|\d+(?:\.\d+)?e[-+]\d+|\b[A-Za-z0-9_]+\.dll\b|\b[a-fA-F0-9]{32,64}\b',hold,text)
    text=re.sub(r'(?<=[A-Za-z])(?=\d)|(?<=\d)(?=[A-Za-z])',' ',text)
    text=re.sub(r'(?<=[,;:])(?=[A-Za-z\uE000-\uF8FF])',' ',text)
    text=re.sub(r'(?<=[A-Za-z%)])([,;:])(?=\d)',r'\1 ',text)
    text=re.sub(r'(?<=[A-Za-z0-9%)])(?=\()', ' ',text)
    for i,value in enumerate(protected):text=text.replace(chr(0xE000+i),value)
    return text


def main():
    original=json.loads((DOC/'original_audit_matrix.json').read_text(encoding='utf-8'))
    # Responses are tied to generated evidence, not the previous wrong-journal response.
    groups=[
      ('Implemented',['E1','R1.g1','R1.g2','R1.g5','R2.4','R3.2'],
       'The abstract, contributions and Discussion now state the small default-threshold trade-off and absence of established superiority. The RF+LightGBM pair is simpler and has fewer false positives; an independently rerun Logistic Regression baseline also performs better on the official test split. Non-significant McNemar/DeLong tests are not called equivalence. The contribution is a reproducible integration and evaluation pipeline.',
       ['manuscript/main_scirep_revised.pdf','results/core.json']),
      ('Partial: external comparison remains',['R1.1','R2.13'],
       'We rerun compatible Logistic Regression and ExtraTrees baselines on the same official483 features and split, and a five-seed deep feed-forward control. MLRan baseline methodology and published RANSEC/XRGuard are discussed separately. Unrelated dataset scores are removed from the same-split table. Reimplementations of2-3 additional published detector pipelines on this split and an external collection benchmark are not available; no claim is made that those comparisons were completed.',
       ['results/core.json','results/neural_control.json','manuscript/main_scirep_revised.pdf']),
      ('Implemented',['R1.2','R2.5'],
       'The three-model and all pairwise configurations are retained for continuity. Their calibration, common-mask corruption, matched-recall performance and temporal results are measured on identical data. These do not justify claiming the full stack is necessary or best. The simpler RF+LightGBM configuration is preferred over the full stack on default-threshold random-split evidence; deployment selection requires new independent validation. The test FPR difference is2/510=0.39216 percentage points. Equal aggregate RF/RF+XGB confusion counts are distinguished from equality of individual predictions.',
       ['results/core.json','results/corruption_summary.csv','results/temporal_strict.json']),
      ('Partial: raw-report coverage remains',['R1.3','R1.m2','R1.g3','R2.12','R4.1'],
       'All20 available raw reports are excluded from training before a separate parser-evaluation model is fitted on3885 records. We report official-versus-parsed confusion counts for all seven models, probability MAE, all eight reconstruction categories, per-record Jaccard, signed stack-logit attribution correlation and top5 intersection fraction. Strings and Windows paths follow the upstream exact lowercase token contract. All reports are ransomware; benign FPR/AUC and population PPV are undefined. The larger/diverse raw-report validation requested by the reviewer remains unavailable; operational reliability is not claimed.',
       ['results/parser.json','results/figures/parser_stability.png','src/xerd/parser.py']),
      ('Partial: independent validation remains',['R1.4','R1.g4','R2.10','R3.4'],
       'The external schema and all483 feature mappings are supplied, with distinct-technique counts, mapped-feature coverage and active positive trained-stack-logit attribution mass. RWX alone is no longer asserted to be APC injection; weak API observations remain low-specificity candidates. A blank reproducible30-item assessor worksheet is provided, without fabricated labels. No93.3% expert accuracy or Android APIMatch concordance is reported. The supplied reports contain legacy Cuckoo TTP annotations, but their retired IDs and incomplete reference coverage are not an independently adjudicated Windows gold standard. Mapping remains an unvalidated proof of concept pending a real analyst/reference audit.',
       ['config/mapping_schema.json','results/mapping.json','results/independent_mapping_audit_blank.csv']),
      ('External release action pending',['E4','R1.5','R2.14'],
       'Parser, training/evaluation, SHAP, dictionary, tests, data checksums, Docker configuration and result logs are now in the isolated revision codebase. New source code has an MIT license with a third-party NOTICE. Full model settings and nested OOF training are documented. Public upload, release creation and DOI-assigning archival deposit have not occurred. The manuscript states this status and does not assert a DOI. The actual published version DOI must be inserted after the corrected release is deposited.',
       ['README.md','LICENSE','NOTICE','Dockerfile','src/xerd/models.py','manuscript/main_scirep_revised.pdf']),
      ('Implemented',['R1.6','R4.3'],
       'In addition to an explicitly qualified fixed483 temporal diagnostic, a strict temporal analysis selects features and fits models using only2830 samples dated<=2019. A raw73892-feature pool covers APIs,network,system,dropped-file indicators and signatures; training-only MI top2000 and RFE select483. All369 records dated2020 are an unused gap. The1681 test records are dated2021-2024. Exact IDs,selection ranks,derived CSVs,aggregate/yearly confusion counts are exported. File/registry/directory path and string modalities are excluded from this additional pool, so it is not presented as full6.4M-feature RFE or an independent environment.',
       ['src/xerd/additional.py','results/temporal_strict.json','results/temporal_selection.csv','results/derived_data']),
      ('Implemented',['R1.7','R2.7','R3.3'],
       'Thresholds for all nine benchmark classifiers are selected on a stratified training-validation split and evaluated after a full-training refit. Test labels are not used for threshold selection. The0.705 policy is explicitly reported to fail the97% test recall floor(451/465=96.989%); it is not called a feasible test optimum. The0.50 threshold remains a common benchmark reference, not a claimed operational optimum. The constrained criterion is distinguished from training log-loss. Validation/test curves and cost sensitivity across declared prevalences and cost ratios are supplied; the0.055 F1-grid choice is documented.',
       ['results/thresholds.json','results/threshold_curves.csv','results/operational_costs.csv']),
      ('Implemented',['R1.8'],
       'RF and RF+XGBoost were independently refitted with the correct target. Their confusion totals agree, but continuous scores and some individual labels can differ with cancelling errors. The response reports learned coefficients and the actual per-record agreement instead of claiming that every divergent probability remains on the same side of0.50.',
       ['results/core.json','results/test_predictions.csv']),
      ('Implemented',['R1.9','R1.m7','R2.2','R2.3','R3.5'],
       'The protocol is repeated stratified5-fold CV over10 declared seeds, yielding50 paired outer-fold evaluations. Each outer training fold fits fresh5-fold inner OOF predictions and a logistic head. FPR and recall are both analyzed. Nadeau-Bengio corrected t-tests use1/50+1/4=0.27; endpoint-specific Holm families contain four comparisons. Wilcoxon is descriptive, drops zero pairs(zero_method=wilcox), and the effect size is paired rank-biserial rather than mislabeled Cliff delta. We do not interpret overlapping folds as independent replicates.',
       ['results/cv_folds.json','results/cv_summary.json','src/xerd/metrics.py']),
      ('Implemented',['R1.10'],
       'The upstream RFE notebook fits the official representation on official training data. The manuscript also discloses that fixed-feature outer CV is conditional on that selection, and that reusing it temporally permits selection lookahead. A separate raw-pool, past-only feature-selection protocol addresses strict temporal evaluation; no blanket claim of fully nested feature selection is made.',
       ['results/temporal_fixed483.json','results/temporal_strict.json','manuscript/main_scirep_revised.pdf']),
      ('Implemented',['R1.11','R1.12'],
       'Original base hyperparameters are preserved as a fixed supplied configuration, with RF defaults and all package versions exposed. The L2 logistic head usesC=1,lbfgs,max_iter1000,no class weighting. It fits training-only OOF base probabilities, after which bases are refit on the complete training partition. No test information enters fitting, and fresh inner OOF fitting occurs in each outer CV fold.',
       ['src/xerd/models.py','results/training_oof.csv','tests/test_scientific_invariants.py']),
      ('Implemented',['R1.13','R2.1'],
       'The release reports ten-bin ECE,Brier score,reliability diagrams,exact fixed-recall FPR(97%,98%,455/465),standardized pAUC atFPR<=5%,zoomed ROC,and the same validation-threshold procedure for every benchmark model. Descriptive test-derived ROC thresholds are not promoted to deployable fitted thresholds. LightGBM has the lower Brier score, and the RF+LightGBM pair has lower ECE than the full stack; the full stack is not called best calibrated.',
       ['results/core.json','results/thresholds.json','results/figures/matched_roc.png','results/figures/calibration.png']),
      ('Implemented',['R1.14','R2.11','R4.2'],
       'All original base learners and stacks are evaluated under shared corruption masks, with FPR and recall. Feature deletion budgets are consistently0,1,3,5,10,20,40. These are empirical feature-space vulnerabilities,not a theoretical upper bound or validated executable evasion. No-op signature scenarios are disclosed. The discussion distinguishes inexpensive strings/DLL changes from edits that might break malware functionality; no functionality claim is asserted without a problem-space test.',
       ['results/corruption_protocol.json','results/corruption_runs.csv','results/targeted_masking.json']),
      ('Implemented',['R1.15','R2.6','R2.m6','R4.7'],
       'For the arithmetic mean of probabilities, the variance is1^T Sigma1/K^2. We export the complete covariance and correlation matrices. It is not the variance of the logistic stack and not a mechanism validation. Probability-output correlation is not substituted with binary false-positive-indicator correlation. Meta-weight magnitude is not interpreted as proof of calibration. Equation typesetting and indices are corrected.',
       ['results/core.json','manuscript/main_scirep_revised.pdf']),
      ('Implemented',['R1.16'],
       'Full-run aggregate indicators are not described as pre-encryption evidence. The paper separates timestamped API prefixes from untimed strings/signatures/summary telemetry and removes claims of early prevention or a latency guarantee. A computational timing benchmark excludes sandbox execution and encryption onset.',
       ['results/api_prefix_protocol.json','results/computational_latency.json']),
      ('Implemented',['R1.m1','R1.m3','R1.m6','R2.m1','R2.m2','R2.m3','R2.m4','R2.m5'],
       'The revision regenerates tables/figures from result files. Rates/F1 use consistent percentage units,whole-dataset and ransomware-only year ranges are distinguished,and figure references follow the generated numbering. The global plot contains14 sorted feature IDs with full companion CSV names. MLRan is called a benchmark rather than an established standard; lower-bound-plus-SD wording is removed.',
       ['scripts/build_manuscript.py','src/xerd/figures.py','results/global_shap.csv']),
      ('Implemented',['R1.m4'],
       'Both recall and benign FPR are exported per seed and model for all corruption conditions,with common masks and explicit SD. Signature scenarios with no selected matching features are reported as no-ops instead of evidence of evasion resilience.',
       ['results/corruption_runs.csv','results/corruption_summary.csv']),
      ('Implemented',['R2.8'],
       'Every TreeExplainer uses model_output=probability,interventional perturbation and the same100 training-background records. Unnormalized fitted meta-coefficients produce exact additive stack-logit attributions; the sigmoid reconstructs the predicted probability within the asserted tolerance. Uniform averaging is a separate probability-mean surrogate. Deletion uses the same300 stratified records,0-20 grid,active features,true-class confidence,random control,per-sample AUC,uncertainty and paired tests. Out-of-distribution caveats and deterministic100-record display sampling are explicit.',
       ['results/explanations.json','results/deletion.json','results/deletion_samples.csv','results/deletion_curves.csv']),
      ('Implemented',['R2.9'],
       'Class-specific training/test prevalence is supplied for every feature,and no-string plus no-string/two-DLL retraining uses the identical original fitting procedure. This detects sensitivity to artifacts but is not proof of artifact independence. PE stubs,generic strings and DLL loads are not described as ransomware-specific or pre-encryption behavior.',
       ['results/feature_prevalence.csv','results/artifact_ablation.json']),
      ('Partial: independent dataset remains',['R3.1'],
       'A strict past-only feature-selection holdout is added and generalization claims are limited to MLRan. No independent collection,enterprise environment,or supervised unseen-family experiment has been supplied or invented. An external benchmark remains necessary before making those broader claims.',
       ['results/temporal_strict.json','manuscript/main_scirep_revised.pdf']),
      ('Implemented with uncalibrated specificity',['R4.4'],
       'A versioned external schema associates exact/prefix evidence rules with candidate techniques,rationale,qualitative tiers and ordinal rule-specificity scores0-3. Multiple features may suggest one technique. The score represents specificity,not a calibrated accuracy or maliciousness probability; weak evidence does not imply a specific injection sub-technique. Independent calibration remains part of the external audit requirement.',
       ['config/mapping_schema.json','src/xerd/mapping.py','results/mapping.json']),
      ('Partial: early-prevention evidence remains',['R4.5'],
       'All20 reports contain timestamped API calls in behavior.processes[*].calls[*].time. A separately fitted API-only stack uses0,1,5,15,30,60-second prefixes from the first observed call,excluding all evaluation reports and all untimed full-run features. Results are exploratory. Encryption-onset labels and benign raw reports are absent,so pre-encryption viability and prefix FPR are not established.',
       ['results/api_prefix_predictions.csv','results/api_prefix_protocol.json']),
      ('Partial: supervised sequence benchmark remains',['R4.6'],
       'A genuine three-hidden-layer feed-forward baseline is run on the official unordered indicators with training-only early stopping and five seeds. It is not described as a sequence model. A relevant supervised CNN/LSTM/Transformer sequence comparison requires labeled benign and ransomware sequences; the20 available ransomware-only reports cannot supply that experiment. No artificial feature-order sequence result is claimed.',
       ['results/neural_control.json','src/xerd/additional.py']),
      ('Implemented',['R1.m5','R2.m7','R2.m8','R2.m9','R4.8'],
       'The revised bibliography uses the published2026 MLRan article and corrects the identified cited metadata(DIMVA venue,Adelodun surname,XRGuard author/pages,APIMatch pages,DeLong title and Pierazzi title/pages/DOI). The unverified Connolly entry and unsupported non-scholarly claim are not cited. Related Work distinguishes different-dataset detectors,Android API mapping and actual same-split reruns. Alert-fatigue claims are tempered rather than supported by unrelated ensemble citations.',
       ['scripts/build_bibliography.py','docs/bibliography_corrections.json','manuscript/references_revision.bib']),
      ('Optional: editorial discretion',['E3','R4.9'],
       'The editor explicitly states that suggested references are discretionary. References are included for relevance rather than to satisfy a numeric citation list. The duplicated suggested DOI is not counted twice,and unrelated citations are not added automatically. Authors can add directly relevant suggested studies after checking their exact claims and metadata.',
       ['docs/bibliography_corrections.json']),
      ('Draft prepared; author review remains',['E2'],
       'A new Scientific Reports manuscript and point-by-point response cover the editor and all four actual reviewers. The earlier unrelated Elsevier/two-reviewer response is not reused. The generated draft transparently marks incomplete independent experiments and the unpublished archive. Author approval and final journal-facing wording are required before submission.',
       ['docs/reviewer_response_draft.txt','manuscript/main_scirep_revised.pdf'])
    ]
    lookup={key:(status,response,evidence) for status,ids,response,evidence in groups for key in ids}
    missing={r['id'] for r in original['matrix']}-set(lookup)
    if missing:raise ValueError(f'Unanswered reviewer items:{missing}')
    rows=[]
    for r in original['matrix']:
        status,response,evidence=lookup[r['id']]
        if r['id'] in ['R1.3','R1.m2','R1.g3','R2.12','R4.1','R4.5']:
            response+=' The upstream fixed feature selection included these20 records; exclusion is from model fitting only. These are conditional representation checks,not an untouched feature-selection holdout.'
        rows.append({'id':r['id'],'reviewer':r['group'],'comment_topic':r['topic'],'status':status,'response':response,'evidence':list(evidence)})
    remaining=[
      'Public GitHub release and an actual published Zenodo version DOI; no DOI exists yet.',
      'Independent analyst/adjudicated reference evaluation of ATT&CK mapping; the supplied blank worksheet is not a completed audit.',
      'More raw reports including benign controls,and encryption-onset annotations for early-prevention claims.',
      'Supervised deep sequence comparison with properly labeled benign/ransomware sequences.',
      'Same-split implementations of additional published detector pipelines; separate-dataset published scores cannot replace them.',
      'Independent collection/family/environment validation before broad deployment or zero-day claims.'
    ]
    from build_extended_report import update_reviewer_items
    rows,remaining=update_reviewer_items(rows,ROOT)
    from build_remediation_report import update_items,html_update
    specific={
      'E2':'The revised manuscript, Supplementary Information and this point-by-point response address the editor and all four reviewer reports. Each response identifies the changed sections and supporting evidence. Results that limit the original claims, incomplete independent adjudication and the pending public archive are stated explicitly. This local revision is awaiting author approval.',
      'R1.m1':'The threshold table has been regenerated from the exported selection results. Validation policy, selected threshold, validation recall, test recall, test FPR and test F1 now have distinct aligned columns in Supplementary Table S3.',
      'R1.m3':'MLRan is described as the benchmark used in this study. The text no longer describes it as an established or standardized benchmark.',
      'R1.m6':'All main displays and supplementary figures now have explicit narrative callouts. Supplementary Figures S1-S14 are first cited in numerical order; the dataset composition and temporal distribution are S1 and S2, respectively.',
      'R2.m1':'The dataset paragraph cites Supplementary Figure S1 for composition and Supplementary Figure S2 for time distribution. Figure S1 explicitly identifies class counts as panel (a) and ransomware categories as panel (b).',
      'R2.m2':'Supplementary Figure S2 distinguishes the 2006-2024 range for all records from the 2008-2024 range for ransomware. Its caption explains that first-submission year is not verified malware creation time.',
      'R2.m3':'Supplementary Figure S8 now shows 14 bars in descending mean absolute trained-stack-logit attribution order. Supplementary Table S6 supplies all 14 complete feature tokens, including ProfileImagePath and MountPoints2; the CSV preserves their exact strings and values.',
      'R2.m4':'The lower-bound-plus-standard-deviation claim has been removed. Supplementary Table S10 reports measured recall and FPR as mean plus/minus sample SD across the ten declared corruption seeds, with deterministic no-op scenarios identified.',
      'R2.m5':'Accuracy, precision, recall, F1 and FPR are consistently reported as percentages in the classification tables. The matched-operating-point table explicitly labels its standardized partial AUC, which remains on a 0-1 scale.',
      'R2.m7':'Ensemble-learning citations are no longer used to support operational alert-fatigue claims. The text restricts the motivation to measured FPR and stated cost sensitivity. The problem-space feasibility discussion cites Pierazzi et al.; generic adversarial-learning citations are not presented as validation of our feature-dropout procedure.',
      'R2.m8':'The unsupported non-scholarly background claim and its associated citation have been removed. The ransomware motivation now cites the relevant technical studies by Kharraz et al., Chen and Bridges, and Scaife et al.',
      'R2.m9':'A dedicated Related Work section now positions X-ERD against MLRan baselines, EldeRan, RANSEC and XRGuard, and distinguishes sequence-oriented iCNN-LSTM+ and API-RED. Supplementary Table S1 states which methods were executed as adaptations and which provide context only; unrelated published dataset scores are not ranked against this benchmark.'}
    for row in rows:
        if row['id'] in specific:row['response']=specific[row['id']]
    rows,remaining=update_items(rows,remaining,ROOT)
    for row in rows:
        row['response']=clean_prose(row['response'])
        row['evidence']=list(dict.fromkeys(row['evidence']))
    counts=dict(collections.Counter(r['status'] for r in rows));result={'verdict':'Corrected local implementation and measured revision draft; NOT unconditional submission clearance','items':rows,'counts':counts,'remaining':remaining}
    (DOC/'reviewer_revision_status.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    response=['SCIENTIFIC REPORTS - POINT-BY-POINT RESPONSE DRAFT','Prepared from the actual editor and four reviewer reports. Author review required.','No public DOI,completed independent audit,or unavailable experiment is asserted.','']
    for r in rows:
        response += [f'{r["id"]} | {r["reviewer"]}',f'Comment topic: {r["comment_topic"]}',f'Status: {r["status"]}','Response: '+r['response'],'Evidence: '+', '.join(r['evidence']),'']
    (DOC/'reviewer_response_draft.txt').write_text('\n'.join(response),encoding='utf-8')
    m=read('core')['metrics'];temp=read('temporal_strict')['metrics'];parser=read('parser')
    from build_extended_report import evidence
    ext=evidence(ROOT);residual=ext['residual']['models']['X-ERD'];raw=ext['external_raw']
    intro='Naya codebase alag folder mein banaya gaya hai. Original RF/XGB/LightGBM-logistic architecture aur official benchmark preserve hain. Reported scores actual Docker experiments se aaye hain. Code/theoretical corrections complete hone ka matlab har external reviewer experiment complete hona nahi hai; neeche pending points openly diye hain.'
    page=['<!doctype html><html><head><meta charset="utf-8"><title>X-ERD corrected revision report</title><style>body{font:16px/1.65 system-ui;max-width:1100px;margin:40px auto;padding:0 24px;color:#172d3d}h1,h2{line-height:1.25}table{border-collapse:collapse;width:100%;font-size:14px}td,th{padding:10px;border:1px solid #ccd8df;text-align:left}th{background:#edf3f6}.pending{padding:18px;background:#fff0d5;border-left:5px solid #bf8100}.item{padding:15px 0;border-bottom:1px solid #ccd8df}code{background:#edf3f6;padding:2px 5px}a{color:#16628a}</style></head><body><h1>X-ERD: corrected code and revision analysis</h1><p>'+intro+'</p><div class="pending"><b>Submission verdict:</b> Local experiments aur manuscript draft taiyar hain; abhi sirf repository link daalkar submit karne ki clearance nahi hai. Independent evidence/public archive ke points bache hain. Reviewer acceptance guarantee nahi ki ja sakti.</div><h2>Verified default-threshold results</h2><table><tr><th>Model</th><th>Accuracy</th><th>Recall</th><th>FPR</th><th>TN/FP/FN/TP</th></tr>']
    for n in ['LGB','RF+LGB','X-ERD','LogisticRegression','ExtraTrees']:
        r=m[n];page.append(f'<tr><td>{html.escape(n)}</td><td>{r["accuracy"]*100:.2f}%</td><td>{r["recall"]*100:.2f}%</td><td>{r["fpr"]*100:.2f}%</td><td>'+ '/'.join(str(r[k]) for k in ['tn','fp','fn','tp'])+'</td></tr>')
    page+=['</table><p>Full stack best model nahi nikla. Uska advantage force nahi kiya gaya. Strong Logistic Regression result aur simpler pair ka result paper mein disclosed hai.</p>',f'<p>Strict temporal full-stack recall:{temp["X-ERD"]["recall"]*100:.2f}%;FPR:{temp["X-ERD"]["fpr"]*100:.2f}%. Parser:20 excluded ransomware reports;mean Jaccard:{parser["mean_jaccard"]:.6f}. Benign FPR unavailable.</p>','<h2>Preserved and corrected</h2><p>Official split,483 benchmark features,three base learners,logistic heads,pairwise configurations,parser,SHAP,mapping,corruption and case study remain. Target leakage,inconsistent SHAP scales,mixed statistics,unsupported expert accuracy,lookahead scope and hand-copied numbers are corrected. The extra strict temporal dataset is generated from a declared raw pool using training-only selection; no labels or observations are synthesized.</p>','<h2>Remaining evidence and release steps</h2><ol>']
    page.insert(-1,'<h2>Measured extension and external evidence</h2><p>Original full stack accuracy97.23% se passthrough extension accuracy'+f'{100*residual["accuracy"]:.2f}%'+
      ' hui; false positives17 se12 hue aur recall455/465 par raha. Paired confidence interval zero ko include karta hai; universal superiority established nahi hai. Same hash-novel external transfer par recall80% aur FPR12.79% raha, compared with76.67% and21.43% for the original stack.</p>'+
      '<p>Published MLRan grids,EldeRan/RANSEC adaptations,genuine80-trace LSTM/CNN comparisons,200 preselected benign raw reports,'+str(raw['evaluated_unique_n'])+
      ' accepted unique executions,MD5-grouped sensitivity and source-reference ATT&amp;CK concordance ab complete result files mein hain. Main paper ke8 displays ke alawa detailed evidence supplementary_information.pdf mein preserved hai.</p>')
    raw_new=ext['external_raw_candidates']['models']['Feature passthrough']['X-ERD']
    page.insert(-1,'<h2>Critical external result</h2><p>'+f'Independent publisher-clean raw reports par original full stack ne {raw["models"]["X-ERD"]["fp"]}/{raw["evaluated_unique_n"]} false alerts diye ({100*raw["models"]["X-ERD"]["fpr"]:.2f}% FPR). Passthrough variant ka FPR {100*raw_new["fpr"]:.2f}% raha.'+
      ' Yeh major transport limitation hai. Is result ke baad low-false-alarm end-to-end deployment ka claim supported nahi hai. Paper abstract, main results aur reviewer response mein yeh outcome explicitly diya hai.</p>')
    page += ['<li>'+html.escape(x)+'</li>' for x in remaining]
    page += ['</ol><h2>Every reviewer item</h2>']
    for r in rows:
        page.append('<div class="item"><h3>'+html.escape(r['id']+' - '+r['comment_topic'])+'</h3><b>'+html.escape(r['status'])+'</b><p>'+html.escape(r['response'])+'</p><p>Evidence: '+', '.join('<code>'+html.escape(e)+'</code>' for e in r['evidence'])+'</p></div>')
    page+=['<h2>GitHub and Zenodo workflow</h2><p>First review this measured draft and remaining evidence. Upload the corrected source release,not the old ZIP. Create a versioned GitHub release and archive that exact version with Zenodo. Insert the actual published DOI and matching repository tag/commit into Code Availability. Rebuild the PDF and rerun the release verification. This workspace has not pushed or published anything.</p></body></html>']
    page=''.join(page).replace('Benign FPR unavailable.','Is original reconstruction subset mein benign FPR undefined hai; separate external controls neeche available hain.')
    page=html_update(page,ROOT)
    (DOC/'Complete_Revision_Report.html').write_text(page,encoding='utf-8')
    print('Wrote actual4-reviewer response:',len(rows),'items;',counts)

if __name__=='__main__':main()
