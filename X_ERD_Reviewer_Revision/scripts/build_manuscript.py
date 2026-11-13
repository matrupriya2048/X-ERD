"""Build the revision from verified result files. Never hand-transcribe scores."""
from pathlib import Path
import json,re,shutil,textwrap
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results';MAN=ROOT/'manuscript'
def read(name):return json.loads((OUT/(name+'.json')).read_text(encoding='utf-8'))
def esc(s):
    return str(s).replace('\\',r'\textbackslash{}').replace('_',r'\_').replace('%',r'\%').replace('&',r'\&').replace('#',r'\#')
def num(x,n=2):return '--' if x is None else f'{x:.{n}f}'
def pc(x):return '--' if x is None else num(100*x)
def table(label,caption,headers,rows):
    return (r'\begin{table*}[t]\centering\small'+ '\n'+r'\caption{'+caption+'}'+ '\n'+r'\label{tab:'+label+r'}\begin{lrbox}{\xerdtablebox}\begin{tabular}{l'+'c'*(len(headers)-1)+r'}\toprule'+'\n'+
      ' & '.join(headers)+r' \\ \midrule'+'\n'+'\n'.join(' & '.join(map(str,row))+r' \\' for row in rows)+
      '\n'+r'\bottomrule\end{tabular}\end{lrbox}\ifdim\wd\xerdtablebox>\textwidth\resizebox{\textwidth}{!}{\usebox{\xerdtablebox}}\else\usebox{\xerdtablebox}\fi\end{table*}'+'\n')
def figure(name,label,caption,wide=False):
    env='figure*';width=r'\textwidth' if wide else r'0.85\textwidth'
    if name=='fig_class_category_real.png':width=r'0.68\textwidth'
    return '\n'+r'\begin{'+env+r'}[t]\centering\includegraphics[width='+width+']{'+name+r'}\caption{'+caption+r'}\label{fig:'+label+r'}\end{'+env+'}\n'

def main():
    required=['core','cv_summary','thresholds','temporal_fixed483','temporal_strict','explanations','deletion','parser','mapping','artifact_ablation','neural_control']
    j={n:read(n) for n in required};core=j['core'];m=core['metrics'];full=m['X-ERD'];pair=m['RF+LGB'];lgb=m['LGB']
    raw=MAN/'supplied_reference_only.tex'
    if not raw.exists():shutil.copy2(MAN/'main_scirep.tex',raw)
    source=raw.read_text(encoding='utf-8');preamble=source.split(r'\begin{abstract}')[0]
    preamble=re.sub(r'\\title\{.*?\}\n',r'\\title{Explainable ransomware detection with stacked ensembles and rule-based MITRE ATT\\&CK suggestions}\n',preamble)
    preamble=preamble.replace(r'\graphicspath{{figures/}{./}}',r'\graphicspath{{../results/figures/}{./}}')
    preamble+=r'\usepackage{placeins}\newsavebox{\xerdtablebox}'+'\n'
    preamble+=r'\renewcommand{\topfraction}{0.85}\renewcommand{\dbltopfraction}{0.85}\renewcommand{\textfraction}{0.12}\renewcommand{\dblfloatpagefraction}{0.75}\setcounter{topnumber}{2}\setcounter{dbltopnumber}{2}\setcounter{totalnumber}{4}'+'\n'
    abstract=(r'\begin{abstract}'+'\n'+
      f'X-ERD combines Random Forest, XGBoost and LightGBM through an L2-regularized logistic meta-learner, with behavioral parsing, additive attribution and candidate MITRE ATT\\&CK mapping. On the official MLRan split, the full stack achieves {pc(full["accuracy"])}\\% accuracy, {pc(full["recall"])}\\% recall and {pc(full["fpr"])}\\% false positive rate (FPR). The RF+LightGBM stack has {pc(pair["fpr"])}\\% FPR at equal recall, and an additional fixed Logistic Regression baseline achieves {pc(m["LogisticRegression"]["accuracy"])}\\% accuracy; the full stack is therefore not an overall performance winner. Repeated stratified five-fold cross-validation over ten seeds yields 50 paired evaluations; corrected resampling inference does not establish a full-stack FPR advantage over LightGBM. A separate past-only feature-selection experiment trains on 2,830 pre-2020 samples and tests 1,681 samples from 2021--2024, obtaining {pc(j["temporal_strict"]["metrics"]["X-ERD"]["recall"])}\\% recall. Common probability-scale base SHAP values, propagated through the fitted logistic coefficients, explain the stack logit. Parser evaluation excludes all20 available ransomware reports from training; all are detected, but benign FPR and population validity cannot be estimated. ATT\\&CK suggestions remain a proof of concept without independent accuracy labels. These findings support a reproducible benchmark pipeline and expose its statistical, temporal and operational limits; they do not establish early prevention or enterprise deployment performance.'+
      '\n'+r'\end{abstract}\begin{document}\raggedbottom\maketitle\thispagestyle{empty}'+'\n'+
      r'\noindent\textbf{Revision draft for author review.} Public code archive DOI and independent ATT\&CK/sequence validation remain incomplete; no public-archive or expert-audit claim is made.'+'\n')
    text=preamble+abstract
    # Preserve the supplied introduction's background paragraph, not its unsupported claims.
    intro=source.split(r'\section*{Introduction}')[1].split('Despite encouraging')[0]
    intro=intro.replace(r'\cite{kharraz2015cutting,chen2017wannacry,connolly2020ransomware}',r'\cite{kharraz2015cutting,chen2017wannacry}')
    intro=intro.replace(r'\cite{kharraz2015cutting,scaife2016cryptolock,oz2022survey}',r'\cite{kharraz2015cutting,scaife2016cryptolock}')
    intro=intro.replace('economically catastrophic','damaging').replace('effectively neutralize conventional static signature defenses','can evade conventional static signature defenses')
    text+=r'\section*{Introduction}'+intro+r'''
Operational evaluation requires separating ranking performance, decision thresholds, probability calibration and analyst interpretation. A lower FPR at one threshold can accompany lower recall; repeated folds are dependent, and post-hoc attribution identifies model reliance rather than malicious intent. Temporal selection must precede the future test period to support drift analysis\cite{pendlebury2019tesseract}. Full-run aggregate features cannot establish whether an alert precedes encryption.

We retain the original X-ERD architecture and all pairwise ablations, adding matched-recall comparisons, corrected inference for both FPR and recall, validation-only threshold policies, probability-consistent attribution, strict temporal feature selection, artifact ablations, parser holdouts and transparent mapping coverage. The contribution is an auditable integration and evaluation of these components, rather than a new tree algorithm or demonstrated superiority over simpler classifiers. The simpler RF+LightGBM configuration is preferred to the full stack on the reported default-threshold random-split results; deployment model selection would require independent validation.

\section*{Related Work}
Dynamic behavioral analysis complements signature and static-file detection\cite{ormeir2019dynamic,scaife2016cryptolock}. MLRan supplies a multi-family behavioral dataset, official partitions, feature selection and baseline pipelines\cite{onwuegbuche2025mlran}. Stacking\cite{wolpert1992stacked} combines learned probability estimates; it does not directly optimize a constrained FPR objective. RANSEC\cite{suthar2026ransec} and XRGuard\cite{alvi2025xrguard} are relevant ransomware detectors, but their published experiments use different data/telemetry. Their scores cannot be treated as results on our official split. We therefore distinguish published context from independently rerun, compatible algorithm baselines; reimplementations of two or three additional published detector pipelines on the same split remain unavailable.

SHAP\cite{lundberg2017unified,lundberg2020tree} provides an additive value-function decomposition; a uniform average of inconsistent probability and margin outputs is invalid. APIMatch-ATT\&CK concerns Android API mapping\cite{li2026apimatch} and is not Windows behavioral ground truth. Security ML must also investigate collection artifacts\cite{arp2022dos} and distinguish feature-space manipulations from executable attacks\cite{pierazzi2020intriguing}. A feed-forward neural control on unordered indicators is informative about classification, but does not replace a supervised API sequence benchmark.

\section*{Results}
\subsection*{Benchmark dataset and retained architecture}
The official benchmark contains4,880 records:2,550 goodware and2,330 ransomware from64 families. We preserve the original3,905/975 training/test partition, with510 benign and465 ransomware test samples and483 binary RFE indicators. Figure~\ref{fig:dataset_overview} shows composition and Figure~\ref{fig:temporal_dist} distinguishes the year range for all samples from the ransomware-only range. The benchmark notebooks fit RFE on the official training partition. Nevertheless, that fixed representation is not selected anew inside each outer CV fold, and includes post-2019 records in its original training partition. Random-split CV is conditional on this representation; a strict chronological representation is constructed separately below.
'''
    text+=figure('fig_class_category_real.png','dataset_overview','Dataset composition retained from the supplied manuscript. Panel(a):goodware and ransomware; panel(b):ransomware categories.')
    text+=figure('temporal_distribution.png','temporal_dist','First-submission years. All-sample and ransomware-only ranges are explicitly distinguished; collection/submission date is a proxy for time, not guaranteed malware creation time.')
    text+=figure('architecture.png','architecture','Retained architecture with corrected attribution. Raw parsing is evaluated separately; stack-logit explanations and candidate ATT\&CK rules do not imply validated end-to-end deployment.',True)
    text+=r'\subsection*{Held-out classification and calibration}'+'\n'
    text+=table('performance',r'Official held-out split at $\theta=0.50$. Rates are percentages; confusion counts use true class0=goodware and1=ransomware. All rows are rerun algorithms, not scores copied from unrelated datasets.',
      ['Model','Accuracy','Precision','Recall','F1','AUC','FPR','TN/FP/FN/TP'],
      [[esc(n),*[pc(r[k]) for k in ['accuracy','precision','recall','f1','auc','fpr']],'/'.join(str(r[k]) for k in ['tn','fp','fn','tp'])] for n,r in m.items()])
    preds=pd.read_csv(OUT/'test_predictions.csv');same=int(((preds.RF>=.5)==(preds['RF+XGB']>=.5)).sum());coef=core['coefficients']['RF+XGB']
    text+=f'The full stack exchanges {full["fn"]-lgb["fn"]} additional missed ransomware samples for {lgb["fp"]-full["fp"]} fewer false alarms relative to LightGBM. RF+LightGBM has {full["fp"]-pair["fp"]} fewer false positives than the full stack, a difference of {pc(full["fpr"]-pair["fpr"])} percentage points. Neither stacking nor a third learner is required to obtain the strongest held-out score: Logistic Regression performs better on this split. This is a fixed baseline comparison, not independent evidence for a post-hoc deployment choice.\n\n'
    text+=f'RF and RF+XGBoost share the same confusion totals but have identical binary predictions on {same}/975 records; different error identities can cancel in aggregate. Their learned coefficients are {num(coef["coef"][0],4)} and {num(coef["coef"][1],4)}, with intercept {num(coef["intercept"],4)}. Their continuous probability rankings and AUCs need not be identical.\n\n'
    text+=table('calibration','Calibration on the official held-out split. ECE uses ten equal-width bins including probability0 and1; smaller values are better. No post-hoc calibrator is fitted.',
      ['Model','Brier score','ECE'],[[esc(n),num(r['brier'],6),num(r['ece'],6)] for n,r in m.items()])
    text+=r'LightGBM has a lower Brier score than the full stack; RF+LightGBM has lower ECE than both. ECE depends on binning and sample size. A large RF meta-coefficient is not proof of calibration, and a predicted probability should not be presented as a validated operational confidence.'+'\n'
    text+=figure('calibration.png','calibration','Reliability diagram using ten equal-width bins on the official test partition.')
    text+=r'\subsection*{Matched recall and threshold selection}'+'\n'
    text+=table('matched','Descriptive test-set ROC comparisons. Thresholds derived from these test labels are not deployment policies. Partial AUC is standardized over FPR$\leq0.05$.',
      ['Model','FPR at97\% recall','FPR at98\% recall','FPR at455/465 recall','Partial AUC'],
      [[esc(n),pc(core['matched_descriptive']['0.97'][n]['fpr']),pc(core['matched_descriptive']['0.98'][n]['fpr']),pc(core['matched_descriptive'][str(455/465)][n]['fpr']),num(core['standardized_partial_auc_005'][n],6)] for n in m])
    text+=figure('matched_roc.png','roc','ROC curves and the low-FPR region. Matched-recall results describe ranking behavior, separating a threshold shift from a ranking improvement.',True)
    t=j['thresholds'];tr=[]
    for n,r in t.items():
        for policy,key,vkey in [('F1 maximum','test_val_f1max','validation_f1max'),('Recall-constrained','test_val_constrained','validation_constrained')]:
            q=r[key];tr.append([esc(n),policy,num(q['theta'],3),pc(r[vkey]['recall']),pc(q['recall']),pc(q['fpr']),pc(q['f1'])])
    text+=table('thresholds','Thresholds selected on a stratified20\% validation partition of official training data, followed by a full-training refit. The validation constraint is empirical recall$\geq0.97$; test recall is reported without reselection.',
      ['Model','Validation policy',r'$\theta$','Val. recall','Test recall','Test FPR','Test F1'],tr)
    text+=r'''The constrained criterion is an evaluation-time policy, not the model training objective. Training minimizes logistic log-loss. The full stack's validation-selected0.705 threshold gives451/465=96.989\% test recall and therefore fails the0.97 floor on this test set; rounding does not make it feasible. We retain0.50 only as the common benchmark reference, not as a claimed optimal deployment threshold. The0.055 F1 threshold is the lowest grid point attaining the validation maximum; the exported curves show the sensitivity and ties rather than establishing a universal plateau. Selection uses the fixed official representation, whose earlier feature selection included validation records; threshold validation is conditional on that representation.

Expected-cost sensitivity is exported for all seven original models and three policies at prevalence0.001,0.01 and465/975, and false-negative/false-positive cost ratios0.1--1000. The cost is $10^4[(1-\pi)\mathrm{FPR}+c\pi\mathrm{FNR}]$. This compares stated candidates; it neither establishes a universal optimum nor transports benchmark rates to enterprise populations without validation.
'''
    text+=figure('threshold_stability.png','threshold','Training-validation and descriptive test FPR/recall curves. Test curves are diagnostic and are not used to choose thresholds.',True)
    text+=r'\subsection*{Repeated cross-validation and corrected inference}'+'\n'
    rows=[]
    for endpoint in ['fpr','recall']:
        for n,r in j['cv_summary'][endpoint]['comparisons'].items():rows.append([endpoint.upper(),esc(n),num(100*r['mean_difference'],4),num(100*r['sd_difference'],4),f'{r["lower"]}/{r["higher"]}/{r["ties"]}',num(r['nb_p'],5),num(r['nb_holm_p'],5),num(r['paired_rank_biserial'],4)])
    text+=table('cv','Full stack minus comparator on50 paired outer-fold evaluations. Lower/higher/tied counts refer to the named endpoint; a lower recall is undesirable. NB=corrected resampled t-test, followed by endpoint-specific Holm correction over four comparisons.',
      ['Endpoint','Comparator','Mean(pp)','SD(pp)','Lower/higher/ties','NB p','Holm p','Paired rank-biserial'],rows)
    text+=f'The corrected FPR comparison against LightGBM gives p={num(j["cv_summary"]["fpr"]["comparisons"]["LGB"]["nb_p"],5)}; no statistically established full-stack advantage is demonstrated. The exact test-set McNemar p={num(core["mcnemar"]["p"],5)} and DeLong p={num(core["delong"]["p"],5)} do not establish equivalence. No equivalence margin was specified. Confidence intervals and all50 observations are supplied; repeated folds remain dependent, so correction is approximate.\n'
    text+=figure('cv_corrected.png','cv','Corrected repeated-CV FPR differences with approximate95\% intervals. These are conditional on the fixed official representation.')
    text+=r'\subsection*{Chronological evaluation and selection scope}'+'\n'
    temp=j['temporal_strict'];fixed=j['temporal_fixed483']
    text+=f'The fixed483 chronological diagnostic trains through2019 and tests2020--2024, yielding full-stack accuracy{pc(fixed["metrics"]["X-ERD"]["accuracy"])}\\% and recall{pc(fixed["metrics"]["X-ERD"]["recall"])}\\%. It is explicitly subject to feature-selection lookahead and is not the strict temporal result. For the separate strict analysis we start from{temp["candidate_features"]:,} raw API, network, system, dropped-file and signature indicators. Only2,830 pre-2020 records enter variable filtering, discrete mutual-information selection(top2000), RFE to483 and model fitting. All369 records from2020 form an unused gap;1,681 records dated2021--2024 form the test set(587 benign,1,094 ransomware). Registry/file/directory paths and strings are excluded from this additional candidate pool; this is not full6.4-million-feature RFE.\n'
    text+=table('temporal','Strict chronological holdout using past-only feature selection in the declared raw candidate pool. No future labels are used in selection, fitting or threshold tuning.',
      ['Model','Accuracy','Recall','FPR','TN/FP/FN/TP'],[[esc(n),pc(r['accuracy']),pc(r['recall']),pc(r['fpr']),'/'.join(str(r[k]) for k in ['tn','fp','fn','tp'])] for n,r in temp['metrics'].items()])
    text+=table('years','Year-specific strict chronological full-stack results. Counts reconcile exactly with the aggregate. Small yearly class counts limit precision.',
      ['Year','Benign','Ransomware','TN/FP/FN/TP','Recall','FPR'],[[yr,r['X-ERD']['tn']+r['X-ERD']['fp'],r['X-ERD']['tp']+r['X-ERD']['fn'],'/'.join(str(r['X-ERD'][k]) for k in ['tn','fp','fn','tp']),pc(r['X-ERD']['recall']),pc(r['X-ERD']['fpr'])] for yr,r in temp['years'].items()])
    text+=figure('temporal_strict.png','temporal','Per-year recall and FPR for the strict past-only representation. This tests chronological shift within MLRan, not an independent collection environment.',True)
    text+=r'\subsection*{Correlated probabilities and variance identity}'+'\n'+r'''
On benign test records, let $\Sigma$ be the empirical covariance matrix of the three base probabilities. The exact identity
\begin{equation}\begin{aligned}\operatorname{Var}\!\left(\frac{1}{K}\sum_{k=1}^{K}p_k\right)&=\frac{\mathbf{1}^{\top}\Sigma\mathbf{1}}{K^2}\\&=\frac{\sum_k\operatorname{Var}(p_k)+2\sum_{j<k}\operatorname{Cov}(p_j,p_k)}{K^2}\end{aligned}\label{eq:variance}\end{equation}
concerns the arithmetic mean, not the nonlinear logistic stack. Agreement of both sides is guaranteed by algebra and does not explain a causal stacking gain. Probability correlations are distinct from correlations of binary false-positive indicators. We export the full covariance and correlation matrices rather than substituting one correlation definition for another.
'''
    text+=r'\subsection*{Probability-consistent attribution and deletion sensitivity}'+'\n'
    text+=r'''For base probabilities $p_k(x)=\phi_{0k}+\sum_j\phi_{jk}(x)$ under a common interventional background, linearity yields
\begin{equation}\begin{aligned}g(x)&=b+\sum_k w_kp_k(x)=\Phi_0+\sum_j\Phi_j(x),\\\Phi_j&=\sum_k w_k\phi_{jk},\qquad\Phi_0=b+\sum_k w_k\phi_{0k}.\end{aligned}\end{equation}
Here$g$ is the fitted stack logit and$\sigma(g)$ reconstructs its probability. The coefficients are not normalized. This gives SHAP for the additive logit under the stated value function; the values are not full-stack probability-space SHAP. Uniform base averaging explains the probability-mean surrogate only.
'''
    ex=j['explanations'];text+=f'The maximum probability reconstruction error is{ex["max_stack_probability_reconstruction_error"]:.3g}. Global attribution uses100 deterministic indices evenly spaced across the ordered test partition; this is a declared display sample, not a random representative sample. All975 test attributions are exported. Figure~\\ref{{fig:global}} shows14 sorted feature IDs, with complete names and values in the accompanying CSV.\n'
    text+=figure('global_shap.png','global','Top14 features by mean absolute trained-stack-logit SHAP on the declared100 display records. Full names and feature IDs are supplied in global\_shap.csv.')
    de=j['deletion']['results'];text+=table('deletion','Normalized deletion AUC over0--20 active-feature deletions on the same300 stratified test records(150 per class). The response is true-class confidence:1-p for benign andp for ransomware. Lower values mean stronger empirical deletion sensitivity, not proven explanation correctness.',
      ['Ranking','Mean AUC','Sample SD','95\% mean interval'],[[esc(n),num(de['all'][n]['mean'],6),num(de['all'][n]['sample_sd'],6),'['+', '.join(num(x,6) for x in de['all'][n]['mean_ci95_normal'])+']'] for n in ['LGB','uniform_surrogate','stack_logit','random']])
    text+=f'Against the LGB ranking, trained-stack-logit ranking gives mean paired AUC difference{num(de["all"]["stack_vs_LGB"]["mean_difference"],6)}(paired t p={de["all"]["stack_vs_LGB"]["paired_t_p"]:.3g}). Uniform-surrogate ranking has a lower deletion AUC than stack-logit ranking in this experiment; this does not turn it into a faithful explanation of the trained stack. These exploratory paired tests are not corrected for all explanation comparisons. Signed rankings target the true class and zero only active indicators. Zeroing can create out-of-distribution records, so the experiment measures sensitivity rather than correctness or feasible evasion.\n'
    text+=figure('deletion.png','deletion','Deletion sensitivity by true class, with approximate95\% intervals of the mean. All rankings are tested against the same trained full-stack model.',True)
    ab=j['artifact_ablation']['results'];text+=table('artifacts','Artifact sensitivity using identical fitting and inner-fold settings. Full-stack results are shown; all seven models and class-specific feature prevalences are exported.',
      ['Representation','Features','Recall','FPR','Accuracy'],[[esc(n),r['features'],pc(r['metrics']['X-ERD']['recall']),pc(r['metrics']['X-ERD']['fpr']),pc(r['metrics']['X-ERD']['accuracy'])] for n,r in ab.items()])
    text+=r'PE-stub strings, generic strings and DLL loads can reflect collection or extraction artifacts. We report class-specific prevalence and retrain without strings, then without strings plus ole32/uxtheme DLL indicators. These checks do not prove artifact independence. Full-run summaries and discriminative OS footprints are not evidence of pre-encryption staging.'+'\n'
    text+=r'\subsection*{ATT\&CK suggestions and evaluation boundary}'+'\n'
    mp=j['mapping'];text+=f'The versioned dictionary maps{mp["mapped_features"]}/483 features({pc(mp["mapped_features"]/483)}\\%) to{len(mp["distinct_techniques"])} distinct candidate technique IDs. Among the{len(mp["attribution_sample_ids"])} detected ransomware records, the mean fraction of active positive stack-logit attribution assigned to mapped features is{pc(mp["mass_mean"])}\\%(median{pc(mp["mass_median"])}\\%). The denominator is active positive attribution mass, rather than all signed or absolute contributions. This is coverage, not mapping accuracy.\n'
    text+=r'''Rules separate exact sandbox signatures from weak primitives using High/Medium/Low specificity tiers. These are qualitative evidence scores, not calibrated probabilities. Multiple feature rules may suggest the same technique. RWX allocation alone remains an unmapped footprint, because an APC queue or remote-injection context is not established. LdrGetProcedureAddress alone is a low-specificity Native API observation; obfuscated dynamic resolution would need additional evidence before assertingT1027.007. Generic DLL loads do not establish malicious technique use. The complete dictionary and rationale are supplied. No93.3\% expert accuracy or APIMatch agreement is claimed. A reproducible stratified30-item blank worksheet(20 mapped,10 unmapped) is provided for independent assessment; it has not been filled by an analyst.
'''
    test=pd.read_csv(ROOT/'data/mlran/MLRan_X_test_RFE.csv');feature_cols=[c for c in test if c not in ['sample_id','sample_type','family_label','type_label']]
    dictionary=json.loads((ROOT/'data/mlran/RFE_selected_feature_names_dic.json').read_text())
    sid=11036;ix=int(np.where(test.sample_id.to_numpy()==sid)[0][0])
    with np.load(OUT/'shap_values.npz') as z:phi=z['stack_logit'][ix]
    active=(test.loc[ix,feature_cols].to_numpy(dtype=float)>0)&(phi>0)
    order=np.argsort(-np.where(active,phi,-np.inf))[:8];case=[]
    for index in order:
        name=dictionary[feature_cols[index]];rule=mp['all_feature_mappings'][name]
        case.append({'feature_id':feature_cols[index],'feature_name':name,'stack_logit_contribution':float(phi[index]),'candidate_technique':rule['technique_id'] or 'Footprint'})
    pd.DataFrame(case).to_csv(OUT/'case_study.csv',index=False)
    text+=f'The original illustrative CryptoLocker case(sample ID11036) is retained. Its full-stack ransomware probability is{pc(preds.loc[preds.sample_id==sid,"X-ERD"].iloc[0])}\\%, interpreted as a model score rather than externally calibrated confidence. Table~\\ref{{tab:case}} reports its leading active positive logit contributions; complete feature names are exported in case\\_study.csv.\n'
    text+=table('case','Retained held-out illustrative specimen11036. Technique IDs are candidate suggestions, not forensic findings.',
      ['Feature ID','Logit contribution','Candidate technique'],[[r['feature_id'],num(r['stack_logit_contribution'],6),r['candidate_technique']] for r in case])
    text+=r'\subsection*{Corruption and targeted feature masking}'+'\n'
    corrupt=pd.read_csv(OUT/'corruption_summary.csv');rows=[]
    for scenario in corrupt.scenario.unique():
        for name in ['LGB','RF+LGB','X-ERD']:
            r=corrupt[(corrupt.scenario==scenario)&(corrupt.model==name)].iloc[0]
            rows.append([scenario,esc(name),pc(r.recall_mean)+r'$\pm$'+pc(r.recall_sd),pc(r.fpr_mean)+r'$\pm$'+pc(r.fpr_sd)])
    text+=table('corruption','Common-mask feature corruption. Mean$\pm$sample SD across ten random seeds; deterministic signature/mimicry scenarios are identical across repeats. Complete base-model and pairwise results are exported.',
      ['Scenario','Model','Recall(\%)','FPR(\%)'],rows)
    prot=read('corruption_protocol');text+=f'The selected representation contains{len(prot["api_indices"])} API indicators,{len(prot["sleep_indices"])} sleep/delay signature indicators and{len(prot["vm_indices"])} anti-VM signature indicators. Zero matching features makes a scenario a no-op and is not evidence of robustness. Per-record dropout uses one shared mask for all models; API masking removes common columns. Mimicry sets ten training-selected benign-associated indicators on ransomware only. These are feature-space stress tests, not validated executable modifications.\n'
    text+=figure('corruption.png','corruption','Recall and FPR under common per-indicator dropout. Error bars are across-mask SD, not confidence intervals for deployment risk.',True)
    text+=figure('targeted.png','targeted','One-shot attribution-guided masking at budgets0,1,3,5,10,20,40. These observed vulnerabilities are not a theoretical upper bound.')
    text+=r'Problem-space constraints may prevent some edits from preserving execution, but strings and DLL-related indicators can also be cheaply altered. No binary rewriting or functionality test was performed, so feasibility is not asserted\cite{pierazzi2020intriguing}.'+'\n'
    text+=r'\subsection*{Raw-report parser, attributions and time prefixes}'+'\n'
    parser=j['parser'];pr=next(r for r in parser['models'] if r['model']=='X-ERD');rho=np.mean([r['spearman'] for r in parser['stability'] if r['spearman'] is not None])
    text+=f'All20 available raw reports are ransomware and were originally in the official training set. We exclude all20 before fitting a separate3,885-record parser-evaluation model. The upstream fixed feature selection included these records; exclusion is from model fitting,not earlier representation construction. This is a conditional reconstruction check rather than a completely untouched holdout. On official and parsed vectors, the full stack detects{pr["official"]["tp"]}/20 and{pr["parsed"]["tp"]}/20,respectively. Benign FPR and ROC-AUC are undefined. The mean feature Jaccard is{num(parser["mean_jaccard"],6)}(minimum{num(min(parser["jaccard_per_sample"]),6)}), probability MAE is{num(pr["probability_mae"],6)}, and mean signed stack-logit attribution Spearman correlation is{num(rho,6)}. Top-five overlap is an intersection fraction, not a Jaccard score. These are small-sample engineering results, not population sensitivity or pipeline reliability.\n'
    text+=table('parser','Micro-aggregated feature reconstruction on the20 excluded reports. Undefined quantities are shown as--. All eight categories are disclosed.',
      ['Category','Feature TP','Feature FP','Feature FN','Precision','Recall'],[[esc(n),r['tp'],r['fp'],r['fn'],num(r['precision'],6),num(r['recall'],6)] for n,r in parser['all_categories'].items()])
    text+=table('parser_detection','Downstream detection on the excluded ransomware-only reports. TN=FP=0 because no benign report is available; apparent precision100\% is not an operational PPV estimate.',
      ['Model','Official TP/FN','Parsed TP/FN','Label agreement','Probability MAE'],[[esc(r['model']),f'{r["official"]["tp"]}/{r["official"]["fn"]}',f'{r["parsed"]["tp"]}/{r["parsed"]["fn"]}',pc(r['label_agreement']),num(r['probability_mae'],6)] for r in parser['models']])
    text+=figure('parser_stability.png','parser','Parser feature agreement and attribution stability on20 reports excluded from the separate training fit.',True)
    api=pd.read_csv(OUT/'api_prefix_predictions.csv');ar=[]
    for sec,g in api.groupby('seconds'):ar.append([sec,len(g),int((g.probability>=.5).sum()),num(g.api_prefix_features.mean(),2)])
    text+=table('prefix','Exploratory timestamped API-prefix experiment with a separately fitted API-only stack. Windows start at the first observed call; untimed full-run summary/string/signature features are excluded.',
      ['Seconds','Ransomware reports','Detected at0.50','Mean active API indicators'],ar)
    text+=r'All20 reports expose nested behavior.processes[*].calls[*].time fields. We use numeric time differences from the first recorded call. Execution offsets need not coincide with payload start. Without encryption-onset labels or benign raw traces, these results cannot establish pre-encryption detection, prevention latency, or early-window FPR.'+'\n'
    text+=r'\subsection*{Neural control and computational timing}'+'\n'
    nn=j['neural_control'];nn_parts=[]
    for label,key in [('accuracy','accuracy'),('recall','recall'),('FPR','fpr'),('AUC','auc')]:
        values=np.array([r[key] for r in nn['runs']])*100
        nn_parts.append(label+' '+num(values.mean())+r'$\pm$'+num(values.std(ddof=1))+r'\%')
    text+='Across five seeds, the feed-forward control achieves '+', '.join(nn_parts)+'. These are mean and sample SD across training seeds, not confidence intervals. The complete per-seed confusion counts,epochs and scores are supplied in neural\\_control.json.\n'
    text+=r'The control has256,128 and64 ReLU hidden units, Adam learning rate0.001, batch128, L2 alpha0.0001, maximum200 epochs and patience15, with a stratified internal20\% validation split and seeds0--4. Feature-index order is not temporal order. Suitable labeled benign/ransomware call sequences were not available; a matched deep sequence comparison remains unfulfilled rather than being approximated by artificial feature ordering. Computational inference latency is reported separately in the release logs; it excludes sandbox execution, parsing, explanations and encryption timing.'+'\n'
    text+=r'''\FloatBarrier
\section*{Discussion}
The retained pipeline is reproducible, but the full three-model stack is not the best default-threshold detector on this benchmark. LightGBM has greater recall; RF+LightGBM reduces FPR with fewer components; and the additional Logistic Regression baseline performs better on the official test split. Matched operating points, calibration metrics and corrected CV inference prevent a conservative threshold shift from being presented as a general stacking gain. Adding XGBoost is retained as an original research configuration, not justified as a necessary operational component by these results.

The corrected logit attribution has an explicit explained output and reconstruction check. Deletion sensitivity and attribution agreement are useful diagnostics, but neither establishes causal explanations. Mapping coverage quantifies how much evidence is translated, not whether techniques were maliciously executed. Missing independent mapping labels preclude a validated threat-intelligence accuracy claim.

The strict temporal analysis improves the original evaluation by fitting selection only on historical data, but its candidate modalities differ from the483-feature benchmark and its environment remains MLRan. It therefore cannot isolate time drift from every representation effect, prove novel-family generalization, or substitute for an external dataset. Sample collection, first-submission dates, family labels and sandbox settings remain potential confounders. CV/threshold estimates on the official fixed representation are conditional on the benchmark's selection and are not fully nested feature-selection estimates.

Twenty excluded raw ransomware reports permit downstream parser tests, but lack benign controls, diverse raw-report coverage and encryption-onset annotations. A supervised sequence-model comparison, independent mapping audit and external-environment validation remain priorities. Repeated use of the public test split during revision is not a fresh prospective trial. No acceptance, enterprise performance or early-prevention guarantee follows from these results.

\section*{Methods}
\subsection*{Data schema and model fitting}
Binary labels are explicitly$type\_label>0$ and checked against$sample\_type$. sample\_id,family\_label and both target metadata columns are excluded from inputs. Input values must be finite binary indicators with declared numeric feature IDs; splits must be disjoint. SHA-256 checksums identify the exact datasets. No class weighting, oversampling, synthetic records or threshold-driven relabeling is used.

We preserve RF100 trees, Gini, unlimited depth, bootstrap, max\_features=sqrt, minimum split2/leaf1; XGBoost100 rounds, depth6, eta0.3, lambda1, binary logistic loss; and LightGBM100 trees,31 leaves, eta0.1. These fixed settings reproduce the supplied configuration; no new test-driven search is claimed. Dependencies and full constructor defaults are locked in the release. The L2 logistic meta-learner usesC=1,lbfgs,max\_iter=1000 and no class weighting\cite{breiman2001random,chen2016xgboost,ke2017lightgbm,wolpert1992stacked}.

Five unshuffled stratified inner folds produce one out-of-fold base probability per training record. Logistic heads fit these OOF probabilities; base learners are then refit on the full training partition. Every outer CV fold repeats this inner process from scratch. Sharing equivalent base fits across pairwise heads saves compute without sharing outer-validation data. The official test set is not used to fit the meta-learner. Additional fixed baselines use liblinear Logistic Regression(C=1,max\_iter=10000) and ExtraTrees100 trees,max\_depth20.

\subsection*{Evaluation and inference}
All models use$p\geq\theta$ for positive classification, including ties. Accuracy, precision, recall,F1,FPR and AUC are calculated from the declared target. Zero-denominator FPR/precision and single-class AUC are undefined, not zero. ECE assigns probability1 to the final equal-width bin and includes probability0 in the first. Stratified percentile bootstrap intervals use2000 resamples preserving each class count.

Repeated outer five-fold CV uses seeds0,1,2,3,4,5,42,123,256,999. For differences$d_i$, corrected inference uses
\begin{equation}t_{NB}=\frac{\bar d}{\sqrt{(1/50+1/4)s_d^2}},\qquad df=49.\end{equation}
The approximation accounts for overlapping training sets\cite{nadeau2003inference,bouckaert2004evaluating}. Holm correction covers four full-stack comparator tests for each endpoint separately. Descriptive Wilcoxon uses zero\_method=wilcox(drops zeros),asymptotic inference, and paired rank-biserial based on nonzero signed ranks; it is not Cliff's delta or Pratt's method. Exact McNemar uses a binomial test with no continuity correction\cite{mcnemar1947}; DeLong covariance uses the correct class-count denominators\cite{delong1988comparing,sun2014fast}.

\subsection*{Parser and temporal isolation}
The parser preserves the upstream exact lowercase-token definitions. DLL tokenization splits at most twice so drive-letter colons remain intact; a full path is not silently replaced by a basename alias. Registry, file, directory and static-string matches require complete lowercased tokens; substring matches are avoided. System command-line,mutex and GUID fields are included. Strings are extracted from the declared report field, not arbitrary JSON descriptions. Timestamped API prefixes exclude untimed aggregate features. The temporal selection pool, MI ranks, RFE support, training/test/gap IDs and derived483-feature CSVs are exported independently from the official benchmark representation.

\subsection*{Attribution and mapping}
TreeExplainer is configured with model\_output=probability and feature\_perturbation=interventional using the same100 randomly sampled training records(seed42) for all base learners. Each base probability and the sigmoid of the weighted stack logit must reconstruct within$10^{-4}$. The deterministic100-record global display sample and all background IDs are recorded. Deletion samples150 benign and150 ransomware records without replacement(seed42), ranks active features toward the true class, evaluates the identical0--20 grid, and normalizes the horizontal axis by20. A fixed random-order control uses the same records. Mean intervals are normal approximations; paired tests are exploratory.

The external JSON mapping schema is the sole runtime/evaluation dictionary. Specificity tiers represent rule evidence, not calibrated probabilities. The generated independent-audit worksheet is blank; supplied annotations must include distinct known features, assessor identity and provenance before agreement can be computed. Software cannot establish assessor independence.

\subsection*{Reproducibility}
All reported runs were executed in a Docker container on the existing WSL environment with Python3.11,NumPy2.4.4,pandas3.0.3,SciPy1.17.1,scikit-learn1.9.0,XGBoost3.2.0,LightGBM4.6.0,SHAP0.51.0 and PyArrow23.0.1. The Python base image is pinned by digest. Source/data hashes, dependency lock, tests, per-record predictions,50 outer-fold observations and execution logs accompany the release. Tests include independent StackingClassifier equivalence, binary-MI reference values, schema exclusion, exact recall targets, undefined FPR, probability-bin endpoints, Windows paths and Parquet ID representations.

\section*{Data Availability}
MLRan data and reports originate from\url{https://github.com/faithfulco/mlran} and\url{https://zenodo.org/records/23403681}. The published dataset article is doi:10.1016/j.jnca.2026.104475. The reproduction script verifies fixed checksums. The strict temporal derived datasets and selection manifests are generated by the release scripts; their modality scope is stated above.

\section*{Code Availability}
The corrected codebase, tests, mapping schema and result logs are supplied with this author-review draft. Public repository upload and DOI-assigning archival deposit have not yet been completed. An actual published version DOI must replace this status before resubmission; no existing archive or DOI is asserted.
'''
    text+=r'\bibliography{references_revision}'+'\n'
    text+=source[source.index(r'\section*{Acknowledgements}'):]
    text=text.replace('All authors reviewed, refined, and approved the final manuscript.',
                      'The author roles above are retained from the supplied submission. All authors must review and approve this revision before submission.')
    text=text.replace('No93.3\\% expert accuracy or APIMatch agreement is claimed.','Mapping accuracy has not been independently established; no external taxonomic agreement estimate is available.')
    text=text.replace('No acceptance, enterprise performance or early-prevention guarantee follows from these results.','These results do not establish enterprise performance or early prevention.')
    from build_extended_report import integrate_extended
    text=integrate_extended(text,ROOT)
    from build_remediation_report import integrate
    text=integrate(text,ROOT)
    preamble,text=text.split(r'\begin{abstract}',1)
    text=r'\begin{abstract}'+text
    # Space prose around numbers while protecting citations, URLs, labels and
    # technique identifiers. Those are machine-readable identifiers, not prose.
    protected=[]
    def hold(match):
        value=match.group(0)
        needs_space=value.startswith((r'\url{',r'\texttt{')) or re.fullmatch(r'T\d{4}(?:\.\d{3})?',value)
        before=' ' if needs_space and match.start() and text[match.start()-1].isalnum() else ''
        protected.append(value);return before+chr(0xE000+len(protected)-1)
    text=re.sub(r'(?<![\d.])(\d+(?:\.\d+)?)e([+-]\d+)',lambda m:'$'+m[1]+r'\times10^{'+str(int(m[2]))+'}$',text)
    def hold_math(match):
        before=' ' if match.start() and text[match.start()-1].isalnum() else ''
        after=' ' if match.end()<len(text) and text[match.end()].isalnum() else ''
        return before+hold(match)+after
    text=re.sub(r'\$[^$]+\$|\\begin\{equation\}.*?\\end\{equation\}',hold_math,text,flags=re.S)
    text=re.sub(r'\\texttt\{[^}]*\}',hold,text)
    text=re.sub(r'\\(?:cite|label|ref|url|includegraphics|bibliography|graphicspath)(?:\[[^\]]*\])?\{[^}]*\}|T\d{4}(?:\.\d{3})?|\d+(?:\.\d+)?e[-+]\d+|\b[a-fA-F0-9]{32,64}\b|10\.\d{4,9}/[-._;()/:A-Za-z0-9]+|\b(?:F1|L1|L2|MD5|SHA256|ole32)\b|\b[A-Za-z0-9_]+\.dll\b',hold,text)
    text=re.sub(r'(?<=[A-Za-z])(?=\d)', ' ',text)
    text=re.sub(r'(?<=\d)(?=[A-Za-z])', ' ',text)
    text=re.sub(r'(?<=[,;:])(?=[A-Za-z\uE000-\uF8FF])', ' ',text)
    # Preserve thousands separators and decimal notation while spacing prose.
    text=re.sub(r'(?<=[A-Za-z%)])([,;:])(?=\d)',r'\1 ',text)
    text=re.sub(r'(?<=[A-Za-z0-9%)])(?=\()', ' ',text)
    text=re.sub(r'(?<=interval)(?=\[)|(?<=CI)(?=\[)', ' ',text)
    for i,value in enumerate(protected):text=text.replace(chr(0xE000+i),value)
    text=text.replace('usesC=', 'uses C=').replace('interval is[', 'interval is [')
    text=preamble+text
    # Preserve complete feature tokens, including Windows paths and GUIDs, in
    # the standalone supplement. Insert after prose spacing to avoid changing IDs.
    features=pd.read_csv(OUT/'global_shap.csv').head(14)
    feature_rows=[]
    def literal(s):
        return ''.join({'\\':r'\textbackslash{}','{':r'\{','}':r'\}','_':r'\_',
                        '&':r'\&','%':r'\%','#':r'\#','$':r'\$'}.get(c,c) for c in str(s))
    for _,row in features.iterrows():
        lines=textwrap.wrap(row['feature_name'],width=66,break_long_words=True,break_on_hyphens=False)
        name=r'\shortstack[l]{'+r'\\'.join(r'\texttt{'+literal(x)+'}' for x in lines)+'}'
        feature_rows.append([str(row['feature_id']),name,num(row['mean_abs_stack_logit_shap'],6)])
    extra=table('global_feature_names','Complete feature names for the 14 global-attribution bars, in descending importance. Path and identifier characters are preserved; line breaks are typographic only.',
                ['Feature ID','Complete feature token','Mean absolute logit SHAP'],feature_rows)
    extra=r'Complete feature tokens are listed in Table~\ref{tab:global_feature_names}.'+'\n'+extra
    at=text.index(r'\label{fig:global}');end=text.index(r'\end{figure*}',at)+len(r'\end{figure*}')
    text=text[:end]+'\n'+extra+text[end:]
    (MAN/'main_scirep_complete.tex').write_text(text,encoding='utf-8')
    from package_manuscripts import partition
    text=partition(text,ROOT)
    (MAN/'main_scirep_revised.tex').write_text(text,encoding='utf-8')
    print('Wrote revised manuscript source from',len(required),'verified result groups')

if __name__=='__main__':main()
