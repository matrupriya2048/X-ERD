from pathlib import Path
import argparse,json,re,shutil
from build_documents import esc
HERE=Path(__file__).resolve().parent

def evidence():
    p=json.loads((HERE/'closure_results/primary_pair_results.json').read_text())
    impact=json.loads((HERE/'closure_results/impact_proxy.json').read_text())
    windows=json.loads((HERE/'closure_results/sequence_window_controls.json').read_text())
    return p,impact,windows

def paragraphs():
    p,impact,windows=evidence()
    primary=('We adopt the reviewer-requested RF+LightGBM logistic stack as the primary source-benchmark configuration, retaining the three-model stack and all pairwise configurations as comparative analyses. This is an explicit revision decision after the reported benchmark comparisons, not a prospectively validated selection. At threshold 0.50, the primary pair yields 495 TN, 15 FP, 10 FN and 455 TP: 97.44% accuracy, 97.85% recall and 2.94% FPR. Its common-background probability-scale base SHAP contributions are recomposed using its own two fitted logistic coefficients; the maximum probability reconstruction error is '+f'{p["maximum_probability_reconstruction_error"]:.2e}'+'. The primary inference artifact invokes only RF and LightGBM. The separately frozen three-model adaptation remains an extension study with its historical selection protocol; its favorable same-collection result is not attributed to the unadapted primary pair. Supplementary Table S23 gives the primary pair across available cohorts.')
    reliance=(f'For the primary pair, a transparent lexical screen identifies five encryption/decryption or cryptographic-library indicators among the 483 selected features. They are active in {impact["detected_with_active_proxy_n"]}/{impact["ransomware_detected_n"]} detected ransomware records ({100*impact["detected_with_active_proxy_fraction"]:.2f}%). Their mean share of positive active logit attribution is {100*impact["positive_active_logit_mass_mean"]:.4f}%; the median is zero. Zeroing these five indicators loses no ransomware detections and preserves the confusion counts. These are untimed library/string indicators, not verified post-encryption events. This sensitivity analysis answers feature reliance only; it cannot establish when payload encryption started or rule out reliance on other impact-related behavior.')
    prefix=('We additionally report benign false alarms and ransomware detections for each 32/128/512-call and shuffled-512 window from the existing held-out LSTM/CNN predictions, retaining all three neural seeds. Each seed evaluates the same 20 benign and 60 ransomware traces, so seed results are dependent and are not pooled as additional samples. Supplementary Table S24 reports the ranges across seeds. The substantial early-window false alarms prevent a deployment or pre-encryption-intervention claim from these traces. Call counts are not elapsed time and no verified onset is supplied.')
    return primary,reliance,prefix

def papers():
    primary,reliance,prefix=paragraphs();p,impact,windows=evidence()
    path=HERE/'manuscript/main_scirep_revised.tex';text=path.read_text(encoding='utf-8')
    abstract=r'''X-ERD integrates behavioral parsing, logistic stacking, additive attribution and candidate MITRE ATT\&CK mapping. Following reviewer assessment, RF+LightGBM is the primary benchmark configuration, yielding 97.44\% accuracy, 97.85\% recall and 2.94\% false positive rate (FPR). The original three-model stack remains a comparison at 97.23\% accuracy and 3.33\% FPR; no necessary third-learner benefit is established. A separately selected three-model adaptation uses 182 external benign development executions and is frozen before new target scoring. It gives 6/362 false alarms (1.66\%) on newly selected same-collection executions and 55/58 detections on separately published, publisher-labeled ransomware reports. These extension results are distinct from the primary pair. The pair's own additive contributions reconstruct its output; masking five cryptographic-library/string indicators loses no benchmark detections. Corrected inference, temporal feature selection, published-method adaptations, raw-report controls and genuine LSTM/CNN comparisons qualify the results. Additional mixed-file testing and sequence-window benign errors expose transport limitations. ATT\&CK suggestions remain a proof of concept, and verified encryption-onset annotations are unavailable. The contribution is a reproducible explainable-detection pipeline with explicit model and dataset boundaries.'''
    text=re.sub(r'\\begin\{abstract\}.*?\\end\{abstract\}',lambda _:r'\begin{abstract}'+'\n'+abstract+'\n'+r'\end{abstract}',text,count=1,flags=re.S)
    text=text.replace(r'\section*{Results}',r'\section*{Results}'+'\n'+r'\subsection*{Primary configuration and retained comparisons}'+'\n'+esc(primary)+'\n\n',1)
    text=text.replace('do not replace the frozen primary configuration after test inspection','do not alter the historically frozen three-model adaptation after test inspection')
    text=text.replace(r'\section*{Discussion}',r'\subsection*{Cryptographic-indicator reliance and sequence-window controls}'+'\n'+esc(reliance)+'\n\n'+esc(prefix)+' The primary-pair global attribution is shown in Supplementary Figure~S15.\n\n'+r'\section*{Discussion}',1)
    text=text.replace('We retain the original X-ERD architecture and all pairwise ablations,','We retain the original X-ERD framework and all pairwise ablations, designate RF+LightGBM as the primary configuration,',1)
    text=text.replace('Retained architecture and optional feature-passthrough extension.','Retained comparative architecture and optional feature-passthrough extension. The primary pair uses the RF and LightGBM branches; XGBoost belongs to the retained three-model comparison.',1)
    path.write_text(text,encoding='utf-8')
    sup=HERE/'manuscript/supplementary_information.tex';text=sup.read_text(encoding='utf-8')
    extra=r'\clearpage\section*{Primary pair and additional reviewer analyses}'+'\n'+esc(primary)+'\n\n'
    extra+=r'\begin{table}[H]\centering\small\caption{Primary-pair results at 0.50. Cohorts differ and must not be pooled. The unadapted primary pair remains separate from the three-model adaptation.}\label{tab:primary_pair}\begin{tabular}{lrrrr}\toprule Setting & TN & FP & FN & TP\\\midrule'+'\n'
    names={'official':'Official benchmark','strict_temporal':'Strict temporal','external_original_182':'Old external benign','new_same_collection_362_original':'New same-collection benign','zirari_58_original':'Separate raw ransomware','external_shared_api':'Shared-API transfer'}
    for key,title in names.items():
        m=p['results'][key]['metrics'];extra+=title+' & '+' & '.join(str(m[k]) for k in ['tn','fp','fn','tp'])+r'\\'+'\n'
    extra+=r'\bottomrule\end{tabular}\end{table}'+'\n\n'+esc(reliance)+'\n\n'+esc(prefix)+'\n\n'
    extra+=r'\begin{table}[H]\centering\small\caption{Sequence-window controls, ranges over three seeds on the same 20 benign and 60 ransomware traces. Seeds are dependent repeated fits.}\label{tab:prefix_controls}\begin{tabular}{llrr}\toprule Model & Calls & Benign FP/20 & Ransomware TP/60\\\midrule'+'\n'
    for model in ['CNN','LSTM']:
        for window in ['32','128','512','shuffled512']:
            rows=[r for r in windows['rows'] if r['model']==model and r['calls']==window]
            fp=[r['benign_fp'] for r in rows];tp=[r['ransomware_tp'] for r in rows]
            extra+=f'{model} & {window} & {min(fp)}--{max(fp)} & {min(tp)}--{max(tp)}'+r'\\'+'\n'
    extra+=r'\bottomrule\end{tabular}\end{table}'+'\n'
    shutil.copy2(HERE/'closure_results/primary_pair_global_attribution.png',HERE/'manuscript/figures/primary_pair_global_attribution.png')
    extra+=r'\begin{figure}[H]\centering\includegraphics[width=0.95\textwidth]{primary_pair_global_attribution.png}\caption{Primary RF+LightGBM global attribution using its own logistic coefficients and the same fixed 100-record sample. Fourteen features are sorted by magnitude; full feature names and all 483 values are exported in the accompanying CSV. Earlier attribution figures concern the retained three-model comparison.}\label{fig:primary_pair_global}\end{figure}'+'\n'
    text=text.replace(r'\end{document}',extra+r'\end{document}',1);sup.write_text(text,encoding='utf-8')

def response():
    primary,reliance,prefix=paragraphs();path=HERE/'docs/response_letter.tex';text=path.read_text(encoding='utf-8')
    updates={'R1.2':primary,'R2.5':primary,'R3.2':primary,'R1.16':reliance+' '+prefix,'R4.5':prefix+' '+reliance}
    for key,paragraph in updates.items():
        start=text.index(r'\subsection*{'+key+' ');end=text.find(r'\subsection*{',start+1)
        if end<0:end=text.index(r'\end{document}',start)
        block=text[start:end];pos=block.index(r'\textit{Location:}')
        block=block[:pos]+r'\textbf{Completed revision analysis.} '+esc(paragraph)+'\n\n'+block[pos:]
        text=text[:start]+block+text[end:]
    text=text.replace('A bounded feature-passthrough extension is reported separately from the original configuration.','RF+LightGBM is now the primary benchmark configuration; the three-model stack and its bounded feature-passthrough adaptation are reported as comparative extensions. The original component and extension results retain their original model identities.',1)
    path.write_text(text,encoding='utf-8')
    for file in ['reviewer_revision_status.json','response_letter_index.json']:
        path=HERE/'docs'/file;data=json.loads(path.read_text())
        for row in data['items']:
            key=row['id']
            if key in ['R1.2','R2.5','R3.2']:row['status']='Implemented: reviewer-requested primary RF+LightGBM pair'
            if key in updates:
                row['closure_evidence']=['closure_results/primary_pair_results.json','closure_results/impact_proxy.json','closure_results/sequence_window_controls.json']
                if 'response' in row:row['response']+=' '+updates[key]
            if key=='R1.16':row['status']='Reliance analysis completed; verified onset remains unavailable'
        path.write_text(json.dumps(data,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['papers','response']);a=p.parse_args();papers() if a.stage=='papers' else response()
