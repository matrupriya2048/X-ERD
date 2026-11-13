from pathlib import Path
import argparse,json,re,shutil,unicodedata
from pypdf import PdfReader
from build_documents import esc
HERE=Path(__file__).resolve().parent

def paragraphs():
    r=json.loads((HERE/'multicollection/results.json').read_text())['metrics'];v=json.loads((HERE/'multicollection/verification.json').read_text());ex=json.loads((HERE/'multicollection/explanations.json').read_text())
    assert v['status']=='passed' and v['fresh_retraining_identical_model_sha256']
    text=('The observed CAPE transport failure motivated a separate, fixed-parameter multi-collection adaptation of the primary RF+LightGBM pair. All 917 CAPE targets had already been inspected with earlier frozen models. We therefore explicitly treat this as retrospective adaptation. A seed-20261113 exact-vector-group split assigns 634 targets to development and 283 to an internal holdout, with no shared target/vector group or fitting-vector match. The 3905 source-training records and 182 previously disclosed Nunes development records are retained. Five-fold group OOF base probabilities train a logistic head with raw-feature passthrough; RF100, LightGBM100, C10, external weight3 and threshold0.50 inherit the earlier development configuration and are fixed before fitting. No search, threshold reselection or parser exception uses this holdout. Raw-ransomware evaluation rows are not fitted. The resulting pair produces 1/283 CAPE alerts (0.35%; exact 95% interval 0.009-1.953%), versus 283/283 for each original pair/full stack and 142/283 for the previous three-model adaptation on those same targets. It gives 4/362 alerts (1.10%) on the earlier Nunes holdout and detects 55/58 separate publisher-labeled ransomware targets. The reused official test gives 501 TN,9 FP,6 FN,459 TP (98.46% accuracy,98.71% recall,1.76% FPR); this is a regression check. All preceding adverse results remain reported. These endpoints are not pooled into one accuracy estimate.')
    quality=('Publisher labels remain a limitation: among the 917 publisher-benign reports, CAPE itself records 597 Clean,270 Suspicious and50 Malicious statuses. These contradictory automated statuses are disclosed, not used to relabel or filter the cohort. The retrospective holdout contains 172 Clean,96 Suspicious and15 Malicious statuses; its sole alert is in the Clean subgroup (1/172). This is a descriptive status sensitivity, not independently adjudicated benign ground truth. Fresh retraining reproduces the exact model SHA-256. The final pair explanation includes both base probability SHAP contributions and the centered raw linear terms; across975 official and283 CAPE-holdout records the maximum probability reconstruction error is '+f'{ex["max_probability_reconstruction_error"]:.2e}'+'. Positive mapped attribution mass is model reliance, not technique accuracy. Supplementary Table S26 and Figure S16 report this final variant; earlier figures and tables retain their original model identities.')
    quality+=' The source-only primary pair detects56/58 raw-ransomware targets; the final adaptation detects55/58, so it has one additional miss and recall non-inferiority is not established. Final RF, LightGBM and pair Brier/ECE metrics and reliability diagrams are provided in Supplementary Table S27 and Figure S17 on the reused975-record regression test; they do not establish deployment calibration.'
    def clean(s):
        s=re.sub(r'(?<=[A-Za-z])(?=\d)|(?<=\d)(?=[A-Za-z])',' ',s)
        s=re.sub(r'\bS (\d+)\b',r'S\1',s)
        return re.sub(r'(\d) e([+-]\d+)',r'\1e\2',s)
    return clean(text),clean(quality)

def run(stage):
    text,quality=paragraphs();r=json.loads((HERE/'multicollection/results.json').read_text())['metrics']
    if stage=='papers':
        path=HERE/'manuscript/main_scirep_revised.tex';main=path.read_text(encoding='utf-8')
        abstract=r'''X-ERD integrates behavioral parsing, logistic stacking, additive attribution and candidate MITRE ATT\&CK mapping. RF+LightGBM is the primary benchmark configuration, yielding97.44\% accuracy,97.85\% recall and2.94\% false positive rate(FPR); the original three-model stack remains a comparison. Larger raw-report controls expose severe transport failure, including complete false alerting by the unadapted pair on917 publisher-benign CAPE targets after schema translation. A fixed-parameter multi-collection adaptation retains483 indicators and fits on3905 source records,182 previously disclosed benign development records and634 CAPE development targets. It yields1/283 false alerts(0.35\%) on a retrospectively defined, vector-group-isolated CAPE holdout,4/362(1.10\%) on an earlier benign holdout and55/58 detections on separate publisher-labeled ransomware reports. The reused official test gives98.46\% accuracy and98.71\% recall. Contradictory sandbox status labels and collection dependence qualify these estimates. Common-background probability SHAP and exact raw linear contributions reconstruct the final pair output. Corrected inference, temporal feature selection, published-method adaptations and genuine LSTM/CNN comparisons provide additional evaluation. ATT\&CK suggestions remain a proof of concept; verified encryption-onset annotations are unavailable. The contribution is a reproducible explainable-detection pipeline with measured, explicitly scoped adaptation.'''
        # Preserve spaces in final prose, including number boundaries.
        abstract=re.sub(r'(?<=[A-Za-z])(?=\d)|(?<=\d)(?=[A-Za-z])',' ',abstract)
        abstract=abstract.replace('rate(FPR)','rate (FPR)').replace('alerts(0.35','alerts (0.35').replace('4/362(1.10','4/362 (1.10')
        main=re.sub(r'\\begin\{abstract\}.*?\\end\{abstract\}',lambda _:r'\begin{abstract}'+'\n'+abstract+'\n'+r'\end{abstract}',main,count=1,flags=re.S)
        short=('A separately declared multi-collection RF+LightGBM adaptation addresses the CAPE failure using634 CAPE development targets and a283-target retrospective vector-group holdout. At the fixed0.50 threshold it gives1/283 CAPE alerts(0.35%),4/362 earlier Nunes alerts(1.10%) and55/58 separate raw-ransomware detections. Reused official-test accuracy is98.46%. The old failures remain disclosed; the holdout is retrospective and publisher labels have contradictory sandbox statuses. Full fitting details, paired old-model counts, uncertainty and status sensitivities appear in Methods and Supplementary Table S26. Final-model attribution is shown in Supplementary Figure S16.')
        short=re.sub(r'(?<=[A-Za-z])(?=\d)|(?<=\d)(?=[A-Za-z])',' ',short)
        main=main.replace(r'\section*{Discussion}',r'\subsection*{Multi-collection primary-pair adaptation}'+'\n'+esc(short)+'\n\n'+r'\section*{Discussion}',1)
        main=main.replace(r'\subsection*{Reproducibility}',r'\subsection*{Retrospective multi-collection adaptation and final attribution}'+'\n'+esc(text)+'\n\n'+esc(quality)+'\n\n'+r'\subsection*{Reproducibility}',1)
        # Relocate long newly added protocol explanations to Methods while retaining
        # their measured results in the main Results and the complete supplement.
        for heading,summary in [
          ('Primary configuration and retained comparisons','RF+LightGBM is the primary source-benchmark configuration at threshold 0.50; the full stack remains a comparative analysis. Its975-record results are495 TN,15 FP,10 FN and455 TP. The primary inference path invokes two base learners. The choice implements the reviewer-requested simpler alternative and is a disclosed revision decision.'),
          ('Independent EN2025 benign PE reports','All918 preselected non-pilot EN2025 reports were audited;917 unique eligible PE targets remained. The original parser omits CAPE nested strings and typed enhanced events. A post-hoc schema adapter exposes severe transport failure:917/917 alerts for the original primary pair and full stack, versus500/917 for the previous three-model adaptation. Both schema versions, provenance limitations and all exclusions appear in Supplementary Table S25. The separate repair below reclassifies a declared subset as development data.')]:
            pattern=r'\\subsection\*\{'+re.escape(heading)+r'\}\n(.*?)(?=\\subsection\*|\\section\*)'
            found=re.search(pattern,main,re.S)
            if found:
                original=found.group(1);replacement=r'\subsection*{'+heading+'}'+chr(10)+esc(summary)+'\n\n'
                main=main[:found.start()]+replacement+main[found.end():]
                main=main.replace(r'\subsection*{Reproducibility}',r'\subsection*{'+heading+' protocol details}'+chr(10)+original+'\n'+r'\subsection*{Reproducibility}',1)
        transfers=[]
        summaries={
          'Benign-domain adaptation and frozen raw holdouts':'The earlier three-model extension used 182 external benign development executions and a new same-collection holdout. It reduced the previous 172/362 alerts to 6/362 (1.66%) and detected 55/58 separately published ransomware targets, one fewer than the previous passthrough variant. The original adverse 182-report results remain in Table 7. This is historical same-collection adaptation, not the new primary pair. Complete previous-variant results and provenance limitations are retained in Supplementary Figure S14 and Tables S19, S20 and S21.',
          'Cryptographic-indicator reliance and sequence-window controls':'Five cryptographic-library/string indicators are active in 19/455 primary-pair detections (4.18%); masking them loses no detections. They are untimed proxies. Sequence-window false alarms, all seeds and complete reliance statistics appear in Supplementary Table S24 and its accompanying text. Primary-pair attribution appears in Supplementary Figure S15.',
          'Additional collection with score-derived benign labels':'The separate score-labeled mixed-file cohort gives 6/142 alerts for the earlier adapted full stack and 5/142 for each earlier configuration. Only 11 targets are PE executables, with 2/11 alerts for every model. Full sampling, uncertainty and modality limitations are in Supplementary Table S22. This exploratory test does not establish verified-clean application performance.'}
        for heading,summary in summaries.items():
            match=re.search(r'\\subsection\*\{'+re.escape(heading)+r'\}\n(.*?)(?=\\subsection\*|\\section\*)',main,re.S)
            if match:
                transfers.append((heading,match.group(1).replace(r'\FloatBarrier','')))
                replacement=r'\subsection*{'+heading+'}'+chr(10)+esc(summary)+'\n\n'
                main=main[:match.start()]+replacement+main[match.end():]
        path.write_text(main,encoding='utf-8')
        sup=HERE/'manuscript/supplementary_information.tex';s=sup.read_text(encoding='utf-8')
        extra=''
        for heading,body in transfers:extra+=r'\section*{'+heading+' -- complete detail}'+chr(10)+body+'\n\n'
        extra+=r'\clearpage\section*{Multi-collection RF+LightGBM adaptation}'+'\n'+esc(text)+'\n\n'+esc(quality)+'\n\n'
        extra+=r'\begin{table}[H]\centering\small\caption{Final multi-collection pair at 0.50. CAPE holdout is retrospective; other endpoints were previously evaluated. Different cohorts are not pooled.}\label{tab:multicollection}\begin{tabular}{lrrrr}\toprule Setting & TN & FP & FN & TP\\\midrule'+'\n'
        for name,key in [('CAPE retrospective holdout','en_retrospective_holdout'),('Reused official test','official_reused_regression'),('Earlier Nunes holdout','prior_nunes_holdout'),('Separate raw ransomware','prior_zirari_ransomware')]:
            m=r[key];extra+=name+' & '+' & '.join(str(m[k]) for k in ['tn','fp','fn','tp'])+r'\\'+'\n'
        extra+=r'\bottomrule\end{tabular}\end{table}'+'\n'
        shutil.copy2(HERE/'multicollection/global_attribution.png',HERE/'manuscript/figures/multicollection_attribution.png')
        extra+=r'\begin{figure}[H]\centering\includegraphics[width=0.95\textwidth]{multicollection_attribution.png}\caption{Final multi-collection RF+LightGBM pair: fourteen largest mean absolute logit contributions on the fixed 100-record source-test sample. Full names and all 483 values are exported. Both base SHAP and raw linear terms belong to this fitted variant.}\label{fig:multicollection_attribution}\end{figure}'+'\n'
        calibration=json.loads((HERE/'multicollection/calibration.json').read_text())
        extra+=r'\clearpage\section*{Final-pair calibration regression check}\begin{table}[H]\centering\small\caption{Final fitted RF, LightGBM and pair on the reused 975-record official test. ECE uses ten equal-width bins.}\label{tab:final_calibration}\begin{tabular}{lrrrr}\toprule Model & Brier & ECE & Recall (\%) & FPR (\%)\\\midrule'+'\n'
        for name,m in calibration['metrics'].items():extra+=f'{name} & {m["brier"]:.5f} & {m["ece"]:.5f} & {100*m["recall"]:.2f} & {100*m["fpr"]:.2f}'+r'\\'+'\n'
        extra+=r'\bottomrule\end{tabular}\end{table}'+'\n'
        shutil.copy2(HERE/'multicollection/calibration.png',HERE/'manuscript/figures/final_pair_calibration.png')
        extra+=r'\begin{figure}[H]\centering\includegraphics[width=0.98\textwidth]{final_pair_calibration.png}\caption{Final-model reliability and probability distribution on the reused official regression test. This is not a new independent calibration assessment or a deployment prevalence estimate.}\label{fig:final_pair_calibration}\end{figure}'+'\n'
        sup.write_text(s.replace(r'\end{document}',extra+r'\end{document}',1),encoding='utf-8')
    else:
        path=HERE/'docs/response_letter.tex';response=path.read_text(encoding='utf-8')
        start=response.index('Dear Editor and Reviewers,')
        end=response.index(r'\section*{Editor}',start)
        intro=('Dear Editor and Reviewers,\n\nWe thank you for the detailed assessment. The revision preserves the483-indicator ingestion, stacking, attribution and candidate ATT&CK framework. Following Reviewer1.2, RF+LightGBM is the primary source-benchmark configuration; the three-model stack remains a comparison. The source-only pair gives97.44% accuracy,97.85% recall and2.94% FPR.\n\nThe expanded raw-report experiments exposed genuine transport failures, including complete alerting on the917-target EN2025 cohort after explicit CAPE schema translation. We retain those adverse results and report a separate repair: a fixed-parameter multi-collection pair trained on3905 source records,182 earlier benign development records and634 CAPE development records. On a vector-group-isolated283-target retrospective CAPE holdout it produces1 false alert(0.35%; exact95% interval0.009-1.953%). It also produces4/362 alerts on the earlier Nunes holdout and detects55/58 separately published ransomware targets. The reused official test yields98.46% accuracy,98.71% recall and1.76% FPR. These endpoints are not pooled, and the CAPE holdout is explicitly retrospective because the collection had already been inspected with earlier models. Publisher labels and conflicting sandbox statuses are disclosed.\n\nThe final model reproduces exactly under fresh retraining. Its own base probability SHAP contributions and centered raw linear terms reconstruct all975 official and283 CAPE-holdout outputs with maximum error1.01e-08. Corrected statistical inference, temporal feature selection, matched operating points, published-method adaptations, genuine LSTM/CNN comparisons and per-window benign errors provide the remaining experimental evidence. Independent human ATT&CK adjudication and verified encryption-onset labels remain unavailable; mapping and early-intervention claims are restricted accordingly.\n\nAll69 tracked editor/reviewer items are answered below, with locations checked against the accompanying manuscript. Earlier local evidence paths refer to the supplied version0.3 codebase; final pair, CAPE schema, adaptation and verification artifacts are in the accompanying follow-up package. Public GitHub/Zenodo publication remains the final author-controlled release step; no unissued DOI is asserted.\n\n')
        intro=re.sub(r'(?<=[A-Za-z])(?=\d)|(?<=\d)(?=[A-Za-z])',' ',intro)
        intro=re.sub(r'(\d) e([+-]\d+)',r'\1e\2',intro)
        response=response[:start]+esc(intro)+response[end:]
        pages=[p.extract_text() or '' for p in PdfReader(HERE/'manuscript/main_scirep_revised.pdf').pages]
        normalize=lambda s:re.sub('[^a-z0-9]','',unicodedata.normalize('NFKD',s).lower())
        locations=[]
        for heading in ['Multi-collection primary-pair adaptation','Retrospective multi-collection adaptation and final attribution']:
            hits=[]
            for i,page in enumerate(pages):
                lines=page.splitlines()
                if any(normalize(''.join(lines[j:j+n]))==normalize(heading) for j in range(len(lines)) for n in [1,2,3,4]):hits.append(i+1)
            assert hits,heading
            locations.append(heading+' (p. '+', '.join(map(str,hits))+')')
        ids=['R1.2','R1.3','R1.m2','R1.g3','R1.13','R2.2','R2.5','R2.8','R2.12','R3.1','R3.2','R3.4','R4.1']
        short=('The final multi-collection RF+LightGBM adaptation now uses a declared634/283 CAPE development/retrospective-holdout split. It gives1/283 CAPE alerts,4/362 earlier Nunes alerts and55/58 separate raw-ransomware detections; reused official-test accuracy is98.46%. All prior failures and contradictory sandbox statuses are disclosed. Response R1.3, Methods and Supplementary Table S26 give the complete protocol. Its own base-SHAP and raw-linear terms reconstruct the fitted logit; this is distinct from the earlier source-only and three-model variants.')
        short=re.sub(r'(?<=[A-Za-z])(?=\d)|(?<=\d)(?=[A-Za-z])',' ',short)
        short+=' The source-only primary pair detects 56/58 raw-ransomware targets, versus 55/58 for this adaptation; no recall non-inferiority is claimed. Its own calibration metrics and reliability plots appear in Supplementary Table S27 and Figure S17.'
        for key in ids:
            start=response.index(r'\subsection*{'+key+' ');end=response.find(r'\subsection*{',start+1)
            if end<0:end=response.index(r'\end{document}',start)
            block=response[start:end];pos=block.index(r'\textit{Location:}')
            block=block[:pos]+r'\textbf{Verified final-pair adaptation.} '+esc((text+' '+quality) if key=='R1.3' else short)+'\n\n'+r'\textit{Final-pair locations:} '+esc('; '.join(locations))+'; Supplementary Table S26 and Figure S16.\n\n'+block[pos:]
            response=response[:start]+block+response[end:]
        path.write_text(response,encoding='utf-8')
        for name in ['response_letter_index.json','reviewer_revision_status.json']:
            path=HERE/'docs'/name;data=json.loads(path.read_text())
            for row in data['items']:
                if row['id'] in ids:
                    row.setdefault('locations',[]).extend(locations)
                    row['final_pair_evidence']=['multicollection/protocol.json','multicollection/results.json','multicollection/verification.json','multicollection/explanations.json']
            path.write_text(json.dumps(data,indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['papers','response']);a=p.parse_args();run(a.stage)
