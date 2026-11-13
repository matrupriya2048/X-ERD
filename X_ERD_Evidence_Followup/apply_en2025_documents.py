from pathlib import Path
import argparse,json,re
from build_documents import esc
HERE=Path(__file__).resolve().parent

def paragraph():
    r=json.loads((HERE/'closure_results/en2025_results.json').read_text());p=r['models']['primary_RF+LGB'];a=r['models']['adapted_three'];o=r['models']['original_three'];lo,hi=p['ci95']
    return (f'We additionally evaluated the frozen primary pair on the independent EN2025 benign-report collection. The published archive checksum was verified. One schema-inspection pilot was excluded before scoring; all remaining 918 reports were inventoried with no outcome-based replacement. Prespecified PE, telemetry, duplicate and prior-target-overlap checks left {p["n"]} eligible unique PE targets. The primary pair produced {p["alert_n"]}/{p["n"]} false alerts ({100*p["fpr"]:.2f}%; exact binomial 95% interval {100*lo:.2f}-{100*hi:.2f}%). On these identical targets, the original three-model stack gave {o["alert_n"]} alerts ({100*o["fpr"]:.2f}%) and the previously frozen adapted three-model stack gave {a["alert_n"]} ({100*a["fpr"]:.2f}%). No model, parser rule, feature selection, threshold or calibration was fitted to this collection. The archive uses CAPE 2.4 reports and publisher benign labels, without new human class adjudication. Strings are available in {r["strings_available_n"]}/{p["n"]} reports; missing modalities and collector differences preclude a matched parser-fidelity estimate. There are {r["distinct_vectors"]} distinct parsed vectors, {r["zero_active_vectors"]} zero-active vectors and {r["source_fit_vector_matches"]} matches to primary fitting vectors. Available target hashes do not overlap prior fitting/evaluation inventories, but complete overlap verification against earlier hashless sequence logs is unavailable. This tests additional-source benign specificity, not ransomware recall or enterprise deployment. Supplementary Table S25 and the per-report audit retain all exclusions and model identities.')

def run(stage):
    text=paragraph();r=json.loads((HERE/'closure_results/en2025_results.json').read_text())
    text=text.replace('No model, parser rule, feature selection, threshold or calibration was fitted to this collection.','For that initial frozen-model assessment, no model, parser rule, feature selection, threshold or calibration was fitted to this collection.')
    translated=json.loads((HERE/'closure_results/en2025_translated_results.json').read_text())
    t=translated['models'];primary=t['primary_RF+LGB']
    schema=(f'A subsequent schema audit identified CAPE static strings under target.file.strings and typed registry/file/library observations under behavior.enhanced. A deterministic adapter translates these recorded fields into the existing parser schema without model fitting or outcome-specific exceptions. This adapter was developed after inspecting the untranslated evaluation, so the following is a post-hoc paired schema analysis on the same targets. It changes {translated["changed_vectors"]} vectors and recovers static-string availability for {translated["strings_available_n"]} targets. With the adapter, the primary pair gives {primary["alert_n"]}/{primary["n"]} alerts ({100*primary["fpr"]:.2f}%), the original three-model stack {t["original_three"]["alert_n"]} ({100*t["original_three"]["fpr"]:.2f}%), and the adapted three-model stack {t["adapted_three"]["alert_n"]} ({100*t["adapted_three"]["fpr"]:.2f}%). Both input schemas and all per-target predictions are retained; this comparison does not create independent feature-vector gold labels or a new prospective cohort.')
    text+=' '+schema
    assert json.loads((HERE/'closure_results/en2025_verification.json').read_text())['status']=='passed'
    if stage=='papers':
        p=HERE/'manuscript/main_scirep_revised.tex';main=p.read_text(encoding='utf-8')
        addition=r'\subsection*{Independent EN2025 benign PE reports}'+'\n'+esc(text).replace('EN2025 benign-report collection.',r'EN2025 benign-report collection\cite{chen2026en2025}.')+'\n\n'
        main=main.replace(r'\section*{Discussion}',addition+r'\section*{Discussion}',1)
        primary=r['models']['primary_RF+LGB']
        main=main.replace('Additional mixed-file testing and sequence-window benign errors expose transport limitations.','Additional EN2025 benign PE reports expose collector-schema and transport limitations. Sequence-window errors qualify early-use claims.',1)
        p.write_text(main,encoding='utf-8')
        p=HERE/'manuscript/references_revision.bib'
        with p.open('a',encoding='utf-8') as f:f.write('\n@misc{chen2026en2025, author={Chen, Ziqian}, title={{EN2025}}, year={2026}, howpublished={Zenodo, version v1}, doi={10.5281/zenodo.18218588}, url={https://zenodo.org/records/18218588}}\n')
        p=HERE/'manuscript/supplementary_information.tex';sup=p.read_text(encoding='utf-8')
        extra=r'\clearpage\section*{Additional independent benign PE collection}'+'\n'+esc(text)+'\n\n'
        extra+=r'\begin{table}[H]\centering\small\caption{Frozen-model EN2025 benign-PE assessment. Publisher labels; no target fitting. Binomial intervals are conditional on those labels and may understate application dependence.}\label{tab:en2025}\begin{tabular}{lrrr}\toprule Configuration & Alerts/N & Rate (\%) & 95\% interval (\%)\\\midrule'+'\n'
        for label,key in [('Raw schema: RF+LGB','primary_RF+LGB'),('Raw schema: original three','original_three'),('Raw schema: adapted three','adapted_three')]:
            m=r['models'][key];lo,hi=m['ci95'];extra+=f'{label} & {m["alert_n"]}/{m["n"]} & {100*m["fpr"]:.2f} & {100*lo:.2f}--{100*hi:.2f}'+r'\\'+'\n'
        for label,key in [('Adapter: RF+LGB','primary_RF+LGB'),('Adapter: original three','original_three'),('Adapter: adapted three','adapted_three')]:
            m=translated['models'][key];lo,hi=m['ci95'];extra+=f'{label} & {m["alert_n"]}/{m["n"]} & {100*m["fpr"]:.2f} & {100*lo:.2f}--{100*hi:.2f}'+r'\\'+'\n'
        extra+=r'\bottomrule\end{tabular}\end{table}'+'\n'+esc('Of the 918 selected non-pilot reports, 917 were evaluated and one was excluded because its target hash occurred in a previously used cohort. No replacement was selected.')+'\n'
        sup=sup.replace(r'\end{document}',extra+r'\end{document}',1);p.write_text(sup,encoding='utf-8')
    else:
        p=HERE/'docs/response_letter.tex';response=p.read_text(encoding='utf-8')
        ids=['R1.3','R1.m2','R1.g3','R2.12','R4.1','R3.1']
        for key in ids:
            start=response.index(r'\subsection*{'+key+' ');end=response.find(r'\subsection*{',start+1)
            if end<0:end=response.index(r'\end{document}',start)
            block=response[start:end];position=block.index(r'\textit{Location:}')
            addition=text if key in ['R1.3','R3.1'] else ('The additional independent EN2025 evaluation is now completed with the frozen primary pair and earlier full-stack comparators, using audited PE-only targets. Response R1.3 and the new EN2025 Results/Table S25 provide exact rates, exclusions, publisher-label provenance and CAPE schema limitations. No target tuning was performed.')
            block=block[:position]+r'\textbf{Additional independent benign controls.} '+esc(addition)+'\n\n'+block[position:]
            response=response[:start]+block+response[end:]
        p.write_text(response,encoding='utf-8')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['papers','response']);a=p.parse_args();run(a.stage)
