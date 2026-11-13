"""Check final document structure and compiled references; visual QA is separate."""
from pathlib import Path
import json,re,hashlib
from pypdf import PdfReader

ROOT=Path(__file__).resolve().parents[1];MAN=ROOT/'manuscript'

def main():
    reports={}
    for name in ['main_scirep_revised','supplementary_information']:
        tex=(MAN/(name+'.tex')).read_text(encoding='utf-8')
        if any(ord(c)<32 and c not in '\n\r' for c in tex):raise AssertionError('Control character in generated TeX: '+name)
        log=(MAN/(name+'.log')).read_text(encoding='utf-8',errors='replace')
        errors=re.findall(r'^.*(?:Overfull|Float too large|undefined references|Citation .* undefined|Reference .* undefined|LaTeX Error).*$' ,log,re.M)
        if errors:raise AssertionError(name+': '+repr(errors))
        pdf=MAN/(name+'.pdf')
        if pdf.stat().st_mtime < (MAN/(name+'.tex')).stat().st_mtime:
            raise AssertionError('PDF older than current source: '+name)
        reader=PdfReader(pdf)
        text='\n'.join(p.extract_text() or '' for p in reader.pages)
        if '[PENDING_DOI]' in text:raise AssertionError('Unissued DOI placeholder')
        labels=re.findall(r'\\label\{([^}]+)\}',tex)
        if len(labels)!=len(set(labels)):raise AssertionError('Duplicate display labels')
        if re.findall(r'\\ref\{([^}]+)\}',tex) and not set(re.findall(r'\\ref\{([^}]+)\}',tex))<=set(labels):raise AssertionError('Cross-PDF reference not resolved')
        info={'pdf_sha256':hashlib.sha256(pdf.read_bytes()).hexdigest(),'tex_sha256':hashlib.sha256((MAN/(name+'.tex')).read_bytes()).hexdigest(),'pages':len(reader.pages),
              'figures':len(re.findall(r'\\begin\{figure\*?\}',tex)),
              'tables':len(re.findall(r'\\begin\{table\*?\}',tex)),
              'latex_checks':'No overfull boxes, oversized floats or unresolved references/citations',
              'visual_inspection':'Pending separate PNG inspection'}
        if name=='main_scirep_revised':
            inventory=json.loads((ROOT/'docs/display_inventory.json').read_text())
            missing=[k for k in inventory['main'] if r'\ref{'+k+'}' not in tex]
            if missing:raise AssertionError('Uncited main displays: '+repr(missing))
            seen=list(dict.fromkeys(map(int,re.findall(r'Figure~S(\d+)',tex))))
            if seen!=list(range(1,15)):raise AssertionError('Supplementary first figure callouts: '+repr(seen))
            info['display_callouts']='All eight main displays cited; supplementary figures first cited in S1-S14 order'
            abstract=re.search(r'\\begin\{abstract\}(.*?)\\end\{abstract\}',tex,re.S).group(1)
            abstract=re.sub(r'\\[A-Za-z]+|[{}$]',' ',abstract)
            info['abstract_words']=len(abstract.split())
            if info['abstract_words']>200:raise AssertionError('Abstract exceeds200 words')
            if info['figures']+info['tables']!=8:raise AssertionError('Main display budget')
            # Exclude Abstract, Methods, references and figure/table legends.
            body=tex.split(r'\section*{Introduction}',1)[1].split(r'\section*{Methods}',1)[0]
            body=re.sub(r'\\begin\{(figure\*?|table\*?)\}.*?\\end\{\1\}','',body,flags=re.S)
            body=re.sub(r'\\(?:cite|ref|label)\{[^}]*\}','',body)
            body=re.sub(r'\\[A-Za-z]+|[{}$]',' ',body)
            info['approx_main_words_excluding_methods_displays']=len(body.split())
        reports[name]=info
    (ROOT/'docs/manuscript_qa.json').write_text(json.dumps(reports,indent=2))
    print(json.dumps(reports,indent=2))

if __name__=='__main__':main()
