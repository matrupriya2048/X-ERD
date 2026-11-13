from pathlib import Path
import hashlib,json,re,subprocess
from pypdf import PdfReader
from PIL import Image,ImageOps,ImageDraw
HERE=Path(__file__).resolve().parent
ROOT=HERE.parent/'X_ERD_Reviewer_Revision'

def run():
    reports={};render=HERE/'tmp/render';render.mkdir(parents=True,exist_ok=True)
    files=[HERE/'manuscript/main_scirep_revised.pdf',HERE/'manuscript/supplementary_information.pdf',HERE/'docs/response_letter.pdf']
    for pdf in files:
        reader=PdfReader(pdf);tex=pdf.with_suffix('.tex').read_text(encoding='utf-8');log=pdf.with_suffix('.log').read_text(encoding='utf-8',errors='replace')
        assert not re.findall(r'^.*(?:Overfull|Float too large|undefined references|Citation .* undefined|Reference .* undefined|LaTeX Error).*$',log,re.M)
        text='\n'.join(p.extract_text() or '' for p in reader.pages)
        assert '\ufffd' not in text,pdf.name
        if pdf.stem=='main_scirep_revised':
            abstract=re.search(r'\\begin\{abstract\}(.*?)\\end\{abstract\}',tex,re.S).group(1)
            original=(ROOT/'manuscript/main_scirep_revised.tex').read_text(encoding='utf-8')
            assert 'RF+LightGBM is the primary' in abstract
            assert len(re.sub(r'\\[A-Za-z]+|[{}$]',' ',abstract).split())<=200
            assert len(re.findall(r'\\begin\{(?:figure|table)\*?\}',tex))==8
            assert '6/142' in text and '2/11' in text
        if pdf.stem=='supplementary_information':
            assert len(re.findall(r'\\begin\{table\*?\}',tex))==27
            assert 'Table S22' in text and 'Table S24' in text
        if pdf.stem=='response_letter':
            assert len(re.findall(r'\\subsection\*\{(?:E\d|R[1-4]\.)',tex))==69
        prefix=render/pdf.stem
        for stale in render.glob(pdf.stem+'-*.png'):
            assert stale.parent.resolve()==render.resolve()
            stale.unlink()
        cmd=['pdftoppm','-r','65','-png',str(pdf),str(prefix)]
        done=subprocess.run(cmd,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
        if done.returncode:raise RuntimeError(done.stderr.decode(errors='replace'))
        pages=sorted(render.glob(pdf.stem+'-*.png'))
        assert len(pages)==len(reader.pages)
        contacts=[]
        for start in range(0,len(pages),6):
            batch=pages[start:start+6];canvas=Image.new('RGB',(1110,1530),'#e8ecf1');draw=ImageDraw.Draw(canvas)
            for i,path in enumerate(batch):
                im=Image.open(path).convert('RGB');im.thumbnail((540,730))
                x=(i%2)*555+(555-im.width)//2;y=(i//2)*510+25
                # Fit each page within a 555 x 510 cell.
                im.thumbnail((540,475));x=(i%2)*555+(555-im.width)//2
                canvas.paste(im,(x,y));draw.text(((i%2)*555+12,(i//2)*510+7),path.stem,fill='black')
            path=render/(pdf.stem+f'_contact_{start//6+1}.jpg');canvas.save(path,quality=90);contacts.append(str(path))
        reports[pdf.stem]={'pages':len(reader.pages),'sha256':hashlib.sha256(pdf.read_bytes()).hexdigest(),'all_pages_rendered':True,'contact_sheets':contacts,'visual_inspection':'pending'}
    (HERE/'docs/document_qa.json').write_text(json.dumps(reports,indent=2))
    print(json.dumps({k:{'pages':v['pages'],'contacts':len(v['contact_sheets'])} for k,v in reports.items()},indent=2))

if __name__=='__main__':run()
