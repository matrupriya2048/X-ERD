from pathlib import Path
import argparse, re, subprocess
HERE=Path(__file__).resolve().parent

def compile_one(folder,name,bib=False):
    commands=[['pdflatex','-interaction=nonstopmode','-halt-on-error',name+'.tex']]
    if bib:commands.append(['bibtex',name])
    commands.extend([['pdflatex','-interaction=nonstopmode','-halt-on-error',name+'.tex']]*2)
    for i,command in enumerate(commands):
        result=subprocess.run(command,cwd=folder,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,encoding='utf-8',errors='replace')
        (folder/(name+f'.compile{i}.txt')).write_text(result.stdout,encoding='utf-8')
        if result.returncode:raise RuntimeError(result.stdout[-4000:])
    log=(folder/(name+'.log')).read_text(encoding='utf-8',errors='replace')
    errors=re.findall(r'^.*(?:Overfull|Float too large|undefined references|Citation .* undefined|Reference .* undefined|LaTeX Error).*$',log,re.M)
    if errors:raise RuntimeError('\n'.join(errors))
    print(name,'compiled cleanly',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('document',choices=['main','supplement','response']);a=p.parse_args()
    if a.document=='main':compile_one(HERE/'manuscript','main_scirep_revised',True)
    elif a.document=='supplement':compile_one(HERE/'manuscript','supplementary_information',True)
    else:compile_one(HERE/'docs','response_letter')
