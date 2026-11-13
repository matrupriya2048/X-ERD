"""Select cited supplied entries and apply independently checked corrections."""
from pathlib import Path
import re,json
ROOT=Path(__file__).resolve().parents[1];MAN=ROOT/'manuscript'

def main():
    raw=(MAN/'references.bib').read_text(encoding='utf-8');entries={}
    for match in re.finditer(r'@\w+\{([^,]+),',raw):
        begin=raw.index('{',match.start());depth=0
        for i in range(begin,len(raw)):
            depth+=(raw[i]=='{')-(raw[i]=='}')
            if depth==0:entries[match.group(1)]=raw[match.start():i+1];break
    changes={
      'kharraz2015cutting':{'booktitle':'Detection of Intrusions and Malware, and Vulnerability Assessment (DIMVA)'},
      'onwuegbuche2025mlran':{'author':'Onwuegbuche, Faithful Chiagoziem and Adelodun, Sunday Olaoluwa and Jurcut, Anca Delia and Pasquale, Liliana','journal':'Journal of Network and Computer Applications','year':'2026','pages':'104475','doi':'10.1016/j.jnca.2026.104475'},
      'kok2022peda':{'number':'5','pages':'1984--1999'},
      'alvi2025xrguard':{'author':'Alvi, M. Adnan and Jalil, Zunera','pages':'53159--53170'},
      'li2026apimatch':{'pages':'1758--1767'},
      'delong1988comparing':{'title':'Comparing the Areas under Two or More Correlated Receiver Operating Characteristic Curves: A Nonparametric Approach'},
      'pierazzi2020intriguing':{'title':'Intriguing Properties of Adversarial {ML} Attacks in the Problem Space','pages':'1332--1349','doi':'10.1109/SP40000.2020.00073'},
      'cuckoo_sandbox':{'year':'2026','note':'Official legacy Cuckoo project; no unverified runtime version asserted','howpublished':r'\url{https://github.com/cuckoosandbox/cuckoo}'}
    }
    for key,fields in changes.items():
        entry=entries[key]
        for field,value in fields.items():
            pattern=r'(?m)^\s*'+field+r'\s*=\s*\{[^\n]*\},?'
            line='  '+field+' = {'+value+'},'
            if re.search(pattern,entry):entry=re.sub(pattern,lambda _:line,entry)
            else:entry=entry[:-1].rstrip().rstrip(',')+',\n'+line+'\n}'
        entries[key]=entry
    entries.update({
      'singh2025representation':r'''@article{singh2025representation,
  author = {Singh, Priyanka and Borgohain, Samir Kumar and Kumar, Jayendra},
  title = {Optimized feature representation and selection for malware detection using portable executable headers and machine learning},
  journal = {Journal of Computer Virology and Hacking Techniques}, volume = {21}, pages = {10}, year = {2025}, doi = {10.1007/s11416-025-00555-3}
}''',
      'singh2023ffdnn':r'''@article{singh2023ffdnn,
  author = {Singh, Priyanka and Borgohain, Samir Kumar and Sarkar, Achintya Kumar and Kumar, Jayendra and Sharma, Lakhan Dev},
  title = {Feed-Forward Deep Neural Network ({FFDNN})-Based Deep Features for Static Malware Detection},
  journal = {International Journal of Intelligent Systems}, volume = {2023}, pages = {9544481}, year = {2023}, doi = {10.1155/2023/9544481}
}''',
      'singh2022mrmr':r'''@article{singh2022mrmr,
  author = {Singh, Priyanka and Borgohain, Samir Kumar and Sharma, Lakhan Dev and Kumar, Jayendra},
  title = {Minimized feature overhead malware detection machine learning model employing {MRMR}-based ranking},
  journal = {Concurrency and Computation: Practice and Experience}, volume = {34}, number = {17}, pages = {e6992}, year = {2022}, doi = {10.1002/cpe.6992}
}''',
      'singh2022svm':r'''@inproceedings{singh2022svm,
  author = {Singh, Priyanka and Borgohain, Samir Kumar and Kumar, Jayendra},
  title = {Performance Enhancement of {SVM}-based {ML} Malware Detection Model Using Data Preprocessing},
  booktitle = {2022 2nd International Conference on Emerging Frontiers in Electrical and Electronic Technologies ({ICEFEET})},
  pages = {1--4}, year = {2022}, doi = {10.1109/ICEFEET51821.2022.9848192}
}''',
      'singh2022clamp':r'''@inproceedings{singh2022clamp,
  author = {Singh, Priyanka and Borgohain, Samir Kumar and Kumar, Jayendra},
  title = {Investigation and pre-processing of {CLaMP} mlaware dataset for machine learning models},
  booktitle = {2022 6th International Conference on Electronics, Communication and Aerospace Technology ({ICECA})},
  pages = {891--895}, year = {2022}, doi = {10.1109/ICECA55336.2022.10009153}
}''',
      'ispahany2025icnn':r'''@article{ispahany2025icnn,
  author = {Ispahany, Jamil and Islam, Md Rafiqul and Khan, M. Arif and Islam, Md Zahidul},
  title = {{iCNN-LSTM+}: A batch-based incremental ransomware detection system using {Sysmon}},
  journal = {IEEE Access}, volume = {13}, pages = {87978--87998}, year = {2025}, doi = {10.1109/ACCESS.2025.3569635}
}''',
      'zhai2026apired':r'''@article{zhai2026apired,
  author = {Zhai, Jiqiang and Zhai, Zhonghui and Miao, Hong and Wang, Xinyu and Yang, Hailu},
  title = {Early Detection of Windows Crypto-Ransomware Based on {API} Sequences and Deep Learning},
  journal = {Concurrency and Computation: Practice and Experience}, volume = {38}, number = {15}, pages = {e70813}, year = {2026}, doi = {10.1002/cpe.70813}
}''',
      'sgandurra2016elderan':r'''@misc{sgandurra2016elderan,
  author = {Sgandurra, Daniele and Mu{\~n}oz-Gonz{\'a}lez, Luis and Mohsen, Rabih and Lupu, Emil C.},
  title = {Automated Dynamic Analysis of Ransomware: Benefits, Limitations and use for Detection},
  year = {2016}, howpublished = {arXiv:1609.03020}, doi = {10.48550/arXiv.1609.03020}
}''',
      'iwan2026apis':r'''@misc{iwan2026apis,
  author = {Iwan, Wana Ardilah and Suryani, Vera and Yulianto, Fazmah Arif and Pardamean, Bens},
  title = {{API} Call Dataset of {LockBit}, {Netwalker}, {WannaCry}, and {Petya} Ransomware},
  year = {2026}, howpublished = {Mendeley Data, version 1}, doi = {10.17632/yh9p6vs53y.1}
}''',
      'nunes2018dynamic':r'''@misc{nunes2018dynamic,
  author = {Nunes, Matthew}, title = {Dynamic Malware Analysis kernel and user-level calls},
  year = {2018}, howpublished = {Zenodo, version 1.0}, doi = {10.5281/zenodo.1203289}
}''',
      'zirari2025deciphering':r'''@article{zirari2025deciphering,
  author = {Zirari, Khalid and Idrissi, Hamza Kamal and El-Yahyaoui, Ahmed and Bensaid, Hicham and En-Nouaary, Abdeslam},
  title = {Deciphering Ransomware: Strategic {API} Usage and Behavioral Patterns for Advanced Detection Techniques},
  journal = {International Journal of Interactive Mobile Technologies},
  volume = {19}, number = {10}, pages = {199--221}, year = {2025},
  doi = {10.3991/ijim.v19i10.49245}
}''',
      'mitre2026stix':r'''@misc{mitre2026stix,
  author = {{The MITRE Corporation}}, title = {{ATT\&CK STIX} Data: Enterprise Dataset},
  year = {2026}, howpublished = {\url{https://github.com/mitre-attack/attack-stix-data}},
  note = {Snapshot commit a8db2f52eb933c9c99060c123e088b3326d3a607; accessed 13 November 2026}
}'''
    })
    tex=(MAN/'main_scirep_revised.tex').read_text(encoding='utf-8')+'\n'+(MAN/'supplementary_information.tex').read_text(encoding='utf-8')
    keys=[]
    for match in re.finditer(r'\\cite\{([^}]+)\}',tex):
        for key in match.group(1).split(','):
            if key not in keys:keys.append(key)
    missing=set(keys)-set(entries)
    if missing:raise ValueError(f'Missing citations:{missing}')
    (MAN/'references_revision.bib').write_text('% Cited entries for the corrected revision; changes recorded in bibliography_corrections.json.\n\n'+'\n\n'.join(entries[k] for k in keys)+'\n',encoding='utf-8')
    (ROOT/'docs/bibliography_corrections.json').write_text(json.dumps({'changes':changes,'cited_keys':keys,'excluded_unverified_connolly_entry':True,'unrelated_suggested_citations_not_added_automatically':True},indent=2),encoding='utf-8')
    print('Wrote',len(keys),'cited bibliography entries')

if __name__=='__main__':main()
