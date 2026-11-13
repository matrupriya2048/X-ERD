"""Ordered API sequence comparison, plus frozen-source API-only transfer.

The 80 original publisher files are the sample unit. Binary identities are not
available, so independence of underlying executables is not asserted. Windows are never
split across training and test. No artificial ordering of aggregate features.
"""
from pathlib import Path
import copy,json,hashlib,time
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold,train_test_split
from .models import fit
from .metrics import metrics,bootstrap
from .io import save,digest

MAX_CALLS=512
SEEDS=[0,1,2]

def read_calls(path):
    data=json.loads(Path(path).read_text(encoding='utf-8'))
    if not isinstance(data,list) or not all(isinstance(x,dict) and isinstance(x.get('api'),str) and isinstance(x.get('time'),(int,float)) for x in data):
        raise ValueError('Expected publisher timestamped API-event list')
    # Stable sort preserves publisher order for timestamps tied at timer precision.
    data=sorted(data,key=lambda x:x['time'])
    if not data:raise ValueError('Empty call trace')
    return data

def vocab_for(calls,indices):
    # 0 padding; 1 out-of-vocabulary. Vocab sees training records only.
    return {t:i+2 for i,t in enumerate(sorted({t for j in indices for t in calls[j][:MAX_CALLS]}))}

def encode(calls,vocab):
    X=np.zeros((len(calls),MAX_CALLS),dtype=np.int64);lengths=[]
    for i,tokens in enumerate(calls):
        toks=tokens[:MAX_CALLS];lengths.append(max(1,len(toks)));X[i,:len(toks)]=[vocab.get(t,1) for t in toks]
    return X,np.array(lengths)

def presence(encoded,vocab_size):
    X=np.zeros((len(encoded),vocab_size-2),dtype=np.uint8)
    for i,row in enumerate(encoded):
        ids=np.unique(row[row>=2])-2;X[i,ids]=1
    return X

def neural(kind,vocab_size):
    import torch
    from torch import nn
    class OrderedModel(nn.Module):
        def __init__(self):
            super().__init__();self.kind=kind;self.embedding=nn.Embedding(vocab_size,32,padding_idx=0)
            if kind=='LSTM':self.seq=nn.LSTM(32,64,num_layers=2,dropout=.2,batch_first=True)
            elif kind=='CNN':self.seq=nn.Sequential(nn.Conv1d(32,64,5,padding=2),nn.ReLU(),nn.Conv1d(64,64,3,padding=1),nn.ReLU())
            else:raise ValueError(kind)
            self.head=nn.Sequential(nn.Dropout(.2),nn.Linear(64,32),nn.ReLU(),nn.Linear(32,1))
        def forward(self,x,lengths):
            z=self.embedding(x)
            if self.kind=='LSTM':
                # Unidirectional recurrence: states at the final observed token
                # cannot depend on subsequent padding. Dense CPU kernels avoid
                # a slow autograd loop while preserving variable-length semantics.
                z,_=self.seq(z);z=z[torch.arange(len(x),device=x.device),lengths-1]
            else:
                z=self.seq(z.transpose(1,2));mask=torch.arange(x.shape[1],device=x.device)[None,:]>=lengths[:,None]
                z=z.masked_fill(mask[:,None,:],-1e9).max(dim=2).values
            return self.head(z).squeeze(1)
    return OrderedModel()

def fit_neural(kind,calls,y,train,test,seed,out,tag):
    import torch
    torch.set_num_threads(4);torch.manual_seed(seed);np.random.seed(seed)
    torch.use_deterministic_algorithms(True)
    # Validation consists of whole source records. Selection uses no test labels.
    inner_groups=np.array([hashlib.sha256('\n'.join(calls[i][:MAX_CALLS]).encode()).hexdigest() for i in train])
    ia,ib=next(StratifiedGroupKFold(4,shuffle=True,random_state=42).split(np.zeros((len(train),1)),y[train],inner_groups))
    a,b=train[ia],train[ib]
    vocab=vocab_for(calls,a);X,L=encode(calls,vocab)
    xt=torch.from_numpy(X);lt=torch.from_numpy(L);yt=torch.tensor(y,dtype=torch.float32)
    model=neural(kind,len(vocab)+2)
    optim=torch.optim.Adam(model.parameters(),lr=.001,weight_decay=.0001)
    lossfn=torch.nn.BCEWithLogitsLoss()
    best=float('inf');best_state=None;wait=0;history=[];generator=torch.Generator().manual_seed(seed)
    start=time.time()
    for epoch in range(1,61):
        model.train();perm=torch.randperm(len(a),generator=generator).numpy();losses=[]
        for chunk in range(0,len(a),16):
            ix=a[perm[chunk:chunk+16]];optim.zero_grad();loss=lossfn(model(xt[ix],lt[ix]),yt[ix]);loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1);optim.step();losses.append(float(loss.detach()))
        model.eval()
        with torch.no_grad():vl=float(lossfn(model(xt[b],lt[b]),yt[b]))
        history.append({'epoch':epoch,'train_loss':float(np.mean(losses)),'validation_loss':vl})
        if vl<best-1e-5:best=vl;best_state=copy.deepcopy(model.state_dict());best_epoch=epoch;wait=0
        else:wait+=1
        if wait>=8:break
    # Refit the selected epoch count on ALL outer-training records. Both neural
    # and tree models therefore have the same labeled outer training budget.
    vocab=vocab_for(calls,train);X,L=encode(calls,vocab);xt=torch.from_numpy(X);lt=torch.from_numpy(L)
    torch.manual_seed(seed);model=neural(kind,len(vocab)+2)
    optim=torch.optim.Adam(model.parameters(),lr=.001,weight_decay=.0001)
    generator=torch.Generator().manual_seed(seed)
    for epoch in range(best_epoch):
        model.train();perm=torch.randperm(len(train),generator=generator).numpy()
        for chunk in range(0,len(train),16):
            ix=train[perm[chunk:chunk+16]];optim.zero_grad();loss=lossfn(model(xt[ix],lt[ix]),yt[ix]);loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1);optim.step()
    model.eval()
    def predict(enc,lens):
        with torch.no_grad():return torch.sigmoid(model(torch.from_numpy(enc),torch.from_numpy(lens))).numpy()
    predictions=predict(X[test],L[test]);prefixes={}
    for length in [32,128,512]:
        xp=X[test].copy();xp[:,length:]=0;lp=np.minimum(L[test],length)
        prefixes[str(length)]=predict(xp,lp)
    shuffled=X[test].copy();rng=np.random.default_rng(42)
    for i,l in enumerate(L[test]):shuffled[i,:l]=rng.permutation(shuffled[i,:l])
    ps=predict(shuffled,L[test])
    save(out/(tag+'_training.json'),{'inner_train_indices':a,'inner_validation_indices':b,'final_train_indices':train,'test_indices':test,'vocabulary':vocab,
      'seed':seed,'kind':kind,'best_epoch':best_epoch,'history':history,'seconds':time.time()-start,
      'parameter_count':sum(p.numel() for p in model.parameters()),'torch':torch.__version__})
    return predictions,prefixes,ps

def run(ds,root,out):
    root=Path(root);out=Path(out);out.mkdir(exist_ok=True,parents=True)
    manifest=json.loads((root/'config/external_data_sources.json').read_text());rows=manifest['files'];calls=[];audit=[]
    for row in rows:
        p=root/row['path']
        if not p.exists() or digest(p)!=row['sha256']:raise ValueError('Missing or unverified external source '+str(p))
        ev=read_calls(p);tokens=[x['api'].lower() for x in ev];calls.append(tokens)
        audit.append({'source_file':p.name,'family':row['family'],'y_true':row['label'],'event_n':len(ev),
          'duration_seconds':ev[-1]['time']-ev[0]['time'],'first512_sha256':hashlib.sha256('\n'.join(tokens[:MAX_CALLS]).encode()).hexdigest(),
          'all_apis_sha256':hashlib.sha256('\n'.join(tokens).encode()).hexdigest()})
    y=np.array([r['label'] for r in rows]);groups=np.array([r['first512_sha256'] for r in audit]);audit=pd.DataFrame(audit)
    protocol={'source':manifest['source'],'source_doi':'10.17632/yh9p6vs53y.1','source_units':'80 publisher-recorded event logs, not 80 complete Cuckoo reports',
      'sample_unit':'one original file; exact first512 sequences grouped across folds',
      'protocol':'5 stratified group folds seed42; seeds0,1,2 for neural fits; inner stratified group4-fold first fold for validation; select epoch count then refit all outer training records; vocabulary built only from fitting subset; no test selection',
      'models':'2-layer LSTM64 and 2-layer CNN64 with embedding32; Adam0.001 weight_decay0.0001 batch16 max60 epochs patience8; BCE; gradient clip1; CPU deterministic PyTorch',
      'matched_tree':'original three tree learners and pair/full OOF stacks on binary API presence from exactly the same first512 calls; train-only vocabulary',
      'sequence_windows':[32,128,512],'prefix_scope':'call-count sensitivity of same trained model; not pre-encryption latency; no encryption-onset labels',
      'data_limits':'20 benign traces and60 ransomware traces across4 families; no executable hashes in source release, so binary overlap or repeated executable identities cannot be ruled out; estimates exploratory',
      'frozen_transfer':'separate API-only original MLRan architecture fit on63 selected APIs, frozen theta0.5; only API events provided, so no full483 feature zero-filling'}
    save(out/'sequence_protocol.json',protocol);audit.to_csv(out/'sequence_source_audit.csv',index=False)
    # Frozen source-to-external inference on full observed APIs, without target fit.
    api=np.array([n.startswith('API:') for n in ds.names]);names=np.array(ds.names)[api]
    api_sets=[set(toks) for toks in calls]
    transfer_X=np.array([[n.split(':',1)[1].lower() in tokens for n in names] for tokens in api_sets],dtype=np.uint8)
    transfer=fit(ds.X[:,api],ds.y);tp=transfer.probabilities(transfer_X)
    pd.DataFrame({'source_file':audit.source_file,'y_true':y,**tp}).to_csv(out/'sequence_transfer_predictions.csv',index=False)
    transfer_results={n:metrics(y,p) for n,p in tp.items()}
    # Matched within-collection OOF evaluation: predictions stored per unit/seed.
    pred_rows=[];prefix_rows=[];splits=[]
    for fold,(train,test) in enumerate(StratifiedGroupKFold(5,shuffle=True,random_state=42).split(np.zeros((len(y),1)),y,groups)):
        if set(groups[train])&set(groups[test]):raise AssertionError('Sequence duplicate leakage')
        print('Sequence fold',fold,'train/test',len(train),len(test),flush=True)
        splits.extend({'source_file':audit.source_file.iloc[i],'fold':fold} for i in test)
        vocab=vocab_for(calls,train);X,_=encode(calls,vocab);Xp=presence(X,len(vocab)+2)
        tree=fit(Xp[train],y[train]);pp=tree.probabilities(Xp[test])
        for name,p in pp.items():
            pred_rows.extend({'source_file':audit.source_file.iloc[i],'fold':fold,'seed':42,'model':name,'y_true':int(y[i]),'probability':float(v)} for i,v in zip(test,p))
        for kind in ['LSTM','CNN']:
            for seed in SEEDS:
                p,pre,shuffle=fit_neural(kind,calls,y,train,test,seed,out,f'sequence_{kind}_{fold}_{seed}')
                pred_rows.extend({'source_file':audit.source_file.iloc[i],'fold':fold,'seed':seed,'model':kind,'y_true':int(y[i]),'probability':float(v)} for i,v in zip(test,p))
                for length,vals in pre.items():prefix_rows.extend({'source_file':audit.source_file.iloc[i],'model':kind,'seed':seed,'calls':int(length),'y_true':int(y[i]),'probability':float(v)} for i,v in zip(test,vals))
                prefix_rows.extend({'source_file':audit.source_file.iloc[i],'model':kind,'seed':seed,'calls':'shuffled512','y_true':int(y[i]),'probability':float(v)} for i,v in zip(test,shuffle))
        pd.DataFrame(pred_rows).to_csv(out/'sequence_oof_predictions.csv',index=False)
        pd.DataFrame(prefix_rows).to_csv(out/'sequence_prefix_predictions.csv',index=False)
    pd.DataFrame(splits).to_csv(out/'sequence_folds.csv',index=False)
    pred=pd.DataFrame(pred_rows);results={}
    for (name,seed),g in pred.groupby(['model','seed']):
        if len(g)!=len(y) or g.source_file.duplicated().any():raise ValueError('OOF coverage failed')
        results.setdefault(name,{})[str(seed)]=metrics(g.y_true,g.probability)
    pre=pd.DataFrame(prefix_rows);prefix_results=[]
    for (name,seed,window),g in pre.groupby(['model','seed','calls']):prefix_results.append({'model':name,'seed':int(seed),'calls':str(window),'metrics':metrics(g.y_true,g.probability)})
    save(out/'sequences.json',{'n':len(y),'benign':int((y==0).sum()),'ransomware':int(y.sum()),'unique_first512':len(set(groups)),
      'transfer':transfer_results,'oof_models_by_seed':results,'prefix_results':prefix_results,
      'uncertainty':'seeds share the same records; do not count them as independent samples; only20 benign controls, exploratory model comparison'})
    print('Sequence comparison complete',flush=True)
