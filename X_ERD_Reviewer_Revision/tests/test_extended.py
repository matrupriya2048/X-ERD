import json
import numpy as np
import pytest
from xerd.published import EldeRan
from xerd.sequences import read_calls,vocab_for,encode,presence,neural
from xerd.external_mapping import AttackReference,agreement
from xerd.external_raw import interval

def test_timestamp_order_ties_and_input_rejection(tmp_path):
    p=tmp_path/'r.json';p.write_text(json.dumps([{'api':'B','time':2},{'api':'A','time':1},{'api':'C','time':2}]))
    assert [r['api'] for r in read_calls(p)]==['A','B','C']
    p.write_text(json.dumps({'behavior':{}}))
    with pytest.raises(ValueError):read_calls(p)

def test_vocabulary_excludes_test_only_tokens():
    calls=[['a','b','a'],['private_test_token','b']]
    v=vocab_for(calls,[0]);X,L=encode(calls,v)
    assert 'private_test_token' not in v and X[1,0]==1 and X[0,3]==0
    assert presence(X,len(v)+2).tolist()==[[1,1],[0,1]]

def test_lstm_padding_does_not_change_last_observed_state():
    torch=pytest.importorskip('torch');torch.manual_seed(42)
    m=neural('LSTM',5).eval();a=torch.tensor([[2,3,4,0,0]]);b=torch.tensor([[2,3,4,2,4]])
    with torch.no_grad():assert torch.allclose(m(a,torch.tensor([3])),m(b,torch.tensor([3])),atol=1e-7)

def test_legacy_attack_ids_use_recorded_replacements(tmp_path):
    a={'type':'attack-pattern','id':'a','revoked':True,'external_references':[{'source_name':'mitre-attack','external_id':'T1234'}]}
    b={'type':'attack-pattern','id':'b','external_references':[{'source_name':'mitre-attack','external_id':'T1234.001'}]}
    r={'type':'relationship','id':'r','relationship_type':'revoked-by','source_ref':'a','target_ref':'b'}
    p=tmp_path/'s.json';p.write_text(json.dumps({'objects':[a,b,r]}));ref=AttackReference(p)
    assert ref.resolve('T1234')=='T1234.001' and ref.resolve('T9999') is None
    assert agreement({'T1','T2'},{'T2','T3'})['jaccard']==pytest.approx(1/3)

def test_elderan_objective_and_selection_exclude_external_labels():
    X=np.array([[0,0],[0,1],[1,0],[1,1]]*5);y=X[:,0]
    m=EldeRan(regularization=.002,k=1).fit(X,y)
    assert m.selected_.tolist()==[0] and m.model_.C==pytest.approx(1/(len(y)*.002))
    assert m.predict_proba(X).shape==(20,2)

def test_binomial_interval_boundary_is_not_false_certainty():
    lo,hi=interval(0,20);assert lo==0 and .15<hi<.20
    assert interval(20,20)[1]==1
