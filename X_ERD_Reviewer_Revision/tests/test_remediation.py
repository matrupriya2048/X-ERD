import numpy as np
import pytest
from xerd.remediation import select,splits,weighted_bases

def test_selection_preserves_ransomware_recall_constraint():
    rows=[{'index':0,'source':{'recall':.4,'fpr':0},'external_development':{'fpr':0}},
          {'index':1,'source':{'recall':.98,'fpr':.03},'external_development':{'fpr':.1}}]
    chosen,ok=select(rows)
    assert ok and chosen['index']==1

def test_executable_groups_do_not_cross_nested_validation():
    groups=np.repeat(np.arange(90).astype(str),2);y=np.repeat(np.tile([0,1,0],30),2)
    domain=np.repeat(np.tile([0,0,1],30),2);X=np.ones((180,3))
    for a,b in splits(X,y,groups,domain):
        assert not set(groups[a])&set(groups[b])
        for c,d in splits(X[a],y[a],groups[a],domain[a]):
            assert not set(groups[a][c])&set(groups[a][d])
            assert not set(groups[a][c])&set(groups[b])

def test_zero_weight_rows_must_not_enter_model_fitting():
    with pytest.raises(ValueError,match='excluded'):
        weighted_bases(np.zeros((20,3)),np.tile([0,1],10),np.arange(20).astype(str),np.ones(20,dtype=int),0.)
