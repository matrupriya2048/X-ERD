import unittest
import numpy as np
from primary_pair import probabilities

class Base:
    def __init__(self,p):self.p=p
    def predict_proba(self,X):return np.tile([1-self.p,self.p],(len(X),1))
class Forbidden:
    def predict_proba(self,X):raise AssertionError('Third learner invoked')
class Meta:
    def predict_proba(self,X):
        assert X.shape[1]==2
        p=X.mean(1);return np.column_stack([1-p,p])
class PassthroughMeta:
    def predict_proba(self,X):
        assert np.allclose(X,[[.2,.6,0.,1.]])
        return np.array([[.3,.7]])
class PrimaryTests(unittest.TestCase):
    def setUp(self):self.a={'feature_ids':['1','2'],'base':{'RF':Base(.2),'LGB':Base(.6),'XGB':Forbidden()},'meta':Meta()}
    def test_only_required_learners_are_invoked(self):self.assertTrue(np.allclose(probabilities(self.a,[[0,1]]),[.4]))
    def test_missing_or_extra_features_rejected(self):
        with self.assertRaises(ValueError):probabilities(self.a,[[1]])
    def test_nonbinary_nonfinite_values_rejected(self):
        for row in [[np.nan,0],[2,0]]:
            with self.assertRaises(ValueError):probabilities(self.a,[row])
    def test_passthrough_appends_raw_features_after_pair_scores(self):
        self.a.update(passthrough=True,meta=PassthroughMeta());self.assertTrue(np.allclose(probabilities(self.a,[[0,1]]),[.7]))
if __name__=='__main__':unittest.main()
