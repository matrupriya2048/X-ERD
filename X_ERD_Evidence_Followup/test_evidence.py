import copy, unittest
from evaluate_mtm import eligible
from evidence_packets import inspect_onset

class EligibilityTests(unittest.TestCase):
    def setUp(self):
        self.report={'info':{'score':1,'platform':'windows','category':'file'},
                     'target':{'file':{'sha256':'a'*64}},
                     'behavior':{'processes':[{'calls':[{'api':'NtWriteFile','time':1}]}]}}

    def test_zero_vector_is_not_an_eligibility_exclusion(self):
        self.assertEqual(eligible(self.report),'eligible')

    def test_missing_hash_prevents_independence_claim(self):
        self.report['target']['file']={}
        self.assertEqual(eligible(self.report),'unprocessed_no_executable_hash')

    def test_malformed_hash_prevents_independence_claim(self):
        self.report['target']['file']={'sha256':'unknown'}
        self.assertEqual(eligible(self.report),'unprocessed_no_executable_hash')

    def test_bad_scores_and_malware_range_are_not_benign(self):
        for score in [None,True,float('nan'),float('inf'),-1,3.01,8]:
            self.report['info']['score']=score
            self.assertEqual(eligible(self.report),'excluded_score_not_publisher_benign_range')

    def test_no_observations_not_counted_as_negative(self):
        self.report['behavior']={}
        self.assertEqual(eligible(self.report),'unprocessed_no_api_observations')

    def test_linux_not_zero_imputed_into_windows_model(self):
        self.report['info']['platform']='linux'
        self.assertEqual(eligible(self.report),'excluded_non_windows_file_execution')

    def test_counts_can_establish_execution_without_call_details(self):
        self.report['behavior']={'apistats':{'1':{'NtClose':2}}}
        self.assertEqual(eligible(self.report),'eligible')

    def test_crypto_followed_by_write_does_not_establish_onset(self):
        self.report['behavior']['processes'][0]['calls']=[{'api':'CryptEncrypt','time':0.5,'status':1},{'api':'NtWriteFile','time':1.0,'status':1}]
        result=inspect_onset(self.report)
        self.assertEqual(result['crypto_api_call_n'],1)
        self.assertEqual(result['write_api_call_n'],1)
        self.assertFalse(result['hint_is_verified_encryption_onset'])
        self.assertEqual(result['ground_truth_status'],'unavailable_in_evaluated_annotation_inputs')

    def test_nonnumeric_timestamp_not_used_as_numeric_onset(self):
        self.report['behavior']['processes'][0]['calls']=[{'api':'CryptEncrypt','time':'2026-11-13'}]
        result=inspect_onset(self.report)
        self.assertEqual(result['other_timestamp_format_calls'],1)
        self.assertIsNone(result['first_timestamped_crypto_hint'])

    def test_sequence_timestamps_preserved_without_inventing_process_or_hash(self):
        result=inspect_onset([{'api':'CryptEncrypt','time':3.0}])
        self.assertEqual(result['numeric_timestamped_calls'],1)
        self.assertEqual(result['first_timestamped_crypto_hint']['json_pointer'],'/0')
        self.assertIsNone(result['first_timestamped_crypto_hint']['pid'])
        self.assertFalse(result['hint_is_verified_encryption_onset'])

if __name__=='__main__':unittest.main()
