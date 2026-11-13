import unittest
from cape_adapter import adapt,canonical_path
class CapeTests(unittest.TestCase):
    def test_original_cuckoo_is_unchanged(self):
        d={'info':{'version':'2.0.7'}};v,a=adapt(d);self.assertIs(v,d);self.assertFalse(a['adapter_applied'])
    def test_typed_events_and_nested_strings_are_preserved(self):
        d={'info':{'version':'2.4-CAPE'},'target':{'file':{'strings':['CryptEncrypt']}},'behavior':{'enhanced':[{'event':'read','object':'registry','data':{'regkey':r'HKEY_LOCAL_MACHINE\\Software'}},{'event':'load','object':'library','data':{'file':'crypt32.dll'}}]}}
        v,a=adapt(d);self.assertEqual(v['strings'],['CryptEncrypt']);self.assertEqual(v['behavior']['summary']['regkey_read'],[r'HKEY_LOCAL_MACHINE\Software']);self.assertEqual(v['behavior']['summary']['dll_loaded'],['crypt32.dll']);self.assertNotIn('summary',d['behavior'])
    def test_aggregate_file_access_does_not_invent_writes(self):
        v,a=adapt({'info':{'version':'CAPE'},'behavior':{'summary':{'files':['x']},'enhanced':[{'event':'rename','object':'file','data':{'file':'x'}}]}})
        self.assertNotIn('file_written',v['behavior']['summary']);self.assertNotIn('strings',v)
    def test_root_strings_have_precedence(self):
        v,a=adapt({'info':{'version':'CAPE'},'strings':['root'],'target':{'file':{'strings':['nested']}}});self.assertEqual(v['strings'],['root'])
    def test_unc_and_plain_paths_preserve_structure(self):
        self.assertEqual(canonical_path(r'\\\\server\\share'),r'\\server\share');self.assertEqual(canonical_path(r'C:\temp'),r'C:\temp')
if __name__=='__main__':unittest.main()
