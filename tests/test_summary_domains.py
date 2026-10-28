import contextlib
import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
from guided_mesh_cad.reporting import compact_report, print_report, MAX_SUMMARY_BYTES
from guided_mesh_cad.source_domains import audit_domains


class SummaryTests(unittest.TestCase):
    def test_large_unicode_report_keeps_evidence_off_stdout(self):
        report = {k:'复杂曲面'*10000 for k in ('status','acceptance','geometry_acceptance','scope','type','error_code','error')}
        report['errors'] = ['错误'*10000]*100
        report['records'] = [{'huge':list(range(10000))}]*20
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):print_report(report, command='validate', report_path='复杂目录/'*2000, log_path='长路径/'*2000)
        self.assertLessEqual(len(buf.getvalue().strip().encode('utf-8')), MAX_SUMMARY_BYTES)
        value = json.loads(buf.getvalue())
        self.assertNotIn('records',value)
        self.assertTrue(value['details_omitted'])

    def test_full_is_explicit_and_success_is_not_geometry_acceptance(self):
        report = {'status':'completed','acceptance':'evidence_only','records':[1,2,3]}
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):print_report(report,full=True)
        self.assertEqual(json.loads(buf.getvalue()),report)
        short = compact_report(report,ok=True)
        self.assertTrue(short['operation_succeeded'])
        self.assertEqual(short['acceptance'],'evidence_only')


class DomainTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.source = self.root/'source.npz'
        np.savez(self.source,vertices=np.array([[0.,0,0],[1,0,0],[0,1,0],[0,0,1]]),faces=np.array([[0,2,1],[0,1,3],[0,3,2],[1,2,3]],dtype=int))
        self.hash = hashlib.sha256(self.source.read_bytes()).hexdigest()
        self.contract = self.root/'domains.json'
    def tearDown(self):self.tmp.cleanup()
    def run_audit(self,regions,source_hash=None):
        self.contract.write_text(json.dumps({'schema':'guided-mesh-cad.domains.v1','units':'mm','source_sha256':source_hash or self.hash,'domains':regions}),encoding='utf-8')
        return audit_domains(self.source,self.contract,self.root/'out')
    def region(self,role='working',indices=None,**extra):
        return dict(id='main',role=role,triangles=[0,1,2,3] if indices is None else indices,basis='explicit original source selection',**extra)
    def test_complete_accounting_does_not_claim_geometry(self):
        result=self.run_audit([self.region()])
        self.assertTrue(result['ok']); self.assertEqual(result['acceptance'],'evidence_only')
        self.assertEqual(result['report']['geometry_acceptance'],'not_checked')
    def test_missing_and_unknown_are_not_full_acceptance(self):
        result=self.run_audit([self.region(indices=[0,1,2])])
        self.assertFalse(result['ok']); self.assertEqual(result['report']['unassigned_indices'],[3])
        self.assertFalse(self.run_audit([self.region('unknown')])['ok'])
    def test_exclusion_requires_string_authorization(self):
        for bad in (None,False,{},[],0,' '):
            with self.subTest(bad=bad),self.assertRaises(ValueError):self.run_audit([self.region('excluded',authorization=bad)])
        result=self.run_audit([self.region('excluded',authorization='caller decision reference')])
        self.assertTrue(result['ok']);self.assertEqual(result['report']['explicitly_excluded_triangles'],4)
    def test_outliers_cannot_disappear_and_source_change_invalidates(self):
        with self.assertRaises(ValueError):self.run_audit([self.region(indices=[0,0,1,2,3])])
        with self.assertRaises(ValueError):self.run_audit([self.region(indices=[0,1,2,4])])
        with self.assertRaises(ValueError):self.run_audit([self.region()],source_hash='0'*64)

if __name__=='__main__':unittest.main()
