import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
import cadquery as cq
from OCP.BRep import BRep_Builder
from guided_mesh_cad.cli import main


class DiagnosticExitStatusTests(unittest.TestCase):
    def diagnostic(self, shape, command='diagnose-native', suffix='brep'):
        with tempfile.TemporaryDirectory() as directory:
            folder=Path(directory);native=folder/('input.'+suffix);report=folder/'report.json'
            if suffix=='brep':shape.exportBrep(str(native))
            else:cq.exporters.export(shape,str(native))
            before=native.read_bytes()
            with contextlib.redirect_stdout(io.StringIO()):
                status=main([command,str(native),'--report',str(report)])
            self.assertEqual(native.read_bytes(),before)
            return status,json.loads(report.read_text(encoding='utf-8'))

    def test_native_tolerance_failure_is_diagnostic_status_two(self):
        shape=cq.Solid.makeBox(2,3,4)
        BRep_Builder().UpdateVertex(shape.Vertices()[0].wrapped,.02)
        status,report=self.diagnostic(shape)
        self.assertEqual(status,2)
        self.assertTrue(report['valid'])
        self.assertFalse(report['formal_solid_gate_passed'])
        self.assertAlmostEqual(report['native_tolerance_max_mm']['vertex'],.02)

    def test_detached_topology_is_diagnostic_status_two(self):
        shape=cq.Compound.makeCompound([cq.Solid.makeBox(2,3,4),cq.Edge.makeLine((10,0,0),(11,0,0))])
        status,report=self.diagnostic(shape)
        self.assertEqual(status,2)
        self.assertGreater(report['unowned_topology']['edges'],0)

    def test_negative_material_is_diagnostic_status_two(self):
        shape=cq.Solid(cq.Solid.makeBox(2,3,4).wrapped.Reversed())
        status,report=self.diagnostic(shape)
        self.assertEqual(status,2)
        self.assertTrue(report['valid'])
        self.assertIn('volume',report['formal_solid_gate_error'])

    def test_valid_native_and_step_diagnostics_are_status_zero(self):
        for command,suffix in [('diagnose-native','brep'),('diagnose-step','step')]:
            status,report=self.diagnostic(cq.Solid.makeBox(2,3,4),command,suffix)
            self.assertEqual(status,0)
            self.assertTrue(report['formal_solid_gate_passed'])


if __name__=='__main__':unittest.main()
