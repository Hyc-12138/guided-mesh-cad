import unittest
import tempfile
from pathlib import Path
import cadquery as cq
from OCP.BRep import BRep_Builder
from guided_mesh_cad.cad import require_solid, shape_check_report, read_native_unchecked, PatchTopologyError


class NativeIntegrityTests(unittest.TestCase):
    def box(self):
        return cq.Solid.makeBox(2, 3, 4)

    def test_valid_solid_has_complete_ownership(self):
        shape = self.box()
        require_solid(shape)
        report = shape_check_report(shape)
        self.assertEqual(report['unowned_topology'], {'faces':0, 'edges':0, 'vertices':0})
        self.assertTrue(report['native_tolerances_within_limit'])

    def test_each_native_tolerance_kind_is_enforced(self):
        for kind, method in [('vertex','Vertices'),('edge','Edges'),('face','Faces')]:
            with self.subTest(kind=kind):
                shape = self.box()
                item = getattr(shape, method)()[0].wrapped
                builder = BRep_Builder()
                getattr(builder, 'Update'+kind.capitalize())(item, .02)
                report = shape_check_report(shape)
                self.assertTrue(report['valid'])
                self.assertAlmostEqual(report['native_tolerance_max_mm'][kind], .02)
                with self.assertRaises(PatchTopologyError):
                    require_solid(shape)

    def test_exact_limit_is_accepted(self):
        shape = self.box()
        BRep_Builder().UpdateVertex(shape.Vertices()[0].wrapped, .01)
        require_solid(shape)

    def test_unowned_face_edge_vertex_are_rejected(self):
        detached = self.box().translate((10, 0, 0))
        for extra in [detached.Faces()[0], detached.Edges()[0], detached.Vertices()[0]]:
            with self.subTest(kind=extra.ShapeType()):
                shape = cq.Compound.makeCompound([self.box(), extra])
                self.assertTrue(shape.isValid())
                self.assertEqual(len(shape.Solids()), 1)
                self.assertGreater(sum(shape_check_report(shape)['unowned_topology'].values()), 0)
                with self.assertRaises(PatchTopologyError):
                    require_solid(shape)

    def test_diagnostic_reader_preserves_bad_tolerance(self):
        shape = self.box()
        BRep_Builder().UpdateVertex(shape.Vertices()[0].wrapped, .02)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'bad.brep'
            shape.exportBrep(str(path))
            fresh = read_native_unchecked(path)
            self.assertAlmostEqual(shape_check_report(fresh)['native_tolerance_max_mm']['vertex'], .02)
            with self.assertRaises(PatchTopologyError):
                require_solid(fresh)


if __name__ == '__main__':
    unittest.main()
