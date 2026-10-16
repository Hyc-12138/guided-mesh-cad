import unittest

import cadquery as cq
from OCP.BRep import BRep_Builder
from OCP.BRepTools import BRepTools
from OCP.TopoDS import TopoDS_Shell, TopoDS_Solid
from OCP.TopAbs import TopAbs_INTERNAL, TopAbs_FORWARD

from guided_mesh_cad.cad import PatchTopologyError, require_solid, shape_check_report, volume
from guided_mesh_cad.material_boundary import extract_material_boundary, _fingerprint


def with_sheet(base=None, *, orientation=TopAbs_INTERNAL, mixed=False,
               origin=(1, 1, .5), size=(1, 1), normal=(0, 0, 1), supplied_face=None):
    if base is None:
        base = cq.Solid.makeBox(4, 4, 4)
    face = supplied_face or cq.Face.makePlane(*size, cq.Vector(*origin), cq.Vector(*normal))
    face.wrapped.Orientation(orientation)
    builder = BRep_Builder()
    shell = TopoDS_Shell()
    builder.MakeShell(shell)
    builder.Add(shell, face.wrapped)
    if mixed:
        other = cq.Face.makePlane(.5, .5, cq.Vector(1, 1, .7))
        builder.Add(shell, other.wrapped)
    native = TopoDS_Solid()
    builder.MakeSolid(native)
    for boundary in base.Shells():
        builder.Add(native, boundary.wrapped)
    builder.Add(native, shell)
    return cq.Solid(native), base


class MaterialBoundaryTests(unittest.TestCase):
    def test_internal_sheet_preserves_all_material_faces(self):
        source, original = with_sheet()
        self.assertTrue(source.isValid())
        with self.assertRaises(PatchTopologyError):
            require_solid(source)
        before = _fingerprint(source)
        result, report = extract_material_boundary(source, evidence='Synthetic Boolean INTERNAL sheet')
        require_solid(result)
        self.assertEqual(_fingerprint(source), before)
        self.assertEqual(report['retained_boundary_face_count'], 6)
        self.assertEqual(len(report['removed_internal_open_shells']), 1)
        self.assertAlmostEqual(volume(result)[0], 64., places=10)
        self.assertTrue(all(any(face.wrapped.IsEqual(other.wrapped) for other in result.Faces())
                            for face in original.Faces()))

    def test_default_validator_stays_strict_after_extraction(self):
        source, _ = with_sheet()
        extract_material_boundary(source, evidence='Explicit separate output')
        with self.assertRaises(PatchTopologyError):
            require_solid(source)

    def test_clean_solid_is_unchanged(self):
        source = cq.Solid.makeBox(2, 3, 4)
        result, report = extract_material_boundary(source, evidence='Already clean')
        self.assertIs(result, source)
        self.assertEqual(report['status'], 'unchanged_already_material_boundary')
        self.assertEqual(report['input_brep_sha256'], report['output_brep_sha256'])

    def test_closed_inner_cavity_is_preserved(self):
        outer = cq.Solid.makeBox(4, 4, 4)
        inner = cq.Solid.makeBox(2, 2, 2, cq.Vector(1, 1, 1))
        hollow = outer.cut(inner)
        require_solid(hollow)
        self.assertEqual(len(hollow.Shells()), 2)
        source, _ = with_sheet(hollow)
        result, report = extract_material_boundary(source, evidence='Closed cavity plus internal partition')
        require_solid(result)
        self.assertEqual(len(result.Shells()), 2)
        self.assertEqual(len(result.Faces()), 12)
        self.assertAlmostEqual(volume(result)[0], 56., places=9)
        self.assertEqual(report['retained_closed_shell_count'], 2)

    def test_located_solid_preserves_actual_world_faces(self):
        source, _ = with_sheet()
        source = source.moved(cq.Location(cq.Vector(15, -20, 3)))
        before = _fingerprint(source)
        result, _ = extract_material_boundary(source, evidence='Located internal partition')
        require_solid(result)
        self.assertEqual(_fingerprint(source), before)
        self.assertAlmostEqual(result.Center().x, 17., places=9)
        self.assertAlmostEqual(result.Center().y, -18., places=9)
        self.assertAlmostEqual(result.Center().z, 5., places=9)

    def test_multiple_solids_are_never_selected_or_dropped(self):
        source, _ = with_sheet()
        extra = cq.Solid.makeBox(.01, .01, .01, cq.Vector(10, 10, 10))
        shape = cq.Compound.makeCompound([source, extra])
        before = _fingerprint(shape)
        with self.assertRaises(PatchTopologyError):
            extract_material_boundary(shape, evidence='Must reject any second solid')
        self.assertEqual(_fingerprint(shape), before)

    def test_free_sheet_outside_solid_is_rejected(self):
        source, _ = with_sheet()
        face = cq.Face.makePlane(1, 1, cq.Vector(9, 9, 9))
        shape = cq.Compound.makeCompound([source, face])
        with self.assertRaises(PatchTopologyError):
            extract_material_boundary(shape, evidence='Unowned topology must fail')

    def test_external_open_shell_is_not_removed(self):
        source, _ = with_sheet(orientation=TopAbs_FORWARD)
        before = _fingerprint(source)
        with self.assertRaises(PatchTopologyError):
            extract_material_boundary(source, evidence='Externally oriented sheet')
        self.assertEqual(_fingerprint(source), before)

    def test_mixed_open_shell_is_not_partially_removed(self):
        source, _ = with_sheet(mixed=True)
        before = _fingerprint(source)
        with self.assertRaises(PatchTopologyError):
            extract_material_boundary(source, evidence='Mixed shell must reject')
        self.assertEqual(_fingerprint(source), before)

    def test_high_tolerance_input_is_not_repaired(self):
        source, _ = with_sheet()
        BRep_Builder().UpdateVertex(source.Vertices()[0].wrapped, .02)
        with self.assertRaises(PatchTopologyError):
            extract_material_boundary(source, evidence='High tolerance forbidden')
        self.assertGreater(shape_check_report(source)['native_tolerance_max_mm']['vertex'], .01)

    def test_missing_explicit_evidence_is_rejected(self):
        source, _ = with_sheet()
        for value in ['', '  ', None, 123]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                extract_material_boundary(source, evidence=value)

    def test_falsely_internal_outside_sheet_is_rejected(self):
        source, _ = with_sheet(origin=(1, 1, 6))
        self.assertTrue(source.isValid())
        before = _fingerprint(source)
        with self.assertRaises(PatchTopologyError) as caught:
            extract_material_boundary(source, evidence='Label does not prove containment')
        self.assertIn('outside material', str(caught.exception))
        self.assertEqual(_fingerprint(source), before)

    def test_partly_outside_internal_sheet_is_rejected(self):
        source, _ = with_sheet(origin=(1, 1, 4), size=(2, 2), normal=(1, 0, 0))
        self.assertTrue(source.isValid())
        with self.assertRaises(PatchTopologyError):
            extract_material_boundary(source, evidence='Crosses actual material boundary')

    def test_internal_sheet_in_closed_void_is_rejected(self):
        outer = cq.Solid.makeBox(4, 4, 4)
        inner = cq.Solid.makeBox(2, 2, 2, cq.Vector(1, 1, 1))
        source, _ = with_sheet(outer.cut(inner), origin=(2, 2, 2), size=(.5, .5))
        self.assertTrue(source.isValid())
        with self.assertRaises(PatchTopologyError):
            extract_material_boundary(source, evidence='Cavity is not material')

    def test_inside_corners_do_not_hide_outside_surface(self):
        import numpy as np
        from guided_mesh_cad.cad import interpolate_face
        u = np.linspace(0, 1, 5)
        grid = np.array([[[.5 + 3*x, .5 + 3*y,
                           .5 + 5*16*x*(1-x)*y*(1-y)] for y in u] for x in u])
        face = interpolate_face(grid, 1e-7, u_parameters=u, v_parameters=u)
        self.assertTrue(all(vertex.Center().z < 4 for vertex in face.Vertices()))
        source, _ = with_sheet(supplied_face=face)
        self.assertTrue(source.isValid())
        with self.assertRaises(PatchTopologyError):
            extract_material_boundary(source, evidence='Entire actual surface required, not corners')

    def test_roundtrip_keeps_hole_and_physical_boundary(self):
        import tempfile
        from pathlib import Path
        from guided_mesh_cad.cad import read_step
        block = cq.Solid.makeBox(4, 4, 4)
        bore = cq.Solid.makeCylinder(.4, 4, cq.Vector(2, 2, 0))
        source, _ = with_sheet(block.cut(bore))
        result, _ = extract_material_boundary(source, evidence='Through bore preserved')
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'material.step'
            cq.exporters.export(result, str(path))
            fresh = read_step(path)
            require_solid(fresh)
            self.assertEqual(len(fresh.Faces()), 7)
            self.assertAlmostEqual(volume(fresh)[0], 64 - 4 * 3.141592653589793 * .4**2, places=7)


if __name__ == '__main__':
    unittest.main()
