import json
import unittest

import cadquery as cq
import numpy as np

from guided_mesh_cad.cad import (
    PatchTopologyError,
    canonical_shared_edge,
    face_from_surface_and_wires,
    interpolate_face,
    sew_solid_with_report,
    shape_check_report,
    validate_shared_boundaries,
)
from guided_mesh_cad.pipeline import _failure_record


class SharedEdgeTests(unittest.TestCase):
    def test_explicit_nonuniform_axial_parameters_prevent_z_foldback(self):
        from OCP.BRep import BRep_Tool
        u = np.array([0.0, 0.18, 0.47, 0.74, 1.0])
        z = np.array([0.0, 0.025, 0.05, 1.7, 2.4])
        grid = np.empty((len(u), len(z), 3))
        for i, ui in enumerate(u):
            angle = 0.35*ui
            for j, zj in enumerate(z):
                radius = 6.0 + 0.03*np.sin(0.7*zj) + 0.02*ui
                grid[i, j] = [radius*np.cos(angle), radius*np.sin(angle), zj]
        face = interpolate_face(grid, 1e-7, u_parameters=u, v_parameters=z)
        surface = BRep_Tool.Surface_s(face.wrapped)
        node_error = max(
            np.linalg.norm(np.array([
                surface.Value(float(ui), float(zj)).X(),
                surface.Value(float(ui), float(zj)).Y(),
                surface.Value(float(ui), float(zj)).Z()])-grid[i, j])
            for i, ui in enumerate(u) for j, zj in enumerate(z))
        self.assertLess(node_error, 1e-9)
        dense_z = np.linspace(z[0], z[-1], 501)
        for ui in np.linspace(u[0], u[-1], 5):
            evaluated_z = np.array([surface.Value(float(ui), float(v)).Z()
                                    for v in dense_z])
            self.assertGreater(np.diff(evaluated_z).min(), 0.0)
            self.assertLess(np.max(np.abs(evaluated_z-dense_z)), 1e-9)

    def test_explicit_parameters_reject_duplicates_and_reordering(self):
        grid = np.zeros((4, 4, 3))
        grid[:, :, 0] = np.arange(4)[:, None]
        grid[:, :, 1] = np.arange(4)[None, :]
        with self.assertRaisesRegex(ValueError, 'strictly increasing'):
            interpolate_face(grid, 1e-7, u_parameters=[0, 0.3, 0.3, 1])
        with self.assertRaisesRegex(ValueError, 'strictly increasing'):
            interpolate_face(grid, 1e-7, v_parameters=[0, 0.7, 0.4, 1])
        with self.assertRaisesRegex(ValueError, 'u_degree'):
            interpolate_face(grid, 1e-7, u_parameters=[0, 0.3, 0.7, 1],
                             u_degree=4)
        with self.assertRaisesRegex(ValueError, 'v_degree'):
            interpolate_face(grid, 1e-7, v_parameters=[0, 0.3, 0.7, 1],
                             v_degree=0)
        with self.assertRaisesRegex(ValueError, 'finite vector'):
            interpolate_face(grid, 1e-7, u_parameters=[0, 0.3, np.nan, 1])
        with self.assertRaisesRegex(ValueError, 'matching the grid'):
            interpolate_face(grid, 1e-7, v_parameters=[0, 0.5, 1])

    def test_one_parameter_direction_uses_normalized_other_axis(self):
        from OCP.BRep import BRep_Tool
        u = np.array([2.0, 2.1, 3.4, 7.0])
        grid = np.empty((4, 4, 3))
        for i, ui in enumerate(u):
            for j, vj in enumerate(np.linspace(0.0, 1.0, 4)):
                grid[i, j] = [ui, vj, ui+2*vj]
        face = interpolate_face(grid, 1e-7, u_parameters=u)
        surface = BRep_Tool.Surface_s(face.wrapped)
        for i, ui in enumerate(u):
            for j, vj in enumerate(np.linspace(0.0, 1.0, 4)):
                p = surface.Value(float(ui), float(vj))
                self.assertLess(np.linalg.norm(
                    np.array([p.X(), p.Y(), p.Z()])-grid[i, j]), 1e-9)

    def test_linear_v_suppresses_nonuniform_cubic_xy_overshoot(self):
        from OCP.BRep import BRep_Tool
        u = np.array([0.0, 0.3, 0.7, 1.0])
        v = np.array([0.0, 0.04, 0.08, 2.0])
        radial_offset = np.array([0.0, 0.2, 0.4, 0.41])
        grid = np.empty((len(u), len(v), 3))
        for i, ui in enumerate(u):
            grid[i, :, 0] = 6.0 + 0.01*ui + radial_offset
            grid[i, :, 1] = 0.2*ui
            grid[i, :, 2] = v
        cubic = BRep_Tool.Surface_s(
            interpolate_face(grid, 1e-7, u_parameters=u,
                             v_parameters=v).wrapped)
        linear = BRep_Tool.Surface_s(
            interpolate_face(grid, 1e-7, u_parameters=u,
                             v_parameters=v, v_degree=1).wrapped)
        dense_v = np.linspace(v[0], v[-1], 1001)
        cubic_x = np.array([cubic.Value(0.5, float(value)).X()
                            for value in dense_v])
        linear_x = np.array([linear.Value(0.5, float(value)).X()
                             for value in dense_v])
        source_min = 6.0 + 0.005 + radial_offset.min()
        source_max = 6.0 + 0.005 + radial_offset.max()
        self.assertGreater(cubic_x.max(), source_max + 3.0)
        self.assertGreaterEqual(linear_x.min(), source_min - 1e-10)
        self.assertLessEqual(linear_x.max(), source_max + 1e-10)

    def test_topology_failure_keeps_value_error_file_contract(self):
        failure = _failure_record(PatchTopologyError('bad seam', {'stage':'sewing'}))
        self.assertEqual(failure['type'], 'ValueError')
        self.assertEqual(failure['diagnostic_type'], 'PatchTopologyError')
        self.assertEqual(failure['error_code'], 'patch_topology_error')
        self.assertEqual(failure['topology_report']['stage'], 'sewing')

    def test_canonical_edge_is_reused_with_explicit_reverse(self):
        points = np.array([[0, 0, 0], [0.4, 0.1, 0.2], [0.9, 0.3, 0.5], [1.4, 0.2, 0.8]])
        binding = {'source_sha256':'a'*64, 'selector':'fixture boundary row 3'}
        edge, source = canonical_shared_edge(points, 1e-7, source_binding=binding)
        report = validate_shared_boundaries([
            {'id':'shared-1', 'edge_a':edge, 'edge_b':edge.reverse(),
             'alignment':'reverse', 'samples':17, 'max_gap_mm':1e-7,
             'source_binding':binding}
        ], 1e-7)
        self.assertFalse(source['source_points_moved'])
        self.assertEqual(source['source_binding'], binding)
        self.assertEqual(report[0]['source_binding'], binding)
        self.assertLessEqual(source['source_point_max_deviation_mm'], 1e-7)
        self.assertTrue(report[0]['same_topods_edge'])
        self.assertLessEqual(report[0]['max_gap_mm'], 1e-7)

    def test_coincident_independent_edges_are_rejected_by_default(self):
        points = np.array([[0, 0, 0], [0.5, 0.2, 0.1], [1, 0, 0.2], [1.5, -0.1, 0.3]])
        a, _ = canonical_shared_edge(points, 1e-7)
        b, _ = canonical_shared_edge(points[::-1], 1e-7)
        with self.assertRaisesRegex(PatchTopologyError, 'do not reuse one TopoDS_Edge') as caught:
            validate_shared_boundaries([
                {'id':'fake-shared', 'edge_a':a, 'edge_b':b,
                 'alignment':'reverse', 'samples':11, 'max_gap_mm':1e-7,
                 'require_same_topology':False}
            ], 1e-7)
        self.assertFalse(caught.exception.report['shared_boundaries'][0]['same_topods_edge'])

    def test_offset_shared_edge_reports_gap_without_snapping(self):
        points = np.array([[0, 0, 0], [0.5, 0.2, 0.1], [1, 0, 0.2], [1.5, -0.1, 0.3]])
        a, _ = canonical_shared_edge(points, 1e-7)
        b, _ = canonical_shared_edge(points[::-1] + [0, 0, 0.002], 1e-7)
        with self.assertRaisesRegex(PatchTopologyError, 'gap exceeds') as caught:
            validate_shared_boundaries([
                {'id':'offset', 'edge_a':a, 'edge_b':b,
                 'alignment':'reverse', 'samples':11, 'max_gap_mm':1e-4}
            ], 1e-7)
        item = caught.exception.report['shared_boundaries'][0]
        self.assertGreater(item['max_gap_mm'], 0.0019)

    def test_foldback_source_boundary_is_rejected(self):
        points = np.array([[0, 0, 0], [1, 0, 0], [0.25, 0, 0], [1.5, 0, 0]])
        with self.assertRaisesRegex(PatchTopologyError, 'folds back') as caught:
            canonical_shared_edge(points, 1e-7)
        self.assertTrue(caught.exception.report['self_contacts'])

    def test_adjacent_foldback_source_boundary_is_rejected(self):
        points = np.array([[0, 0, 0], [1, 0, 0], [0.25, 0, 0]])
        with self.assertRaisesRegex(PatchTopologyError, 'folds back') as caught:
            canonical_shared_edge(points, 1e-7)
        self.assertTrue(caught.exception.report['adjacent_foldbacks'])

    def test_declared_shared_edge_must_belong_to_two_input_faces(self):
        box = cq.Workplane('XY').box(2, 3, 4).val()
        unrelated, _ = canonical_shared_edge(
            np.array([[10, 0, 0], [10.5, 0.2, 0], [11, 0, 0]]), 1e-7)
        with self.assertRaisesRegex(PatchTopologyError, 'exactly two input faces') as caught:
            sew_solid_with_report(
                box.Faces(), 1e-7,
                shared_boundaries=[{'id':'unrelated', 'edge_a':unrelated,
                                    'edge_b':unrelated.reverse(),
                                    'alignment':'reverse', 'max_gap_mm':1e-7}])
        report = caught.exception.report
        self.assertEqual(report['stage'], 'shared_boundary_ownership')
        self.assertEqual(report['shared_boundaries'][0]['owner_faces'], [])

    def test_declared_shared_edge_reports_two_owner_faces(self):
        box = cq.Workplane('XY').box(2, 3, 4).val()
        selected = None
        owners = None
        for edge in box.Edges():
            found = [face for face in box.Faces()
                     if any(edge.wrapped.IsSame(candidate.wrapped)
                            for candidate in face.Edges())]
            if len(found) == 2:
                selected, owners = edge, found
                break
        self.assertIsNotNone(selected)
        _, report = sew_solid_with_report(
            box.Faces(), 1e-7,
            shared_boundaries=[{'id':'box-seam', 'edge_a':selected,
                                'edge_b':selected.reverse(),
                                'alignment':'reverse', 'max_gap_mm':1e-7}])
        self.assertEqual(len(report['shared_boundaries'][0]['owner_faces']), 2)

    def test_valid_box_sewing_returns_detailed_report(self):
        box = cq.Workplane('XY').box(2, 3, 4).val()
        solid, report = sew_solid_with_report(box.Faces(), 1e-7)
        self.assertTrue(solid.isValid())
        self.assertEqual(report['status'], 'valid_closed_solid')
        self.assertEqual(report['free_edges'], 0)
        self.assertEqual(report['multiple_edges'], 0)
        self.assertEqual(report['solid_check']['face_errors'], [])
        json.dumps(report)

    def test_face_from_surface_preserves_caller_wire_edges(self):
        from OCP.BRep import BRep_Tool
        original = cq.Workplane('XY').rect(2, 3).toPending().extrude(1).val().Faces()[0]
        surface = BRep_Tool.Surface_s(original.wrapped)
        outer = original.outerWire()
        rebuilt, report = face_from_surface_and_wires(surface, outer, tolerance=1e-7,
                                                       return_report=True)
        self.assertTrue(report['valid'])
        self.assertEqual(len(rebuilt.Edges()), len(outer.Edges()))
        for source_edge in outer.Edges():
            self.assertTrue(any(source_edge.wrapped.IsSame(edge.wrapped) for edge in rebuilt.Edges()))

    def test_open_network_raises_with_sewing_counts(self):
        box = cq.Workplane('XY').box(2, 3, 4).val()
        with self.assertRaises(PatchTopologyError) as caught:
            sew_solid_with_report(box.Faces()[:-1], 1e-7)
        self.assertGreater(caught.exception.report['free_edges'], 0)
        json.dumps(caught.exception.report)

    def test_shape_check_report_is_json_serializable(self):
        box = cq.Workplane('XY').box(1, 1, 1).val()
        report = shape_check_report(box)
        self.assertTrue(report['valid'])
        self.assertEqual(report['solids'], 1)
        self.assertEqual(report['closed_shells'], 1)
        json.dumps(report)


if __name__ == '__main__':
    unittest.main()
