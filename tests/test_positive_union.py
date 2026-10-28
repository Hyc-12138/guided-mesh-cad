import unittest
from unittest.mock import patch

import cadquery as cq

from guided_mesh_cad import union_positive_solids, validate_positive_union
from guided_mesh_cad.cad import PatchTopologyError
from guided_mesh_cad.material_boundary import _fingerprint
from guided_mesh_cad.positive_union import _native_boolean


class PositiveUnionTests(unittest.TestCase):
    def setUp(self):
        self.left = cq.Solid.makeBox(2, 2, 2)
        self.overlap = cq.Solid.makeBox(2, 2, 2, cq.Vector(1, 0, 0))
        self.remote = cq.Solid.makeBox(1, 1, 1, cq.Vector(8, 0, 0))

    def test_overlapping_material_passes_without_mutating_inputs(self):
        inputs = {'left': self.left, 'overlap': self.overlap}
        before = [_fingerprint(shape) for shape in inputs.values()]
        result, report = union_positive_solids(inputs, expected_solids=1,
            inside_witnesses={'left': [(0.5, 1, 1)], 'overlap': [(2.5, 1, 1)]})
        self.assertAlmostEqual(result.Volume(), 12)
        self.assertTrue(report['inputs_unchanged'])
        self.assertEqual([_fingerprint(shape) for shape in inputs.values()], before)
        self.assertEqual(len(report['steps']), 1)
        self.assertFalse(report['source_conformity_accepted'])
        self.assertFalse(report['native_self_interference_checked'])
        self.assertFalse(report['full_BOP_geometry_acceptance'])
        self.assertTrue(report['native_material_preservation_checked'])
        self.assertEqual(report['native_warning_policy'], 'any_native_warning_is_unresolved')

    def test_disconnected_material_and_multisolid_input_are_retained(self):
        two = cq.Compound.makeCompound([self.left, self.remote])
        result, report = union_positive_solids({'two islands': two, 'overlap': self.overlap},
                                               expected_solids=2)
        self.assertEqual(len(result.Solids()), 2)
        self.assertAlmostEqual(result.Volume(), 13)
        self.assertEqual(report['inputs'][0]['check']['solids'], 2)

    def test_single_multisolid_input_still_gets_final_preservation_checks(self):
        two = cq.Compound.makeCompound([self.left, self.remote])
        result, report = union_positive_solids({'two': two})
        self.assertEqual(len(result.Solids()), 2)
        self.assertEqual(len(report['original_input_preservation']), 1)

    def test_three_named_operands_keep_both_islands(self):
        result, report = union_positive_solids({'left': self.left, 'overlap': self.overlap,
                                               'remote': self.remote}, expected_solids=2)
        self.assertAlmostEqual(result.Volume(), 13)
        self.assertEqual(len(report['steps']), 2)
        self.assertEqual(len(report['original_input_preservation']), 3)

    def test_no_unrequested_minimum_positive_volume(self):
        tiny = cq.Solid.makeBox(.00005, .00005, .00005)
        self.assertGreater(tiny.Volume(), 0)
        self.assertLess(tiny.Volume(), 1e-12)
        result, report = union_positive_solids({'tiny native solid': tiny}, expected_solids=1)
        self.assertGreater(result.Volume(), 0)
        self.assertTrue(report['inputs_unchanged'])

    def test_valid_closed_result_missing_material_is_rejected(self):
        before = _fingerprint(self.left)
        with self.assertRaises(PatchTopologyError) as caught:
            validate_positive_union(self.left, {'left': self.left, 'remote': self.remote})
        report = caught.exception.report
        self.assertEqual(report['status'], 'rejected')
        self.assertGreater(report['original_input_preservation'][-1]['outside_topology']['solids'], 0)
        self.assertTrue(report['inputs_unchanged'])
        self.assertEqual(_fingerprint(self.left), before)

    def test_wrong_optional_component_count_is_rejected(self):
        with self.assertRaisesRegex(PatchTopologyError, 'expected_solids'):
            union_positive_solids({'left': self.left, 'remote': self.remote}, expected_solids=1)

    def test_open_material_is_rejected(self):
        with self.assertRaises(PatchTopologyError):
            union_positive_solids({'sheet': self.left.Faces()[0]})

    def test_negative_or_excessive_tolerance_input_is_rejected(self):
        from OCP.BRep import BRep_Builder
        negative = self.left.copy()
        negative.wrapped.Reverse()
        with self.assertRaisesRegex(PatchTopologyError, 'positive oriented volume'):
            union_positive_solids({'negative': negative})
        BRep_Builder().UpdateVertex(self.left.Vertices()[0].wrapped, .02)
        before = _fingerprint(self.left)
        with self.assertRaisesRegex(PatchTopologyError, 'tolerance'):
            union_positive_solids({'too tolerant': self.left})
        self.assertEqual(_fingerprint(self.left), before)

    def test_zero_volume_residual_vertex_is_never_treated_as_empty(self):
        vertex = cq.Vertex.makeVertex(0, 0, 0)
        with patch('guided_mesh_cad.positive_union._native_boolean', return_value=vertex):
            with self.assertRaisesRegex(PatchTopologyError, 'residual topology'):
                validate_positive_union(self.left, {'left': self.left})

    def test_native_not_done_errors_and_null_shapes_are_rejected(self):
        from OCP.TopoDS import TopoDS_Shape
        for done, errors, message in [(False, False, 'native Perform failed'),
                                      (True, True, 'did not complete'),
                                      (True, False, 'null shape')]:
            from unittest.mock import MagicMock
            operation = MagicMock()
            if not done:
                operation.Perform.side_effect = RuntimeError('native Perform failed')
            operation.HasErrors.return_value = errors
            operation.HasWarnings.return_value = False
            operation.Shape.return_value = TopoDS_Shape()
            with self.subTest(done=done, errors=errors), \
                 patch('OCP.BOPAlgo.BOPAlgo_BOP', return_value=operation), \
                 self.assertRaisesRegex(PatchTopologyError, message):
                union_positive_solids({'left': self.left, 'overlap': self.overlap})

    def test_warning_does_not_certify_otherwise_valid_empty_difference(self):
        from unittest.mock import MagicMock
        empty = cq.Compound.makeCompound([])
        self.assertTrue(empty.isValid())
        for detail in [b'BOPAlgo_AlertUnableToOrientTheShape\n', b'UnclassifiedFutureNativeWarning\n', b'']:
            operation = MagicMock()
            operation.HasErrors.return_value = False
            operation.HasWarnings.return_value = True
            operation.DumpWarnings.side_effect = lambda stream: stream.write(detail)
            operation.Shape.return_value = empty.wrapped
            with self.subTest(detail=detail), patch('OCP.BOPAlgo.BOPAlgo_BOP', return_value=operation):
                with self.assertRaisesRegex(PatchTopologyError, 'evidence unresolved') as caught:
                    validate_positive_union(self.left, {'same': self.left})
            report = caught.exception.report
            self.assertEqual(report['status'], 'unresolved_native_warning')
            self.assertFalse(report['native_material_preservation_checked'])
            self.assertTrue(report['inputs_unchanged'])
            row = report['original_input_preservation'][0]
            self.assertTrue(row['has_warnings'])
            self.assertEqual(row['warnings'], detail.decode())
            self.assertEqual(row['evidence_status'], 'unresolved_native_warning')
            operation.Shape.assert_not_called()

    def test_native_fuse_warning_and_inconsistent_warning_dump_are_unresolved(self):
        from unittest.mock import MagicMock
        for flag in [True, False]:
            operation = MagicMock()
            operation.HasErrors.return_value = False
            operation.HasWarnings.return_value = flag
            operation.DumpWarnings.side_effect = lambda stream: stream.write(b'BOPAlgo_AlertUnableToOrientTheShape\n')
            with self.subTest(flag=flag), patch('OCP.BOPAlgo.BOPAlgo_BOP', return_value=operation):
                with self.assertRaises(PatchTopologyError) as caught:
                    union_positive_solids({'left': self.left, 'overlap': self.overlap})
            self.assertEqual(caught.exception.report['status'], 'unresolved_native_warning')
            self.assertFalse(caught.exception.report['native_material_preservation_checked'])
            self.assertIn('BOPAlgo_AlertUnableToOrientTheShape',
                          caught.exception.report['steps'][0]['fuse']['warnings'])
            operation.Shape.assert_not_called()

    def test_native_failure_cannot_return_success(self):
        with patch('guided_mesh_cad.positive_union._native_boolean', side_effect=RuntimeError('native atomic failure')):
            with self.assertRaisesRegex(PatchTopologyError, 'native atomic failure') as caught:
                union_positive_solids({'left': self.left, 'overlap': self.overlap})
        self.assertTrue(caught.exception.report['inputs_unchanged'])

    def test_step_checks_reject_dropped_previous_or_new_material(self):
        for drop_previous in (False, True):
            def faulty_fuse(kind, left, right, row):
                if kind == 'fuse':
                    return right if drop_previous else left
                return _native_boolean(kind, left, right, row)
            with self.subTest(drop_previous=drop_previous), \
                 patch('guided_mesh_cad.positive_union._native_boolean', side_effect=faulty_fuse), \
                 self.assertRaisesRegex(PatchTopologyError, 'residual topology'):
                union_positive_solids({'left': self.left, 'remote': self.remote})

    def test_strict_inside_witness_rejects_boundary_outside_and_cavity(self):
        hollow = cq.Solid.makeBox(4, 4, 4).cut(cq.Solid.makeBox(2, 2, 2, cq.Vector(1, 1, 1)))
        for point in [(0, 1, 1), (5, 1, 1), (2, 2, 2)]:
            with self.subTest(point=point), self.assertRaisesRegex(PatchTopologyError, 'strictly IN'):
                validate_positive_union(hollow, {'hollow': hollow}, inside_witnesses={'hollow': [point]})

    def test_witness_uses_original_input_even_if_boolean_check_is_unresolved(self):
        with patch('guided_mesh_cad.positive_union._preserved'):
            with self.assertRaisesRegex(PatchTopologyError, 'strictly IN'):
                validate_positive_union(self.left, {'remote': self.remote},
                                        inside_witnesses={'remote': [(8.5, .5, .5)]})

    def test_argument_validation(self):
        for inputs in [{}, {'': self.left}, [('left', self.left)]]:
            with self.subTest(inputs=type(inputs).__name__), self.assertRaises(ValueError):
                union_positive_solids(inputs)
        for count in [0, -1, True, 1.5]:
            with self.subTest(count=count), self.assertRaises(ValueError):
                union_positive_solids({'left': self.left}, expected_solids=count)


if __name__ == '__main__':
    unittest.main()
