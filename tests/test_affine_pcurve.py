import unittest
import numpy as np
from guided_mesh_cad import pullback_affine_pcurve, make_affine_pcurve_edge
from OCP.Geom import Geom_BSplineSurface
from OCP.TColgp import TColgp_Array2OfPnt
from OCP.TColStd import TColStd_Array1OfReal, TColStd_Array1OfInteger
from OCP.gp import gp_Pnt


def fixture(rational=False, c0=False, offset=0.):
    degree = 2 if c0 else 3
    n = 5
    pp = TColgp_Array2OfPnt(1, n, 1, n)
    from OCP.TColStd import TColStd_Array2OfReal
    ww = TColStd_Array2OfReal(1, n, 1, n)
    for i in range(n):
        for j in range(n):
            z = .07*i*i+.13*j*j+.04*i*j
            if c0:
                z += max(i-2, 0)*1.7
            pp.SetValue(i+1, j+1, gp_Pnt(offset+i*1.1, -offset+j*.8, offset+z))
            ww.SetValue(i+1, j+1, 1+.1*((i+2*j) % 3) if rational else 1.)
    def kv(values):
        kk = TColStd_Array1OfReal(1, 3)
        mm = TColStd_Array1OfInteger(1, 3)
        for i, v in enumerate(values, 1):
            kk.SetValue(i, v)
            mm.SetValue(i, degree+1 if i in [1, 3] else 2 if c0 else 1)
        return kk, mm
    ku, mu = kv([-2., .3, 5.])
    kvv, mv = kv([3., 4.7, 9.])
    return Geom_BSplineSurface(pp, ww, ku, kvv, mu, mv, degree, degree, False, False)


class AffinePCurveTests(unittest.TestCase):
    def compare(self, surface, start, end, tr=(-7.2, 13.4), limit=1e-8):
        curve, report, arrays = pullback_affine_pcurve(surface, start, end,
            parameter_range=tr, max_deviation_mm=limit)
        errors = []
        for t in np.linspace(*tr, 1001):
            uv = np.array(start)+(np.array(end)-start)*((t-tr[0])/(tr[1]-tr[0]))
            errors.append(curve.Value(float(t)).Distance(surface.Value(*map(float, uv))))
        self.assertLessEqual(max(errors), report['whole_curve_bound_with_floating_allowance_mm'])
        self.assertLessEqual(report['whole_curve_bound_with_floating_allowance_mm'], limit)
        self.assertEqual(curve.FirstParameter(), tr[0])
        self.assertEqual(curve.LastParameter(), tr[1])
        self.assertEqual(len(arrays['source_homogeneous_Bernstein']), report['native_span_count'])
        return curve, report, arrays

    def test_nonrational_diagonal_crosses_both_axes(self):
        self.compare(fixture(), [-2, 3], [5, 9])

    def test_rational_diagonal_crosses_both_axes(self):
        self.compare(fixture(True), [-2, 3], [5, 9])

    def test_reverse_uv_both_axes(self):
        self.compare(fixture(True), [5, 9], [-2, 3])

    def test_reverse_only_U(self):
        self.compare(fixture(True), [4.9, 3.1], [-1.9, 8.9])

    def test_interior_partial_range(self):
        self.compare(fixture(True), [-.8, 4.1], [3.7, 7.8], (21.3, 22.9))

    def test_exact_domain_max_iso_U(self):
        _, report, _ = self.compare(fixture(True), [5, 3], [5, 9])
        self.assertEqual(report['degree'], 3)

    def test_exact_domain_min_iso_V_reverse(self):
        _, report, _ = self.compare(fixture(True), [5, 3], [-2, 3])
        self.assertEqual(report['degree'], 3)

    def test_iso_internal_surface_knot(self):
        self.compare(fixture(True), [.3, 3], [.3, 9])

    def test_knot_endpoint_partial(self):
        self.compare(fixture(True), [.3, 4.7], [4.7, 8.7])

    def test_C0_is_preserved(self):
        from OCP.BRepAdaptor import BRepAdaptor_Curve
        from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeEdge
        from OCP.GeomAbs import GeomAbs_C1
        curve, report, _ = self.compare(fixture(c0=True), [-2, 3.8], [5, 8.1])
        self.assertTrue(any(x['preserved_continuity_order'] == 0 for x in report['continuity_at_crossings']))
        edge = BRepBuilderAPI_MakeEdge(curve).Edge()
        self.assertGreater(BRepAdaptor_Curve(edge).NbIntervals(GeomAbs_C1), 1)

    def test_isoline_does_not_cross_orthogonal_C0(self):
        _, report, _ = self.compare(fixture(c0=True), [.3, 3.1], [.3, 4.6])
        self.assertEqual(report['continuity_at_crossings'], [])

    def test_low_coordinate_translation(self):
        self.compare(fixture(True, offset=1000), [-2, 3], [5, 9])

    def test_surface_unchanged(self):
        from guided_mesh_cad.affine_pcurve import _geometry_hash
        s = fixture(True)
        before = _geometry_hash(s)
        self.compare(s, [-2, 3], [5, 9])
        self.assertEqual(before, _geometry_hash(s))

    def test_native_affine_pcurve_actual_range(self):
        from OCP.BRep import BRep_Tool
        from OCP.BRepAdaptor import BRepAdaptor_Curve
        from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeFace
        from OCP.BRepAdaptor import BRepAdaptor_Curve2d
        s = fixture(True)
        edge, report, _ = make_affine_pcurve_edge(s, [-2, 3], [5, 9], parameter_range=(-3, 9))
        self.assertTrue(edge.isValid())
        face = BRepBuilderAPI_MakeFace(s, 1e-7).Face()
        pc = BRepAdaptor_Curve2d(edge.wrapped, face)
        ad = BRepAdaptor_Curve(edge.wrapped)
        self.assertEqual([pc.FirstParameter(), pc.LastParameter()], [-3, 9])
        self.assertEqual([ad.FirstParameter(), ad.LastParameter()], [-3, 9])
        self.assertLessEqual(BRep_Tool.Tolerance_s(edge.wrapped), 1e-7)
        for t in np.linspace(-3, 9, 129):
            uv = pc.Value(float(t))
            self.assertLessEqual(ad.Value(float(t)).Distance(s.Value(uv.X(), uv.Y())), report['whole_curve_bound_with_floating_allowance_mm'])

    def test_topological_reverse_only(self):
        from OCP.TopAbs import TopAbs_REVERSED
        edge, _, _ = make_affine_pcurve_edge(fixture(), [-2, 3], [5, 9], reverse=True)
        self.assertEqual(edge.wrapped.Orientation(), TopAbs_REVERSED)

    def test_shared_vertices_retained(self):
        import cadquery as cq
        from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeVertex
        s = fixture()
        a = cq.Vertex(BRepBuilderAPI_MakeVertex(s.Value(-2, 3)).Vertex())
        b = cq.Vertex(BRepBuilderAPI_MakeVertex(s.Value(5, 9)).Vertex())
        edge, _, _ = make_affine_pcurve_edge(s, [-2, 3], [5, 9], start_vertex=a, end_vertex=b)
        self.assertTrue(any(v.isSame(a) for v in edge.Vertices()))
        self.assertTrue(any(v.isSame(b) for v in edge.Vertices()))

    def test_bad_shared_vertex_rejected(self):
        import cadquery as cq
        with self.assertRaises(ValueError):
            make_affine_pcurve_edge(fixture(), [-2, 3], [5, 9], start_vertex=cq.Vertex.makeVertex(99, 0, 0))

    def test_invalid_uv_and_domain(self):
        for a, b in [([0, 3], [0, 3]), ([np.nan, 3], [1, 4]), ([-2-1e-12, 3], [5, 9]), ([0, 3], [5, 9+1e-12])]:
            with self.subTest(a=a, b=b), self.assertRaises(ValueError):
                pullback_affine_pcurve(fixture(), a, b)

    def test_bad_parameter_ranges(self):
        for tr in [(1, 1), (3, -2), (np.inf, np.inf), (-1e308, 1e308)]:
            with self.subTest(tr=tr), self.assertRaises(ValueError):
                pullback_affine_pcurve(fixture(), [-2, 3], [5, 9], parameter_range=tr)

    def test_invalid_budgets(self):
        for lim in [0, -1, np.nan, np.inf, True]:
            with self.subTest(limit=lim), self.assertRaises(ValueError):
                pullback_affine_pcurve(fixture(), [-2, 3], [5, 9], max_deviation_mm=lim)

    def test_impossible_floating_budget_rejected(self):
        with self.assertRaises(ValueError):
            pullback_affine_pcurve(fixture(True), [-2, 3], [5, 9], max_deviation_mm=1e-18)

    def test_large_parameter_offset_budget_is_accounted(self):
        with self.assertRaises(ValueError):
            pullback_affine_pcurve(fixture(True), [-2, 3], [5, 9],
                parameter_range=(1e12, 1e12+10), max_deviation_mm=1e-8)

    def test_arbitrary_shifted_native_range(self):
        self.compare(fixture(True), [-1, 3.4], [4.6, 8.2], (3200., 3812.))

    def test_true_reversed_parameter_range_is_rejected(self):
        with self.assertRaises(ValueError):
            make_affine_pcurve_edge(fixture(), [-2, 3], [5, 9], parameter_range=(3, 1))

    def test_narrow_native_range_is_certified_or_rejected(self):
        self.compare(fixture(), [-2, 3], [5, 9], (-.00001, .00001))

    def test_unrepresentable_native_span_rejected(self):
        with self.assertRaises(ValueError):
            pullback_affine_pcurve(fixture(), [-2, 3], [5, 9], parameter_range=(1e16, 1e16+2))

    def test_span_budget_rejected(self):
        with self.assertRaises(ValueError):
            pullback_affine_pcurve(fixture(), [-2, 3], [5, 9], max_spans=1)

    def test_topology_budget_rejected(self):
        with self.assertRaises(ValueError):
            make_affine_pcurve_edge(fixture(), [-2, 3], [5, 9], tolerance=.1)

    def test_polyline_real_corner_is_not_joined_or_smoothed(self):
        s = fixture()
        a, _, _ = make_affine_pcurve_edge(s, [-2, 3], [.3, 4.7])
        b, _, _ = make_affine_pcurve_edge(s, [.3, 4.7], [5, 4.7])
        self.assertEqual(len(a.Edges())+len(b.Edges()), 2)
        na = np.array(a.tangentAt(1).toTuple())
        nb = np.array(b.tangentAt(0).toTuple())
        self.assertLess(abs(na@nb), .999)


if __name__ == '__main__':
    unittest.main()
