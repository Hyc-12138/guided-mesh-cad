import math
import tempfile
import unittest
from pathlib import Path

import numpy as np
import cadquery as cq
from OCP.BRep import BRep_Builder
from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeEdge
from OCP.BRepAdaptor import BRepAdaptor_Curve
from OCP.Geom import Geom_BezierCurve, Geom_BSplineCurve
from OCP.GeomConvert import GeomConvert
from OCP.TColgp import TColgp_Array1OfPnt
from OCP.TColStd import TColStd_Array1OfReal, TColStd_Array1OfInteger
from OCP.gp import gp_Pnt
from guided_mesh_cad import sample_native_edge_bounded


def bezier(points, weights=None, trim=None):
    poles=TColgp_Array1OfPnt(1,len(points))
    for i,p in enumerate(points,1):poles.SetValue(i,gp_Pnt(*p))
    if weights is None:curve=Geom_BezierCurve(poles)
    else:
        w=TColStd_Array1OfReal(1,len(weights))
        for i,v in enumerate(weights,1):w.SetValue(i,v)
        curve=Geom_BezierCurve(poles,w)
    maker=BRepBuilderAPI_MakeEdge(curve) if trim is None else BRepBuilderAPI_MakeEdge(curve,*trim)
    return cq.Edge(maker.Edge())


class NativeCurveSamplingTests(unittest.TestCase):
    def check_dense(self,edge,limit=.001,**kwargs):
        before=None
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'edge.brep';edge.exportBrep(str(path));before=path.read_bytes()
            points,parameters,report=sample_native_edge_bounded(edge,limit,**kwargs)
            edge.exportBrep(str(path));self.assertEqual(before,path.read_bytes())
        adaptor=BRepAdaptor_Curve(edge.wrapped)
        # Independent native evaluations at 19 interior locations per returned
        # interval; this catches aliasing and parameter/location mistakes.
        worst=0.
        for i,(a,b) in enumerate(zip(parameters[:-1],parameters[1:])):
            for f in np.linspace(0,1,21)[1:-1]:
                p=adaptor.Value(float(a+(b-a)*f));actual=np.array([p.X(),p.Y(),p.Z()])
                error=float(np.linalg.norm(actual-((1-f)*points[i]+f*points[i+1])))
                self.assertLessEqual(error,report['intervals'][i]['interpolation_error_upper_bound_mm']+1e-10)
                worst=max(worst,error)
        self.assertLessEqual(worst,limit+1e-10)
        self.assertEqual(parameters[0],adaptor.FirstParameter())
        self.assertEqual(parameters[-1],adaptor.LastParameter())
        self.assertFalse(report['input_geometry_modified'])
        return points,parameters,report

    def test_line_has_two_exact_samples(self):
        points,_,report=self.check_dense(cq.Edge.makeLine((2,3,4),(7,9,11)))
        self.assertEqual(len(points),2);self.assertEqual(report['segments'],1)

    def test_midpoint_aliasing_cubic_is_not_accepted_as_chord(self):
        edge=bezier([(0,0,0),(1,4,0),(2,-4,0),(3,0,0)])
        points,_,report=self.check_dense(edge)
        self.assertGreater(report['segments'],4)
        self.assertGreater(np.max(abs(points[:,1])),1.)

    def test_circle_and_ellipse(self):
        for edge in [cq.Edge.makeCircle(7.5),cq.Edge.makeEllipse(9,2)]:
            self.check_dense(edge,.002)

    def test_positive_rational_bezier_and_translation(self):
        radius=7.5;pts=np.array([(radius,0,0),(radius,radius,0),(0,radius,0)])
        weights=[1,math.sqrt(.5),1]
        p0,_,r0=self.check_dense(bezier(pts,weights),.002)
        p1,_,r1=self.check_dense(bezier(pts+[100,-50,150],weights),.002)
        self.assertEqual(r0['segments'],r1['segments'])
        np.testing.assert_allclose(p1-p0,np.tile([100,-50,150],(len(p0),1)),atol=1e-11)
        np.testing.assert_allclose(np.linalg.norm(p0[:,:2],axis=1),radius,atol=1e-11)

    def test_trimmed_bezier_covers_only_native_trim(self):
        self.check_dense(bezier([(0,0,0),(1,4,0),(2,-4,0),(3,0,0)],trim=(.17,.82)))

    def test_rational_bspline_conversion_preserves_circle(self):
        original=bezier([(10,0,0),(10,10,0),(0,10,0)],[1,math.sqrt(.5),1])
        spline=GeomConvert.CurveToBSplineCurve_s(BRepAdaptor_Curve(original.wrapped).Bezier())
        edge=cq.Edge(BRepBuilderAPI_MakeEdge(spline,.13,.91).Edge())
        points,_,_=self.check_dense(edge,.001)
        np.testing.assert_allclose(np.linalg.norm(points[:,:2],axis=1),10,atol=1e-11)

    def test_periodic_located_bspline(self):
        edge=cq.Edge.makeSpline([cq.Vector(*p) for p in [(0,0,0),(2,1,0),(1,3,0),(-1,2,0)]],periodic=True)
        edge=edge.rotate((0,0,0),(1,2,3),37).translate((100,-50,150))
        points,_,report=self.check_dense(edge,.002)
        np.testing.assert_allclose(points[0],points[-1],atol=1e-10)
        self.assertGreater(report['native_bezier_spans'],1)

    def test_c0_bspline_knot_is_preserved(self):
        poles=TColgp_Array1OfPnt(1,5)
        for i,p in enumerate([(0,0,0),(1,1,0),(2,0,0),(3,-2,0),(4,0,0)],1):poles.SetValue(i,gp_Pnt(*p))
        knots=TColStd_Array1OfReal(1,3);mult=TColStd_Array1OfInteger(1,3)
        for i,(u,m) in enumerate([(0,3),(1,2),(2,3)],1):knots.SetValue(i,u);mult.SetValue(i,m)
        edge=cq.Edge(BRepBuilderAPI_MakeEdge(Geom_BSplineCurve(poles,knots,mult,2,False)).Edge())
        _,parameters,_=self.check_dense(edge)
        self.assertIn(1.,parameters)

    def test_budget_failure_does_not_relax_requested_bound(self):
        with self.assertRaisesRegex(ValueError,'budget'):
            sample_native_edge_bounded(cq.Edge.makeCircle(10),.00001,max_segments=2)

    def test_bspline_decomposition_overhang_is_intersected_with_exact_native_trim(self):
        original=bezier([(0,0,0),(1,4,0),(2,-4,0),(3,0,0)])
        for move_first,move_last in [(True,False),(False,True),(True,True)]:
            spline=GeomConvert.CurveToBSplineCurve_s(BRepAdaptor_Curve(original.wrapped).Bezier())
            edge=cq.Edge(BRepBuilderAPI_MakeEdge(spline).Edge())
            first=np.nextafter(spline.FirstParameter(),np.inf) if move_first else spline.FirstParameter()
            last=np.nextafter(spline.LastParameter(),-np.inf) if move_last else spline.LastParameter()
            # Construct a legitimate synthetic trimmed edge; the sampler must
            # not change this range or the input geometry to make it pass.
            BRep_Builder().Range(edge.wrapped,float(first),float(last))
            self.assertTrue(edge.isValid())
            _,parameters,report=self.check_dense(edge,.0005)
            self.assertEqual(parameters[0],first);self.assertEqual(parameters[-1],last)
            self.assertTrue(report['decomposition_spans_intersected_with_exact_native_trim'])

    def test_even_one_ulp_of_missing_native_parameter_coverage_is_rejected(self):
        from guided_mesh_cad.curve_sampling import _intersect_native_arc_ranges
        with self.assertRaisesRegex(ValueError,'complete actual edge interval'):
            _intersect_native_arc_ranges([(np.nextafter(0.,1.),1.,None,0.)],0.,1.)
        with self.assertRaisesRegex(ValueError,'complete actual edge interval'):
            _intersect_native_arc_ranges([(0.,np.nextafter(1.,0.),None,0.)],0.,1.)

    def test_internal_one_ulp_span_gap_is_not_hidden_by_endpoint_matching(self):
        from guided_mesh_cad.curve_sampling import _intersect_native_arc_ranges
        with self.assertRaisesRegex(ValueError,'gap or overlap'):
            _intersect_native_arc_ranges([(0.,.5,None,0.),(np.nextafter(.5,1.),1.,None,0.)],0.,1.)

    def test_bad_tolerance_and_nonedge_are_rejected(self):
        edge=cq.Edge.makeLine((0,0,0),(1,0,0));BRep_Builder().UpdateVertex(edge.Vertices()[0].wrapped,.02)
        with self.assertRaisesRegex(ValueError,'tolerance'):sample_native_edge_bounded(edge)
        with self.assertRaisesRegex(ValueError,'edge'):sample_native_edge_bounded(cq.Compound.makeCompound([edge]))

    def test_invalid_numeric_arguments_and_binding(self):
        edge=cq.Edge.makeLine((0,0,0),(1,0,0))
        for limit in [0,-1,float('nan'),float('inf')]:
            with self.assertRaises(ValueError):sample_native_edge_bounded(edge,limit)
        with self.assertRaises(ValueError):sample_native_edge_bounded(edge,source_binding={'source_sha256':'x','selector':'e'})


if __name__=='__main__':unittest.main()
