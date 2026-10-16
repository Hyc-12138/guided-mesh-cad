import json,math,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import cadquery as cq
from OCP.BRep import BRep_Builder,BRep_Tool
from OCP.BRepAdaptor import BRepAdaptor_Curve
from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeEdge
from OCP.BRepExtrema import BRepExtrema_DistShapeShape
from OCP.gp import gp_Pnt
from OCP.TopAbs import TopAbs_EDGE
from guided_mesh_cad.memory_validation import (
    NativeEdgeTrimParameterError,owned_extrema_point,
    owned_extrema_point_with_recovery)
from guided_mesh_cad import validate_native_boundary_low_memory
from guided_mesh_cad.mesh import sha256

def xyz(p):return np.array([p.X(),p.Y(),p.Z()])

class ReportedEdge:
    def __init__(self,edge,parameter,query):self.edge=edge;self.parameter=parameter;self.query=query
    def SupportOnShape1(self,index):return self.edge.wrapped
    def ParOnEdgeS1(self,index):return(self.parameter,)
    def PointOnShape2(self,index):return self.query
    def PointOnShape1(self,index):return self.query # Never accepted as the witness.


class OwnedTrimRecoveryTests(unittest.TestCase):


    def test_valid_path_preserves_strict_point_and_needs_no_query(self):
        edge=cq.Edge.makeLine(cq.Vector(),cq.Vector(1,0,0));owner=cq.Compound.makeCompound([edge])
        class Valid:
            def SupportOnShape1(self,index):return edge.wrapped
            def ParOnEdgeS1(self,index):return(.5,)
        old=owned_extrema_point(Valid(),owner);new,record=owned_extrema_point_with_recovery(Valid(),owner,return_report=True)
        np.testing.assert_array_equal(xyz(old),xyz(new));self.assertIsNone(record)

    def test_nonperiodic_outside_is_still_strictly_rejected(self):
        edge=cq.Edge.makeLine(cq.Vector(),cq.Vector(1,0,0))
        for p in [-1.,np.nextafter(0.,-np.inf),np.nextafter(1.,np.inf),2.]:
            with self.subTest(parameter=p):
                with self.assertRaises(NativeEdgeTrimParameterError):owned_extrema_point(ReportedEdge(edge,p,gp_Pnt(2,0,0)),edge)

    def test_recovery_recomputes_large_distance_instead_of_accepting_cpp(self):
        edge=cq.Edge.makeLine(cq.Vector(),cq.Vector(1,0,0));query=gp_Pnt(2,.02,0)
        point,record=owned_extrema_point_with_recovery(ReportedEdge(edge,2.,query),edge,return_report=True)
        np.testing.assert_allclose(xyz(point),[1,0,0],atol=0,rtol=0)
        self.assertGreater(point.Distance(query),1.);self.assertFalse(record['untrimmed_CPP_used_as_witness'])

    def test_bounded_interior_projection_uses_actual_located_frame(self):
        edge=cq.Edge.makeLine(cq.Vector(),cq.Vector(3,0,0)).located(cq.Location(cq.Vector(7,-3,2),cq.Vector(1,2,3),37))
        ad=BRepAdaptor_Curve(edge.wrapped);lo,hi=ad.FirstParameter(),ad.LastParameter();mid=(lo+hi)/2;base=xyz(ad.Value(mid));tangent=xyz(ad.Value(hi))-xyz(ad.Value(lo));normal=np.cross(tangent,[0,0,1]);normal/=np.linalg.norm(normal);query=gp_Pnt(*map(float,base+.02*normal));snapshot=np.array([xyz(ad.Value(u))for u in np.linspace(lo,hi,11)])
        point,record=owned_extrema_point_with_recovery(ReportedEdge(edge,hi+.1,query),edge,return_report=True)
        np.testing.assert_allclose(xyz(point),base,atol=1e-10,rtol=0);self.assertAlmostEqual(point.Distance(query),.02,places=10)
        self.assertEqual(record['selected_candidate']['kind'],'bounded_native_curve_projection')
        np.testing.assert_array_equal(snapshot,np.array([xyz(ad.Value(u))for u in np.linspace(lo,hi,11)]))

    def test_periodic_trim_is_resolved_by_fresh_bounded_projection_not_wrapping(self):
        edge=cq.Edge.makeCircle(2,angle1=30,angle2=150).translate((3,4,5));ad=BRepAdaptor_Curve(edge.wrapped);lo,hi=ad.FirstParameter(),ad.LastParameter();mid=(lo+hi)/2;base=xyz(ad.Value(mid));query=gp_Pnt(*map(float,base+[0,0,.003]))
        point,record=owned_extrema_point_with_recovery(ReportedEdge(edge,mid+2*math.pi,query),edge,return_report=True)
        np.testing.assert_allclose(xyz(point),base,atol=1e-10,rtol=0);self.assertFalse(record['periodic_normalization_applied']);self.assertAlmostEqual(point.Distance(query),.003,places=10)

    def test_trimmed_translated_bspline_uses_original_parameter_interval(self):
        full=cq.Edge.makeSpline([cq.Vector(0,0,0),cq.Vector(1,1,0),cq.Vector(2,-.2,0),cq.Vector(3,.4,0)],parameters=[0,1,2,3]);curve=BRepAdaptor_Curve(full.wrapped).Curve().Curve();edge=cq.Edge(BRepBuilderAPI_MakeEdge(curve,.3,2.4).Edge()).translate((7,-4,2));ad=BRepAdaptor_Curve(edge.wrapped);lo,hi=ad.FirstParameter(),ad.LastParameter();u=1.2;base=xyz(ad.Value(u));query=gp_Pnt(*map(float,base+[0,0,.005]));point,record=owned_extrema_point_with_recovery(ReportedEdge(edge,hi+.1,query),edge,return_report=True)
        np.testing.assert_allclose(xyz(point),base,atol=1e-8,rtol=0);self.assertEqual((record['actual_first_parameter'],record['actual_last_parameter']),(lo,hi));self.assertAlmostEqual(point.Distance(query),.005,places=8)

    def test_foreign_copy_or_different_location_never_recovers(self):
        edge=cq.Edge.makeLine(cq.Vector(),cq.Vector(1,0,0));ex=ReportedEdge(edge,2.,gp_Pnt(2,0,0))
        for owner in [edge.copy(),edge.located(cq.Location(cq.Vector(0,1,0)))]:
            with self.subTest(owner=owner):
                with self.assertRaisesRegex(ValueError,'owned'):owned_extrema_point_with_recovery(ex,owner)

    def test_nonfinite_parameters_and_queries_are_rejected(self):
        edge=cq.Edge.makeLine(cq.Vector(),cq.Vector(1,0,0))
        for p in [np.nan,np.inf,-np.inf]:
            with self.subTest(parameter=p):
                with self.assertRaises(ValueError)as caught:owned_extrema_point_with_recovery(ReportedEdge(edge,p,gp_Pnt()),edge)
                self.assertNotIsInstance(caught.exception,NativeEdgeTrimParameterError)
        with self.assertRaisesRegex(ValueError,'nonfinite'):owned_extrema_point_with_recovery(ReportedEdge(edge,2.,gp_Pnt(np.nan,0,0)),edge)

    def test_high_native_tolerance_is_not_relaxed_by_recovery(self):
        edge=cq.Edge.makeLine(cq.Vector(),cq.Vector(1,0,0));BRep_Builder().UpdateVertex(edge.Vertices()[0].wrapped,.02)
        with self.assertRaisesRegex(ValueError,'tolerance'):owned_extrema_point_with_recovery(ReportedEdge(edge,2.,gp_Pnt()),edge)

    def test_other_errors_are_not_swallowed(self):
        edge=cq.Edge.makeLine(cq.Vector(),cq.Vector(1,0,0))
        with patch('guided_mesh_cad.memory_validation.owned_extrema_point',side_effect=RuntimeError('unrelated failure')):
            with self.assertRaisesRegex(RuntimeError,'unrelated'):owned_extrema_point_with_recovery(None,edge)
        with self.assertRaisesRegex(ValueError,'boolean'):owned_extrema_point_with_recovery(None,edge,return_report=1)

    def test_public_low_memory_streams_recovery_and_measures_actual_points(self):
        import guided_mesh_cad.memory_validation as mv
        V=np.array([[0,0,0],[2,0,0],[2,2,0],[0,2,0],[0,0,2],[2,0,2],[2,2,2],[0,2,2],[1,0,0]],float)
        F=np.array([[0,2,1],[0,3,2],[4,5,6],[4,6,7],[0,1,5],[0,5,4],[1,2,6],[1,6,5],[2,3,7],[2,7,6],[3,0,4],[3,4,7]],int)
        real=mv.owned_extrema_point
        def injected(extrema,owner,index=1):
            support=extrema.SupportOnShape1(index)
            if support.ShapeType()==TopAbs_EDGE:
                from OCP.TopoDS import TopoDS
                edge=cq.Edge(TopoDS.Edge_s(support));ad=BRepAdaptor_Curve(edge.wrapped);p=ad.LastParameter()+.01;ex=ReportedEdge(edge,p,extrema.PointOnShape2(index));raise NativeEdgeTrimParameterError(ex,owner,index,p,ad.FirstParameter(),ad.LastParameter())
            return real(extrema,owner,index)
        with tempfile.TemporaryDirectory()as temp:
            root=Path(temp);np.savez(root/'source.npz',vertices=V,faces=F);cq.exporters.export(cq.Solid.makeBox(2,2,2),str(root/'shape.step'))
            with patch('guided_mesh_cad.memory_validation.owned_extrema_point',side_effect=injected):
                report=validate_native_boundary_low_memory(root/'source.npz',root/'shape.step',root/'report.json',samples=100,seed=73191,diagnostics_dir=root/'diagnostics')
            self.assertTrue(report['passed']);strategy=report['execution_strategy'];self.assertGreater(strategy['native_trim_parameter_recoveries'],0);record=strategy['native_trim_parameter_recovery_records'];path=Path(record['path']);self.assertEqual(sha256(path),record['sha256']);rows=[json.loads(x)for x in path.read_text().splitlines()];self.assertEqual(len(rows),record['records'])
            for row in rows:self.assertFalse(row['original_parameter_accepted']);self.assertFalse(row['original_parameter_clamped'])
            with np.load(report['diagnostics']['source_actual_boundary_samples'])as arrays:
                np.testing.assert_allclose(np.linalg.norm(arrays['source_sample_points']-arrays['source_projection_points'],axis=1),arrays['source_to_native_distances'],atol=1e-14,rtol=0)

if __name__=='__main__':unittest.main()
