import json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import cadquery as cq
from OCP.BRep import BRep_Builder
from guided_mesh_cad import validate_native_boundary,validate_native_boundary_uv_witness
from guided_mesh_cad.boundary_validation import _native_uv_triangle_hint


BOX_V=np.array([[0,0,0],[2,0,0],[2,2,0],[0,2,0],[0,0,2],[2,0,2],[2,2,2],[0,2,2]],float)
BOX_F=np.array([[0,2,1],[0,3,2],[4,5,6],[4,6,7],[0,1,5],[0,5,4],[1,2,6],[1,6,5],[2,3,7],[2,7,6],[3,0,4],[3,4,7]],np.int64)


class UVWitnessValidationTests(unittest.TestCase):
    def setUp(self):
        self.temporary=tempfile.TemporaryDirectory();self.root=Path(self.temporary.name)
    def tearDown(self):self.temporary.cleanup()
    def fixture(self,vertices=BOX_V,faces=BOX_F,shape=None):
        source=self.root/'source.npz';step=self.root/'native.step';np.savez_compressed(source,vertices=vertices,faces=faces)
        cq.exporters.export(cq.Solid.makeBox(2,2,2) if shape is None else shape,str(step));return source,step
    def run_api(self,source,step,name,fast):
        folder=self.root/name
        function=validate_native_boundary_uv_witness if fast else validate_native_boundary
        result=function(source,step,self.root/(name+'.json'),samples=100,diagnostics_dir=folder)
        with np.load(folder/'actual_boundary_samples.npz') as cache:
            arrays={key:cache[key].copy() for key in cache.files}
        return result,arrays

    def test_cube_passes_and_records_actual_point_witnesses(self):
        source,step=self.fixture();before=step.read_bytes();report,cache=self.run_api(source,step,'fast',True)
        self.assertTrue(report['passed']);self.assertEqual(before,step.read_bytes())
        self.assertEqual(report['verification_method'],'actual_trimmed_native_boundary_uv_witness')
        flags=cache['source_uv_witness_used'];self.assertGreater(flags.sum(),0)
        np.testing.assert_allclose(np.linalg.norm(cache['source_sample_points']-cache['source_projection_points'],axis=1),cache['source_to_native_distances'],atol=1e-12)
        self.assertTrue(np.isfinite(cache['source_uv_witness_parameters'][flags]).all())
        self.assertTrue((cache['source_uv_witness_face_ids'][flags]>=0).all())
        self.assertEqual(report['source_vertices'],8);self.assertEqual(report['source_triangle_centres'],12)
        self.assertEqual(len(report['native_face_coverage']),6)

    def test_old_default_does_not_use_witness_helper(self):
        source,step=self.fixture()
        with patch('guided_mesh_cad.boundary_validation._native_uv_triangle_hint',side_effect=AssertionError('old API must not call hint')):
            report,cache=self.run_api(source,step,'old',False)
        self.assertTrue(report['passed']);self.assertEqual(report['verification_method'],'actual_trimmed_native_boundary')
        self.assertNotIn('source_uv_witness',report);self.assertNotIn('source_uv_witness_used',cache)

    def test_old_and_fast_preserve_complete_reverse_geometry_and_source_identity(self):
        source,step=self.fixture();old,oc=self.run_api(source,step,'old',False);fast,fc=self.run_api(source,step,'fast',True)
        for key in ['source_sample_points','native_actual_points','native_to_source_distances']:np.testing.assert_array_equal(oc[key],fc[key])
        for key in ['source_sha256','step_sha256','source_vertices','source_triangle_centres','native_edges_sampled','native_vertices_sampled','native_face_coverage']:self.assertEqual(old[key],fast[key])

    def test_over_limit_source_points_keep_original_global_fallback(self):
        vertices=BOX_V.copy();vertices[4:,2]+=.025;source,step=self.fixture(vertices)
        old,_=self.run_api(source,step,'old',False);fast,cache=self.run_api(source,step,'fast',True)
        self.assertFalse(old['passed']);self.assertFalse(fast['passed'])
        self.assertGreaterEqual(fast['source_to_actual_native_boundary_upper_bounds']['max_mm'],.025-1e-10)
        self.assertGreater(fast['source_uv_witness']['hint_rejections']['actual_point_over_limit'],0)
        self.assertGreater(fast['source_complete_global_boundary_fallbacks'],0)
        self.assertFalse(cache['source_uv_witness_used'][4:8].any())

    def test_missing_source_faces_still_fail_full_native_reverse_check(self):
        source,step=self.fixture(BOX_V,BOX_F[[0,1,4,5]]);report,_=self.run_api(source,step,'partial',True)
        self.assertFalse(report['passed']);self.assertGreater(report['actual_native_boundary_to_source']['max_mm'],1)
        self.assertEqual(len(report['native_face_coverage']),6)

    def test_a_point_inside_the_real_trim_hole_is_not_accepted_on_supporting_plane(self):
        shape=cq.Solid.makeBox(4,4,1).cut(cq.Solid.makeCylinder(1,1,cq.Vector(2,2,0)))
        angle=np.array([-.003,-.001,.001,.003]);radius=.9998
        vertices=np.c_[2+radius*np.cos(angle),2+radius*np.sin(angle),np.ones(4)]
        source,step=self.fixture(vertices,np.array([[0,1,2],[0,2,3],[0,1,3],[1,2,3]]),shape);report,cache=self.run_api(source,step,'hole',True)
        self.assertFalse(report['passed']);self.assertFalse(cache['source_uv_witness_used'][:3].any())
        self.assertGreater(cache['source_to_native_distances'][:3].min(),.00019)
        np.testing.assert_allclose(np.linalg.norm(cache['source_projection_points'][:3,:2]-[2,2],axis=1),1,atol=1e-7)

    def test_unusable_hints_run_the_unchanged_original_projection(self):
        source,step=self.fixture();old,oc=self.run_api(source,step,'old',False)
        with patch('guided_mesh_cad.boundary_validation._native_uv_triangle_hint',return_value=None):
            report,cache=self.run_api(source,step,'unusable',True)
        self.assertEqual(report['source_uv_witness']['accepted_actual_in_trim_points'],0)
        self.assertEqual(report['source_uv_witness']['hint_rejections']['unusable_uv_hint'],len(cache['source_sample_points']))
        np.testing.assert_array_equal(oc['source_projection_points'],cache['source_projection_points'])
        self.assertEqual(old['passed'],report['passed'])

    def test_optional_hint_exception_runs_original_projection_instead_of_aborting(self):
        source,step=self.fixture();old,oc=self.run_api(source,step,'old',False)
        with patch('guided_mesh_cad.boundary_validation._native_uv_triangle_hint',side_effect=ValueError('optional seed failure')):
            report,cache=self.run_api(source,step,'seed_error',True)
        self.assertTrue(report['passed']);self.assertFalse(cache['source_uv_witness_used'].any())
        self.assertEqual(report['source_uv_witness']['hint_rejections']['native_evaluation_error'],len(cache['source_sample_points']))
        self.assertEqual(report['source_uv_witness']['hint_rejections']['unusable_uv_hint'],0)
        np.testing.assert_array_equal(oc['source_projection_points'],cache['source_projection_points'])

    def test_outside_uv_seed_cannot_accept_or_drop_source_points(self):
        source,step=self.fixture()
        with patch('guided_mesh_cad.boundary_validation._native_uv_triangle_hint',return_value=np.array([1e6,1e6])):
            report,cache=self.run_api(source,step,'outside',True)
        self.assertTrue(report['passed']);self.assertFalse(cache['source_uv_witness_used'].any())
        self.assertEqual(len(cache['source_sample_points']),8+12+100)
        self.assertEqual(report['source_uv_witness']['hint_rejections']['outside_or_on_trim'],120)

    def test_native_tolerance_guard_still_rejects_before_witnesses(self):
        shape=cq.Solid.makeBox(2,2,2);BRep_Builder().UpdateVertex(shape.Vertices()[0].wrapped,.02)
        source,step=self.fixture(shape=shape)
        # STEP export normalizes this artificial vertex tolerance. Test the
        # actual reader-result guard explicitly instead of falsely labelling
        # the normalized STEP a bad input.
        from guided_mesh_cad.cad import shape_check_report
        self.assertGreater(shape_check_report(shape)['native_tolerance_max_mm']['vertex'],.01)
        with patch('guided_mesh_cad.cad.read_step_unchecked',return_value=shape):
            with self.assertRaises(ValueError):validate_native_boundary_uv_witness(source,step,self.root/'bad.json',samples=100)

    def test_distinct_entry_point_rejects_strategy_override(self):
        with self.assertRaisesRegex(ValueError,'fixed'):
            validate_native_boundary_uv_witness('source','native','report',_uv_witness=False)

    def test_hint_interpolates_native_uv_without_modifying_arrays(self):
        v=np.array([[0,0,0],[2,0,0],[0,4,0]],float);f=np.array([[0,1,2]]);uv=np.array([[1,2],[5,2],[1,10]],float);before=v.copy()
        np.testing.assert_allclose(_native_uv_triangle_hint(v,f,uv,0,[.5,1,0]),[2,4]);np.testing.assert_array_equal(before,v)

    def test_degenerate_or_invalid_hint_is_only_rejected_not_fitted(self):
        v=np.array([[0,0,0],[1,0,0],[2,0,0]],float);f=np.array([[0,1,2]]);uv=np.array([[0,0],[1,0],[2,0]],float)
        self.assertIsNone(_native_uv_triangle_hint(v,f,uv,0,[0,0,0]))
        v[2]=[0,1,0];self.assertIsNone(_native_uv_triangle_hint(v,f,uv,0,[10,10,0]))

    def test_analytical_twisted_bilinear_sides_cover_nonplanar_native_surfaces(self):
        bottom=np.array([[-2,-1,0],[2,-1,0],[2,1,0],[-2,1,0]],float)
        top=np.array([[-1.2,-.7,2],[1.2,-.7,2],[1.2,.7,2],[-1.2,.7,2]],float)
        rotation=np.array([[np.cos(.4),-np.sin(.4)],[np.sin(.4),np.cos(.4)]])
        top[:,:2]=top[:,:2]@rotation.T+[.3,.2]
        wires=[cq.Wire.makePolygon([cq.Vector(*p) for p in ring],close=True) for ring in [bottom,top]]
        shape=cq.Solid.makeLoft(wires,ruled=True)
        points=[];triangles=[];side=25
        for face in range(4):
            offset=len(points);following=(face+1)%4
            for t in np.linspace(0,1,side):
                for s in np.linspace(0,1,side):
                    points.append((1-t)*((1-s)*bottom[face]+s*bottom[following])+t*((1-s)*top[face]+s*top[following]))
            for i in range(side-1):
                for j in range(side-1):
                    a=offset+i*side+j;triangles.extend([[a,a+1,a+side+1],[a,a+side+1,a+side]])
        for ring in [bottom,top]:
            offset=len(points);points.extend(ring);triangles.extend([[offset,offset+1,offset+2],[offset,offset+2,offset+3]])
        source,step=self.fixture(np.array(points),np.array(triangles),shape);report,_=self.run_api(source,step,'bilinear',True)
        self.assertTrue(report['passed']);self.assertTrue(any(face['surface_type']=='BSPLINE' for face in report['native_face_coverage']))
        self.assertGreater(report['source_uv_witness']['accepted_actual_in_trim_points'],100)

    def test_translated_analytic_sphere_retains_actual_world_coordinates(self):
        from scipy.spatial import ConvexHull
        count=4000;i=np.arange(count);z=1-2*(i+.5)/count;phi=i*np.pi*(3-np.sqrt(5));xy=np.sqrt(1-z*z)
        center=np.array([23.,-17.,5.]);vertices=2*np.c_[xy*np.cos(phi),xy*np.sin(phi),z]+center
        faces=ConvexHull(vertices).simplices;shape=cq.Solid.makeSphere(2,angleDegrees1=-90,angleDegrees2=90).moved(cq.Location(cq.Vector(*center)))
        source,step=self.fixture(vertices,faces,shape);report,cache=self.run_api(source,step,'sphere',True)
        self.assertTrue(report['passed']);flags=cache['source_uv_witness_used']
        self.assertGreater(flags.sum(),100)
        np.testing.assert_allclose(np.linalg.norm(cache['source_projection_points'][flags]-center,axis=1),2,atol=1e-7)


if __name__=='__main__':unittest.main()
