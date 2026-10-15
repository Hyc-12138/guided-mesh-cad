import json,tempfile,unittest,hashlib
from pathlib import Path
from unittest.mock import patch
import numpy as np
import cadquery as cq
from OCP.gp import gp_Pnt
from guided_mesh_cad import validate_native_boundary_low_memory,validate_native_boundary_uv_witness
from guided_mesh_cad.memory_validation import NativeArrayStore,mapped_native_sampling,mapped_mesh_hints,owned_extrema_point
from guided_mesh_cad.mesh import sha256

V=np.array([[0,0,0],[2,0,0],[2,2,0],[0,2,0],[0,0,2],[2,0,2],[2,2,2],[0,2,2]],float)
F=np.array([[0,2,1],[0,3,2],[4,5,6],[4,6,7],[0,1,5],[0,5,4],[1,2,6],[1,6,5],[2,3,7],[2,7,6],[3,0,4],[3,4,7]],np.int64)

class LowMemoryValidationTests(unittest.TestCase):
    def setUp(self):self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
    def tearDown(self):self.tmp.cleanup()
    def fixture(self,vertices=V,triangles=F,shape=None):
        source=self.root/'source.npz';step=self.root/'native.step';np.savez_compressed(source,vertices=vertices,faces=triangles);cq.exporters.export(cq.Solid.makeBox(2,2,2) if shape is None else shape,str(step));return source,step
    def run_api(self,source,step,name='mapped'):
        return validate_native_boundary_low_memory(source,step,self.root/(name+'.json'),samples=100,diagnostics_dir=self.root/name)
    def arrays(self,report):
        d=report['diagnostics'];arrays={k:np.load(v['path']) for section in ['native_uv_tessellation_arrays','native_actual_sample_arrays'] for k,v in d[section].items()}
        with np.load(d['source_actual_boundary_samples']) as data:arrays.update({k:data[k].copy() for k in data.files})
        return arrays

    def test_cube_mapped_arrays_have_full_source_native_coverage_and_hashes(self):
        source,step=self.fixture();before=step.read_bytes();r=self.run_api(source,step);a=self.arrays(r)
        self.assertTrue(r['passed']);self.assertEqual(step.read_bytes(),before)
        self.assertEqual(r['verification_method'],'actual_trimmed_native_boundary_uv_witness_low_memory')
        self.assertFalse(r['execution_strategy']['mesh_parallel'])
        self.assertEqual(len(a['source_sample_points']),8+12+100)
        self.assertEqual(len(a['native_actual_points']),len(a['uv_nodes'])+len(a['faces'])+100+12*33+8)
        np.testing.assert_allclose(np.linalg.norm(a['source_projection_points']-a['source_sample_points'],axis=1),a['source_to_native_distances'],atol=1e-14)
        for section in ['native_uv_tessellation_arrays','native_actual_sample_arrays']:
            for entry in r['diagnostics'][section].values():self.assertEqual(sha256(entry['path']),entry['sha256'])

    def test_cube_reverse_arrays_match_previous_complete_in_memory_strategy(self):
        source,step=self.fixture();old=validate_native_boundary_uv_witness(source,step,self.root/'old.json',samples=100,diagnostics_dir=self.root/'old');r=self.run_api(source,step);a=self.arrays(r)
        with np.load(self.root/'old/actual_boundary_samples.npz') as data:
            np.testing.assert_array_equal(a['source_sample_points'],data['source_sample_points'])
            np.testing.assert_allclose(a['native_actual_points'],data['native_actual_points'],atol=1e-14)
            np.testing.assert_allclose(a['native_to_source_distances'],data['native_to_source_distances'],atol=1e-14)
        self.assertEqual(r['native_face_coverage'],old['native_face_coverage'])

    def test_shifted_source_fails_without_reducing_samples(self):
        v=V.copy();v[4:,2]+=.025;source,step=self.fixture(v);r=self.run_api(source,step)
        self.assertFalse(r['passed']);self.assertGreater(r['source_to_actual_native_boundary_upper_bounds']['max_mm'],.0249)
        self.assertEqual(r['source_to_actual_native_boundary_upper_bounds']['count'],120)

    def test_missing_source_faces_still_fail_reverse(self):
        source,step=self.fixture(V,F[[0,1,4,5]]);r=self.run_api(source,step)
        self.assertFalse(r['passed']);self.assertGreater(r['actual_native_boundary_to_source']['max_mm'],1.)
        self.assertEqual(len(r['native_face_coverage']),6)

    def test_curved_hole_and_translated_native_shape_are_kept(self):
        shape=cq.Solid.makeBox(4,4,1).cut(cq.Solid.makeCylinder(1,1,cq.Vector(2,2,0))).translate((7,-3,2));v,f=shape.tessellate(.0005,.03)
        source,step=self.fixture(np.array([x.toTuple() for x in v]),np.array(f,dtype=np.int64),shape);r=self.run_api(source,step)
        self.assertTrue(r['passed']);self.assertEqual(len(r['native_face_coverage']),len(shape.Faces()))
        self.assertGreater(r['execution_strategy']['owned_support_coordinate_reconstructions'],0)

    def test_hint_failure_falls_back_to_owned_supports_and_actual_norms(self):
        source,step=self.fixture()
        with patch('guided_mesh_cad.boundary_validation._native_uv_triangle_hint',return_value=None):r=self.run_api(source,step)
        a=self.arrays(r);self.assertTrue(r['passed']);self.assertFalse(a['source_uv_witness_used'].any())
        np.testing.assert_allclose(np.linalg.norm(a['source_sample_points']-a['source_projection_points'],axis=1),a['source_to_native_distances'],atol=1e-14)

    def test_cached_cpp_coordinate_cannot_replace_actual_owned_curve_value(self):
        edge=cq.Edge.makeLine(cq.Vector(0,0,0),cq.Vector(1,0,0));owner=cq.Compound.makeCompound([edge])
        class Extrema:
            def SupportOnShape1(self,index):return edge.wrapped
            def ParOnEdgeS1(self,index):return (.5,)
            def PointOnShape1(self,index):return gp_Pnt(.5,.0004,0)
        point=owned_extrema_point(Extrema(),owner);self.assertAlmostEqual(point.Y(),0)
        self.assertGreater(np.linalg.norm(np.array([point.X(),point.Y(),point.Z()])-[.5,.0102,0]),.01)
        wrong=cq.Compound.makeCompound([edge.copy()])
        with self.assertRaisesRegex(ValueError,'owned'):owned_extrema_point(Extrema(),wrong)

    def test_unsafe_or_nonfresh_workspace_is_rejected(self):
        source,step=self.fixture()
        with self.assertRaisesRegex(ValueError,'diagnostics'):validate_native_boundary_low_memory(source,step,self.root/'bad.json',samples=100)
        with self.assertRaisesRegex(ValueError,'boolean'):validate_native_boundary_low_memory(source,step,self.root/'bad2.json',samples=100,diagnostics_dir=self.root/'bad2',mesh_parallel=0)
        self.run_api(source,step)
        with self.assertRaisesRegex(ValueError,'already exists'):self.run_api(source,step)

    def test_stream_hash_matches_standard_bytes_for_large_file(self):
        p=self.root/'data.bin';payload=bytes(range(256))*9000;p.write_bytes(payload)
        self.assertEqual(sha256(p),hashlib.sha256(payload).hexdigest())

    def test_array_handles_release_when_a_stage_fails(self):
        source,step=self.fixture()
        with patch('guided_mesh_cad.memory_validation.mapped_native_sampling',side_effect=RuntimeError('injected sample-stage failure')):
            with self.assertRaisesRegex(RuntimeError,'sample-stage'):self.run_api(source,step)
        # Removing only this test's known individual array proves no mapped
        # handle remains; it does not turn an interrupted run into a PASS.
        path=self.root/'mapped/native_arrays/faces.npy';self.assertTrue(path.exists());path.unlink()
        self.assertFalse((self.root/'mapped.json').exists())

    def test_mapped_capture_keeps_native_trimmed_curved_face_state_alive(self):
        import gc
        from OCP.BRep import BRep_Tool
        from OCP.TopLoc import TopLoc_Location
        from guided_mesh_cad.memory_validation import native_mesh_uv_mapped
        shape=cq.Workplane('XY').box(4,5,3).edges().fillet(.4).val().translate((7,-3,2))
        # Face handles are acquired before meshing, as in the public verifier.
        faces=shape.Faces();store=NativeArrayStore(self.root/'retained_curved_native')
        try:
            vertices,triangles,uv,node_faces,triangle_faces,counts=native_mesh_uv_mapped(shape,store)
            gc.collect()
            self.assertEqual(len(counts),len(faces));offset=0
            for face,(nodes,facets) in zip(faces,counts):
                mesh=BRep_Tool.Triangulation_s(face.wrapped,TopLoc_Location())
                self.assertIsNotNone(mesh,'Captured native face state must remain live until geometric validation exits')
                self.assertEqual(mesh.NbNodes(),nodes);self.assertEqual(mesh.NbTriangles(),facets)
                surface=BRep_Tool.Surface_s(face.wrapped)
                for i in range(offset,offset+nodes):
                    u,v=map(float,uv[i]);point=surface.Value(u,v)
                    np.testing.assert_allclose([point.X(),point.Y(),point.Z()],vertices[i],atol=1e-8,rtol=0)
                offset+=nodes
            self.assertEqual(offset,len(vertices));self.assertTrue(shape.isValid())
        finally:store.close()

    def test_facewise_hint_search_matches_complete_mesh_distances(self):
        # Separate triangle faces include overlapping boxes and distinct planes.
        from guided_mesh_cad.mesh import mesh_distances
        vertices=np.array([[0,0,0],[2,0,0],[0,2,0],[0,0,1],[2,0,1],[0,2,1],[-1,-1,.4],[3,-1,.4],[-1,3,.4]],float)
        triangles=np.arange(9).reshape(3,3);points=np.random.default_rng(710).uniform([-2,-2,-1],[4,4,2],(120,3))
        closest,cells,squared,coverage=mapped_mesh_hints(points,vertices,triangles,[(3,1)]*3)
        np.testing.assert_allclose(np.sqrt(squared),mesh_distances(points,vertices,triangles),atol=1e-12)
        np.testing.assert_allclose(np.linalg.norm(points-closest,axis=1),np.sqrt(squared),atol=1e-12)
        self.assertEqual(len(coverage),3);self.assertTrue(((cells>=0)&(cells<3)).all())

if __name__=='__main__':unittest.main()
