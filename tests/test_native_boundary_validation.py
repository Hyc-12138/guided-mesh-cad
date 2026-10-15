from pathlib import Path
import tempfile
import unittest
import numpy as np
import cadquery as cq
from guided_mesh_cad.cad import triangulate
from guided_mesh_cad.boundary_validation import validate_native_boundary


class ActualBoundaryValidationTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.out=Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def inputs(self,source,model):
        v,f=triangulate(source,.001)
        np.savez_compressed(self.out/'source.npz',vertices=v,faces=f)
        cq.exporters.export(model,str(self.out/'model.step'))
        return self.out/'source.npz',self.out/'model.step'

    def test_equal_rotated_box_complete_counts_and_real_points(self):
        box=cq.Solid.makeBox(2,3,4).rotate((0,0,0),(0,0,1),37).translate((11,-8,3))
        src,step=self.inputs(box,box)
        result=validate_native_boundary(src,step,self.out/'report.json',samples=100,
                                        diagnostics_dir=self.out/'evidence')
        self.assertTrue(result['passed'])
        self.assertLess(result['sampled_max_mm'],1e-7)
        self.assertEqual(len(result['native_face_coverage']),6)
        self.assertEqual(result['source_to_actual_native_boundary_upper_bounds']['count'],
                         result['source_vertices']+result['source_triangle_centres']+100)
        expected=sum(r['mesh_uv_nodes']+r['mesh_triangles'] for r in result['native_face_coverage'])+100+result['native_edge_vertex_points']
        self.assertEqual(result['actual_native_boundary_to_source']['count'],expected)
        data=np.load(self.out/'evidence/actual_boundary_samples.npz')
        measured=np.linalg.norm(data['source_sample_points']-data['source_projection_points'],axis=1)
        self.assertTrue(np.allclose(measured,data['source_to_native_distances'],atol=1e-9,rtol=0))

    def test_known_shift_remains_rejected_at_modeling_tolerance(self):
        box=cq.Solid.makeBox(2,3,4)
        src,step=self.inputs(box,box.translate((.025,0,0)))
        result=validate_native_boundary(src,step,self.out/'report.json',samples=100)
        self.assertFalse(result['passed'])
        self.assertAlmostEqual(result['sampled_max_mm'],.025,places=7)
        self.assertFalse(result['source_completeness_and_manufacturing_intent_verified'])

    def test_equal_hole_keeps_trim_and_missing_hole_is_rejected(self):
        box=cq.Solid.makeBox(2,2,2)
        hole=box.cut(cq.Solid.makeCylinder(.35,2,cq.Vector(1,1,0)))
        src,step=self.inputs(hole,hole)
        good=validate_native_boundary(src,step,self.out/'good.json',samples=100)
        self.assertTrue(good['passed'])
        self.assertGreater(good['on_boundary_uv_samples_projected_to_edges'],0)
        cq.exporters.export(box,str(self.out/'missing_hole.step'))
        bad=validate_native_boundary(src,self.out/'missing_hole.step',self.out/'bad.json',samples=100)
        self.assertFalse(bad['passed'])
        self.assertGreater(bad['sampled_max_mm'],.1)

    def test_equal_sphere_samples_true_surface_and_original_mesh_fidelity(self):
        sphere=cq.Solid.makeSphere(2,angleDegrees1=-90,angleDegrees2=90)
        # Independent analytic source avoids the legacy native mesher's zero
        # area pole triangles. The production source validator remains strict.
        from scipy.spatial import ConvexHull
        count=5000;indices=np.arange(count);z=1-2*(indices+.5)/count
        angle=indices*(np.pi*(3-np.sqrt(5)));rad=np.sqrt(1-z*z)
        vertices=2*np.c_[rad*np.cos(angle),rad*np.sin(angle),z]
        faces=ConvexHull(vertices).simplices
        src=self.out/'source.npz';step=self.out/'model.step'
        np.savez_compressed(src,vertices=vertices,faces=faces)
        cq.exporters.export(sphere,str(step))
        result=validate_native_boundary(src,step,self.out/'report.json',samples=100,
                                        diagnostics_dir=self.out/'evidence')
        self.assertTrue(result['passed'])
        data=np.load(self.out/'evidence/actual_boundary_samples.npz')
        radius=np.linalg.norm(data['native_actual_points'],axis=1)
        self.assertLess(float(abs(radius-2).max()),1e-7)

    def test_degenerate_supplied_source_is_still_rejected(self):
        box=cq.Solid.makeBox(2,3,4)
        src,step=self.inputs(box,box)
        data=np.load(src);faces=data['faces'].copy();faces[0]=faces[0,0]
        np.savez_compressed(src,vertices=data['vertices'],faces=faces)
        with self.assertRaises(ValueError):
            validate_native_boundary(src,step,self.out/'report.json',samples=100)

    def test_rejects_unowned_face_without_turning_it_into_a_solid(self):
        box=cq.Solid.makeBox(2,3,4)
        extra=cq.Face.makePlane(.5,.5,cq.Vector(.5,.5,.5))
        src,step=self.inputs(box,cq.Compound.makeCompound([box,extra]))
        with self.assertRaises(ValueError):
            validate_native_boundary(src,step,self.out/'report.json',samples=100)
        self.assertFalse((self.out/'report.json').exists())

    def test_frozen_report_and_diagnostic_directory_are_not_overwritten(self):
        box=cq.Solid.makeBox(2,3,4)
        src,step=self.inputs(box,box)
        report=self.out/'frozen.json';report.write_bytes(b'original evidence')
        before=report.stat().st_mtime_ns
        with self.assertRaises(ValueError):
            validate_native_boundary(src,step,report,samples=100)
        self.assertEqual(report.read_bytes(),b'original evidence')
        self.assertEqual(report.stat().st_mtime_ns,before)
        evidence=self.out/'evidence';evidence.mkdir();(evidence/'frozen').write_bytes(b'keep')
        with self.assertRaises(ValueError):
            validate_native_boundary(src,step,self.out/'new.json',samples=100,diagnostics_dir=evidence)
        self.assertEqual((evidence/'frozen').read_bytes(),b'keep')


if __name__=='__main__':unittest.main()
