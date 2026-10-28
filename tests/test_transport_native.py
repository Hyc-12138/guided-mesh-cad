"""Small real native transport fixtures; no case models or external files."""
import gzip
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cadquery as cq

from guided_mesh_cad.native_witness import (ActualNativeWitnessOracle, PeriodicEdgeWitness,
    complete_boundary_distance, file_binding, sample_native_boundary)
from guided_mesh_cad.native_transport import (assemble, compare_native, identity_matrix,
    referenced_inputs, verify_assembly)
from guided_mesh_cad.representation import (compare_mass_properties, diagnose_representation,
    repair_representation, representation_hash, restrict_active_bspline)


class NativeTransportFixtures(unittest.TestCase):
    def setUp(self):
        parent = Path(__file__).resolve().parents[1]/'native_test_artifacts'
        parent.mkdir(exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(prefix='fixture_', dir=parent)
        self.root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def shape_file(self, shape, name='source.brep'):
        path = self.root/name
        if path.suffix == '.step': cq.exporters.export(shape, str(path))
        else: shape.exportBrep(str(path))
        return path

    def plan(self, shape=None, two=True):
        shape = shape if shape is not None else cq.Solid.makeBox(2,3,4).moved(cq.Location(cq.Vector(2,3,4)))
        source = self.shape_file(shape)
        rotation = identity_matrix(); c, s = math.cos(.7), math.sin(.7)
        rotation[0][:2] = [c,-s]; rotation[1][:2] = [s,c]
        rotation[0][3], rotation[1][3], rotation[2][3] = 12,-7,9
        other = identity_matrix(); other[0][3] = -6
        document = {'schema':'guided-mesh-cad.assembly.v1', 'units':'mm',
            'resources':[{'id':'body', 'path':source.name, 'sha256':file_binding(source)['sha256'],
                          'scope':{'acceptance':'finite source witness only', 'fullBOP':False}}],
            'instances':[{'id':'first', 'resource_id':'body', 'matrix':rotation}],
            'excluded':[{'id':'optional_pin', 'basis':'explicit fixture exclusion'}]}
        if two: document['instances'].append({'id':'second', 'resource_id':'body', 'matrix':other})
        path = self.root/'plan.json'; path.write_text(json.dumps(document), encoding='utf-8')
        return path, document, source

    def test_real_hole_and_solid_interior_are_not_zero(self):
        ring = cq.Workplane('XY').rect(10,10).circle(2).extrude(2).val()
        hole = complete_boundary_distance(ring, [0,0,1])
        self.assertGreater(hole['distance_mm'], 1.9)
        interior = complete_boundary_distance(cq.Solid.makeBox(2,2,2), [1,1,1])
        self.assertAlmostEqual(interior['distance_mm'], 1., places=8)
        # A point in the top face's hole is outside its actual trim as well.
        top = max(ring.Faces(), key=lambda f:f.Center().z)
        self.assertGreater(complete_boundary_distance(top,[0,0,2])['distance_mm'], 1.9)

    def test_owned_certificates_recompute_actual_XYZ_and_reject_wrong_leaf(self):
        shape = cq.Solid.makeBox(2,2,2); source = self.shape_file(shape)
        oracle = ActualNativeWitnessOracle(shape, {'leaf_name':'first', 'native_source':file_binding(source)},
                                           self.root/'cloud.json.gz')
        points = [[0,0,0], [0,0,0], [.7,.4,0], [1,1,1]]
        result = oracle.batch(points, self.root/'cert.json.gz','first')
        self.assertEqual(result['over_tolerance'], 1)
        self.assertEqual(result['exact_query_reuses'], 1)
        self.assertGreaterEqual(result['complete_boundary_fallbacks'], 2)
        with gzip.open(self.root/'cert.json.gz','rt',encoding='utf-8') as stream: evidence=json.load(stream)
        for row in evidence['records']:
            if 'actual_owned_point' in row:
                self.assertAlmostEqual(row['distance_mm'], math.dist(points[row['query_index']],row['actual_owned_point']))
        with self.assertRaisesRegex(ValueError,'cross-leaf'):
            oracle.batch([[0,0,0]],self.root/'bad.json.gz','second')

    def test_periodic_seam_trim_complement_and_true_offset(self):
        edge = cq.Edge.makeCircle(2, angle1=300, angle2=420)
        oracle = PeriodicEdgeWitness(edge)
        self.assertTrue(oracle.edges)
        point = [2*math.cos(math.radians(330)),2*math.sin(math.radians(330)),0]
        actual = oracle.query(point)
        self.assertLess(actual['distance_mm'], 1e-8)
        self.assertGreater(oracle.query([-2,0,0])['distance_mm'], 1.)
        self.assertGreater(oracle.query([point[0],point[1],.02])['distance_mm'], .019)
        self.assertTrue(actual['actual_trim'][0] <= actual['original_trim_parameter'] <= actual['actual_trim'][1])

    def test_offset_and_invalid_topology_rejected(self):
        box = cq.Solid.makeBox(2,2,2)
        source = self.shape_file(box)
        shifted = self.shape_file(box.translate((.02,0,0)),'shifted.brep')
        result = compare_native(source,shifted,self.root/'offset')
        self.assertFalse(result['ok'])
        self.assertGreater(result['report']['forward']['max_distance_mm'], .019)
        open_face = self.shape_file(box.Faces()[0],'sheet.brep')
        self.assertFalse(compare_native(open_face,source,self.root/'sheet')['ok'])
        self.assertFalse(compare_native(source,source,self.root/'loose',tolerance_mm=.02)['ok'])

    def test_real_freeform_STEP_roundtrip_positive(self):
        points = [[cq.Vector(i,j,.12*math.sin(i)*math.cos(j)) for j in range(4)] for i in range(4)]
        face = cq.Face.makeSplineApprox(points,tol=1e-6,minDeg=3,maxDeg=3)
        solid = face.thicken(.3)
        self.assertTrue(solid.isValid())
        source = self.shape_file(solid)
        step = self.shape_file(solid,'freeform.step')
        result = compare_native(source,step,self.root/'freeform')
        self.assertTrue(result['ok'], result['report'])
        self.assertFalse(result['report']['strict_material_equality_proven'])
        self.assertFalse(result['report']['continuous_surface_equivalence_proven'])

    def test_assembly_named_locations_resume_and_tamper(self):
        path, document, source = self.plan()
        self.assertEqual(referenced_inputs('assemble',{'plan':str(path)}), [str(source.resolve())])
        build = assemble(path,self.root/'build')
        self.assertTrue(build['ok'],build['report'])
        progress=json.loads((self.root/'build'/'progress.json').read_text(encoding='utf-8'))
        self.assertEqual(progress['schema'],'guided-mesh-cad.progress.v1')
        self.assertTrue(progress['snapshot_only'])
        self.assertEqual((progress['completed'],progress['total']),(2,2))
        checkpoint = self.root/'checkpoints'
        first = verify_assembly(path,build['artifacts']['manifest'],self.root/'verify1', checkpoint_dir=str(checkpoint))
        self.assertTrue(first['ok'],first['report'])
        progress=json.loads((self.root/'verify1'/'progress.json').read_text(encoding='utf-8'))
        self.assertEqual((progress['completed'],progress['total'],progress['current_instance']),(2,2,'second'))
        self.assertEqual(first['report']['freshly_checked_instances'],2)
        second = verify_assembly(path,build['artifacts']['manifest'],self.root/'verify2',checkpoint_dir=str(checkpoint))
        self.assertTrue(second['ok'],second['report'])
        self.assertEqual(second['report']['reused_instances'],2)
        record = json.loads(Path(first['report']['records'][0]['record']['path']).read_text(encoding='utf-8'))
        evidence = Path(record['forward']['certificate_evidence']['path'])
        with evidence.open('ab') as stream:stream.write(b'tampered')
        third = verify_assembly(path,build['artifacts']['manifest'],self.root/'verify3',checkpoint_dir=str(checkpoint))
        self.assertTrue(third['ok'],third['report'])
        self.assertEqual(third['report']['reused_instances'],1)
        self.assertEqual(third['report']['freshly_checked_instances'],1)
        wrong = compare_native(source,build['artifacts']['step'],self.root/'wrong_leaf',target_instance='second')
        self.assertFalse(wrong['ok'])
        missing = compare_native(source,build['artifacts']['step'],self.root/'missing_leaf',target_instance='absent')
        self.assertFalse(missing['ok'])
        self.assertFalse(first['report']['whole_assembly_accepted'])

    def test_interrupted_verification_reuses_only_finished_instance(self):
        import guided_mesh_cad.native_transport as transport
        path, _, _ = self.plan(); build = assemble(path,self.root/'build')
        self.assertTrue(build['ok'],build['report'])
        original, calls = transport._compare_shapes, []
        def interrupted(*args, **kwargs):
            calls.append(True)
            if len(calls) == 2: raise RuntimeError('simulated process interruption between leaves')
            return original(*args, **kwargs)
        checkpoint = self.root/'checkpoints'
        with patch.object(transport,'_compare_shapes',side_effect=interrupted):
            first = verify_assembly(path,build['artifacts']['manifest'],self.root/'interrupted',checkpoint_dir=str(checkpoint))
        self.assertFalse(first['ok'])
        second = verify_assembly(path,build['artifacts']['manifest'],self.root/'continued',checkpoint_dir=str(checkpoint))
        self.assertTrue(second['ok'],second['report'])
        self.assertEqual(second['report']['reused_instances'],1)
        self.assertEqual(second['report']['freshly_checked_instances'],1)

    def test_binding_changes_and_unowned_checkpoint_rejected(self):
        path, document, _ = self.plan(two=False); build=assemble(path,self.root/'build')
        self.assertTrue(build['ok'],build['report'])
        unowned = self.root/'unowned';unowned.mkdir();(unowned/'foreign').write_text('foreign')
        self.assertFalse(verify_assembly(path,build['artifacts']['manifest'],self.root/'verify',checkpoint_dir=str(unowned))['ok'])
        self.assertFalse(verify_assembly(path,build['artifacts']['manifest'],self.root/'outside',checkpoint_dir=str(self.root.parent))['ok'])
        document['resources'][0]['scope']['fullBOP']=True
        path.write_text(json.dumps(document),encoding='utf-8')
        self.assertFalse(verify_assembly(path,build['artifacts']['manifest'],self.root/'changed_scope')['ok'])

    def test_nonrigid_duplicate_and_missing_hash_rejected(self):
        path, document, _=self.plan(two=False)
        document['instances'][0]['matrix'][0][0]=2
        path.write_text(json.dumps(document),encoding='utf-8')
        self.assertFalse(assemble(path,self.root/'scale')['ok'])
        document['instances'][0]['matrix']=identity_matrix()
        document['instances'].append(dict(document['instances'][0]))
        path.write_text(json.dumps(document),encoding='utf-8')
        self.assertFalse(assemble(path,self.root/'duplicate')['ok'])
        document['instances'].pop();document['resources'][0].pop('sha256')
        path.write_text(json.dumps(document),encoding='utf-8')
        self.assertFalse(assemble(path,self.root/'hash')['ok'])

    def test_unknown_parameters_fail_before_native_work(self):
        # Missing paths would produce a different error if parameter validation
        # happened after input/CAD work. Typos must never become tolerance passes.
        result=compare_native('missing-source','missing-target',self.root/'unknown_compare',tolerence_mm=.0001)
        self.assertFalse(result['ok'])
        self.assertIn('unsupported native parameter',result['report']['errors'][0]['message'])
        result=assemble('missing-plan',self.root/'unknown_build',samples=1)
        self.assertFalse(result['ok'])
        self.assertIn('unsupported native parameter',result['report']['errors'][0]['message'])
        result=verify_assembly('missing-plan','missing-manifest',self.root/'unknown_verify',repair=True)
        self.assertFalse(result['ok'])
        self.assertIn('unsupported native parameter',result['report']['errors'][0]['message'])

    def test_representation_C0_active_domain_and_read_only(self):
        from OCP.Geom import Geom_BSplineSurface
        from OCP.TColgp import TColgp_Array2OfPnt
        from OCP.TColStd import TColStd_Array1OfInteger,TColStd_Array1OfReal
        from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeFace
        from OCP.gp import gp_Pnt
        poles=TColgp_Array2OfPnt(1,3,1,2)
        for i in range(1,4):
            for j in range(1,3):poles.SetValue(i,j,gp_Pnt(i-1,j-1,.2 if i==2 else 0))
        def array(values,kind):
            result=kind(1,len(values))
            for i,v in enumerate(values,1):result.SetValue(i,v)
            return result
        surface=Geom_BSplineSurface(poles,array([0.,.5,1.],TColStd_Array1OfReal),
            array([0.,1.],TColStd_Array1OfReal),array([2,1,2],TColStd_Array1OfInteger),
            array([2,2],TColStd_Array1OfInteger),1,1)
        face=cq.Face(BRepBuilderAPI_MakeFace(surface,0.,1.,0.,1.,1e-7).Face())
        trimmed=cq.Face(BRepBuilderAPI_MakeFace(surface,0.,.4,0.,1.,1e-7).Face())
        before=representation_hash(face)
        self.assertEqual(len(diagnose_representation(face)['faces'][0]['C0_knots']),1)
        diagnostic=diagnose_representation(trimmed)['faces'][0]
        self.assertEqual(diagnostic['C0_knots'],[])
        self.assertEqual(len(diagnostic['support_C0_knots_outside_active_domain']),1)
        self.assertEqual(before,representation_hash(face))
        mass=compare_mass_properties(cq.Solid.makeBox(2,2,2),cq.Solid.makeBox(2,2,2))
        self.assertEqual(len(mass['source']['levels']),3)
        self.assertFalse(mass['strict_material_equality_proven'])
        with self.assertRaises(NotImplementedError):repair_representation(face)

    def test_workflow_success_evidence_tamper_invalidates_outer_receipt(self):
        """Exercise the actual guarded worker/run/resume path, not direct API reentry."""
        from guided_mesh_cad.workflow import run, resume
        assembly_plan, _, _ = self.plan()
        plan={'schema':'guided-mesh-cad.workflow.v1','tasks':[
            {'id':'build','operation':'assemble','inputs':{'plan':str(assembly_plan)},
             'params':{},'depends_on':[],'scope':'fixture named export only'},
            {'id':'verify','operation':'verify-assembly','inputs':{'plan':str(assembly_plan),
                'build_manifest':{'task':'build','artifact':'manifest'}},
             'params':{},'depends_on':['build'],'scope':'fixture finite transport only'}]}
        plan_path=self.root/'workflow.json';plan_path.write_text(json.dumps(plan),encoding='utf-8')
        run_dir=self.root/'run'
        first=run(plan_path,run_dir)
        self.assertTrue((run_dir/'checkpoint.json').is_file(), first)
        state=json.loads((run_dir/'checkpoint.json').read_text(encoding='utf-8'))
        self.assertEqual(state['tasks']['verify']['status'],'succeeded',(first,state['tasks']['verify']))
        prior_attempt=state['tasks']['verify']['attempt']
        report=json.loads(Path(state['tasks']['verify']['artifacts']['report']).read_text(encoding='utf-8'))
        record=json.loads(Path(report['records'][0]['record']['path']).read_text(encoding='utf-8'))
        evidence=Path(record['forward']['certificate_evidence']['path'])
        with evidence.open('ab') as stream:stream.write(b'tampered after workflow success')
        resumed=resume(run_dir)
        state=json.loads((run_dir/'checkpoint.json').read_text(encoding='utf-8'))
        self.assertEqual(state['tasks']['verify']['status'],'succeeded',(resumed,state['tasks']['verify']))
        self.assertNotEqual(state['tasks']['verify']['attempt'],prior_attempt)
        report=json.loads(Path(state['tasks']['verify']['artifacts']['report']).read_text(encoding='utf-8'))
        self.assertEqual(report['reused_instances'],1)
        self.assertEqual(report['freshly_checked_instances'],1)
        self.assertEqual(resumed['reused_tasks'],1)


class NativeBSplineRestrictionFixtures(unittest.TestCase):
    @staticmethod
    def array(values, kind):
        result=kind(1,len(values))
        for i,value in enumerate(values,1):result.SetValue(i,value)
        return result

    def curve(self, periodic=False):
        from OCP.Geom import Geom_BSplineCurve
        from OCP.TColgp import TColgp_Array1OfPnt
        from OCP.TColStd import TColStd_Array1OfInteger,TColStd_Array1OfReal
        from OCP.gp import gp_Pnt
        poles=TColgp_Array1OfPnt(1,3)
        for i,p in enumerate([(0,0,0),(1,.3,0),(2,0,.2)],1):poles.SetValue(i,gp_Pnt(*p))
        knots,mults=([0.,1.,2.,3.],[1,1,1,1]) if periodic else ([0.,.5,1.],[2,1,2])
        return Geom_BSplineCurve(poles,self.array(knots,TColStd_Array1OfReal),
            self.array(mults,TColStd_Array1OfInteger),1,periodic)

    def surface(self, periodic=False):
        from OCP.Geom import Geom_BSplineSurface
        from OCP.TColgp import TColgp_Array2OfPnt
        from OCP.TColStd import TColStd_Array1OfInteger,TColStd_Array1OfReal
        from OCP.gp import gp_Pnt
        poles=TColgp_Array2OfPnt(1,3,1,2)
        for i in range(1,4):
            for j in range(1,3):poles.SetValue(i,j,gp_Pnt(i-1,j-1,.2 if i==2 else 0))
        knots,mults=([0.,1.,2.,3.],[1,1,1,1]) if periodic else ([0.,.5,1.],[2,1,2])
        return Geom_BSplineSurface(poles,self.array(knots,TColStd_Array1OfReal),
            self.array([0.,1.],TColStd_Array1OfReal),self.array(mults,TColStd_Array1OfInteger),
            self.array([2,2],TColStd_Array1OfInteger),1,1,periodic,False)

    @staticmethod
    def controls(geometry):
        from guided_mesh_cad.representation import _bspline_control_data
        return _bspline_control_data(geometry)

    def test_curve_outside_C0_native_segment_copy_and_input_unchanged(self):
        source=self.curve();before=self.controls(source)
        restricted,report=restrict_active_bspline(source,[.05,.4])
        self.assertEqual(restricted.NbKnots(),2)
        self.assertEqual(source.NbKnots(),3)
        self.assertEqual(restricted.Degree(),source.Degree())
        self.assertEqual(self.controls(source),before)
        self.assertEqual(report['original_control_data_sha256'],before[1])
        self.assertEqual(report['restricted_control_data_sha256'],self.controls(restricted)[1])
        self.assertEqual(report['restricted_bounds'],[.05,.4])
        self.assertLess(report['finite_point_check']['maximum_distance_mm'],1e-12)
        self.assertEqual(report['finite_point_check']['samples'],33)
        self.assertFalse(report['caller_complete_usage_domain_verified'])
        self.assertFalse(report['continuous_geometry_equivalence_proven'])
        self.assertFalse(report['BRep_topology_replaced'])

    def test_surface_outside_C0_native_segment_copy_and_input_unchanged(self):
        source=self.surface();before=self.controls(source)
        restricted,report=restrict_active_bspline(source,[.05,.4,.1,.9])
        self.assertEqual(restricted.NbUKnots(),2)
        self.assertEqual(source.NbUKnots(),3)
        self.assertEqual(restricted.Bounds(),(.05,.4,.1,.9))
        self.assertEqual(self.controls(source),before)
        self.assertLess(report['finite_point_check']['maximum_distance_mm'],1e-12)
        self.assertEqual(report['finite_point_check']['samples'],1089)
        self.assertFalse(report['fitting_performed'])
        self.assertFalse(report['strict_material_equality_proven'])

    def test_native_restriction_retains_inside_C0_and_rational_geometry(self):
        source=self.curve();source.SetWeight(2,1.5);before=self.controls(source)
        restricted,report=restrict_active_bspline(source,[.2,.8])
        self.assertTrue(restricted.IsRational())
        self.assertEqual(restricted.NbKnots(),3)
        self.assertEqual(restricted.Knot(2),.5)
        self.assertEqual(restricted.Multiplicity(2),restricted.Degree())
        self.assertEqual(self.controls(source),before)
        self.assertLess(report['finite_point_check']['maximum_distance_mm'],1e-12)
        self.assertFalse(report['fitting_performed'])

    def test_bounds_invalid_inputs_and_periodic_rejected_without_mutation(self):
        source=self.curve();before=self.controls(source)
        for bounds in (None,[],[0,1,0,1],[0,0],[.7,.2],[-.1,.4],[0,1.0000001],
                       [0,float('nan')],[False,.4],['0',.4]):
            with self.subTest(bounds=bounds),self.assertRaises(ValueError):
                restrict_active_bspline(source,bounds)
            self.assertEqual(self.controls(source),before)
        surface=self.surface();surface_before=self.controls(surface)
        for bounds in ([0,1],[0,1,1,.5],[0,1,0,1.01],[0,.4,0,float('inf')]):
            with self.subTest(bounds=bounds),self.assertRaises(ValueError):
                restrict_active_bspline(surface,bounds)
            self.assertEqual(self.controls(surface),surface_before)
        for geometry,bounds in ((self.curve(True),[.2,.8]),(self.surface(True),[.2,.8,0,1])):
            original=self.controls(geometry)
            with self.assertRaisesRegex(ValueError,'periodic'):
                restrict_active_bspline(geometry,bounds)
            self.assertEqual(self.controls(geometry),original)
        for tolerance in (0,.02,float('nan'),True):
            with self.assertRaises(ValueError):restrict_active_bspline(source,[0,.4],tolerance)
        with self.assertRaises(TypeError):restrict_active_bspline(cq.Solid.makeBox(1,1,1),[0,1])
        with self.assertRaises(NotImplementedError):repair_representation(source)


if __name__ == '__main__':unittest.main()
