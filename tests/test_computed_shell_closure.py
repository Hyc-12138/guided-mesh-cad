import unittest
import cadquery as cq
from guided_mesh_cad.cad import shape_check_report,require_solid,PatchTopologyError,sew_solid_with_report

class ComputedShellClosureTests(unittest.TestCase):
    def test_false_cached_flag_does_not_reject_a_closed_shell(self):
        solid=cq.Solid.makeBox(2,3,4);shell=solid.Shells()[0];shell.wrapped.Closed(False)
        check=shape_check_report(solid)
        self.assertFalse(check['shell_closure_checks'][0]['stored_closed_flag'])
        self.assertEqual(check['closed_shells'],1);require_solid(solid)
        self.assertFalse(shell.Closed())

    def test_true_cached_flag_does_not_accept_an_open_shell(self):
        shell=cq.Shell.makeShell(cq.Solid.makeBox(2,3,4).Faces()[:-1]);shell.wrapped.Closed(True)
        check=shape_check_report(shell)
        self.assertTrue(check['shell_closure_checks'][0]['stored_closed_flag'])
        self.assertEqual(check['closed_shells'],0)
        with self.assertRaises(PatchTopologyError):require_solid(cq.Solid.makeSolid(shell))

    def test_seam_and_degenerate_edges_do_not_create_false_openings(self):
        for solid in [cq.Solid.makeSphere(2),cq.Solid.makeCylinder(2,4),cq.Solid.makeTorus(4,1)]:
            with self.subTest(solid=solid.ShapeType()):
                solid.Shells()[0].wrapped.Closed(False)
                self.assertEqual(shape_check_report(solid)['closed_shells'],1);require_solid(solid)

    def test_inner_negative_shell_is_retained(self):
        solid=cq.Solid.makeBox(4,4,4).cut(cq.Solid.makeBox(1,1,1).translate((1,1,1)))
        self.assertEqual(len(solid.Shells()),2)
        for shell in solid.Shells():shell.wrapped.Closed(False)
        require_solid(solid);self.assertEqual(shape_check_report(solid)['closed_shells'],2)

    def test_public_sewing_keeps_formal_solid_guards(self):
        faces=cq.Solid.makeBox(2,3,4).Faces();solid,report=sew_solid_with_report(faces,1e-7)
        require_solid(solid);self.assertEqual(report['closed_shell_count'],1)
        with self.assertRaises(PatchTopologyError):sew_solid_with_report(faces[:-1],1e-7)

if __name__=='__main__':unittest.main()
