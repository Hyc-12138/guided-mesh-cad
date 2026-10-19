import unittest
import numpy as np

from guided_mesh_cad.freeform_control import make_control_face


class NativeControlFaceTests(unittest.TestCase):
    def test_known_bilinear_geometry_in_cubic_basis(self):
        from OCP.BRep import BRep_Tool
        from OCP.BRepAdaptor import BRepAdaptor_Surface
        from OCP.GeomAbs import GeomAbs_C2
        u = np.arange(4)/3
        v = np.arange(4)/3
        grid = np.array([[[a, .3*a+.2*b+.4*a*b, b] for b in v] for a in u])
        before = grid.tobytes()
        face = make_control_face(grid)
        self.assertTrue(face.isValid())
        surface = BRep_Tool.Surface_s(face.wrapped)
        for a in np.linspace(0, 1, 11):
            for b in np.linspace(0, 1, 13):
                expected = [a, .3*a+.2*b+.4*a*b, b]
                np.testing.assert_allclose(surface.Value(float(a), float(b)).Coord(), expected, rtol=0, atol=2e-14)
        adaptor = BRepAdaptor_Surface(face.wrapped)
        self.assertEqual(adaptor.NbUIntervals(GeomAbs_C2), 1)
        self.assertEqual(adaptor.NbVIntervals(GeomAbs_C2), 1)
        self.assertEqual(before, grid.tobytes())

    def test_explicit_poles_are_not_mislabelled_interpolation(self):
        from OCP.BRep import BRep_Tool
        grid = np.array([[[i/6, .2*np.sin(i)+.1*np.cos(j), j/7] for j in range(8)] for i in range(7)])
        face = make_control_face(grid)
        s = BRep_Tool.Surface_s(face.wrapped)
        self.assertEqual((s.NbUPoles(), s.NbVPoles()), (7, 8))
        np.testing.assert_array_equal(s.Pole(4, 5).Coord(), grid[3, 4])
        self.assertGreater(np.linalg.norm(np.array(s.Value(.5, 4/7).Coord())-grid[3, 4]), 1e-4)
        self.assertTrue(all(s.UMultiplicity(k) == 1 for k in range(2, s.NbUKnots())))
        self.assertTrue(all(s.VMultiplicity(k) == 1 for k in range(2, s.NbVKnots())))

    def test_bad_grids_and_tolerances_rejected(self):
        good = np.array([[[i/3, 0., j/3] for j in range(4)] for i in range(4)])
        bad = good.copy()
        bad[1, 1, 0] = np.nan
        for grid in [bad, good[:3], np.zeros((4, 4, 2))]:
            with self.assertRaises(ValueError):
                make_control_face(grid)
        for tolerance in [0, -1, float('nan'), float('inf')]:
            with self.assertRaises(ValueError):
                make_control_face(good, tolerance=tolerance)


if __name__ == '__main__':
    unittest.main()
