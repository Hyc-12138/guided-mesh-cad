import unittest
import numpy as np
from guided_mesh_cad import simplify_corresponding_polylines as simplify


class BoundedPolylinesTests(unittest.TestCase):
    def test_straight_nonuniform_grid_reduces_to_endpoints(self):
        u = np.array([0., .01, .13, .7, 1.])
        grid = np.stack([np.c_[u, 2*u, 3+u], np.c_[2-u, 1+4*u, u]], axis=1)
        reduced, params, report = simplify(grid, u, 1e-12)
        np.testing.assert_array_equal(reduced, grid[[0, -1]])
        np.testing.assert_array_equal(params, [0, 1])
        self.assertEqual(report['retained_node_count'], 2)

    def test_one_station_feature_protects_all_correspondence(self):
        u = np.linspace(0, 1, 101)
        grid = np.stack([np.c_[u, u*0, u*0], np.c_[u, u*0, u*0]], axis=1)
        grid[43, 1, 2] = .2
        result, params, report = simplify(grid, u, .001)
        self.assertTrue({42, 43, 44}.issubset(report['retained_indices']))
        np.testing.assert_array_equal(result[:, 0, 0], params)

    def test_dense_independent_parameter_probes_respect_bound(self):
        u = np.linspace(0, 1, 181)**1.7
        grid = np.stack([np.c_[u, np.sin(7*u), .2*np.cos(13*u)],
                         np.c_[u+u*u, .6*np.cos(5*u), np.sin(11*u)]], axis=1)
        reduced, pu, report = simplify(grid, u, .002, required_indices=[11, 74])
        self.assertTrue({11, 74}.issubset(report['retained_indices']))
        dense = np.random.default_rng(8394).uniform(0, 1, 9001)
        for station in range(2):
            original = np.c_[[np.interp(dense, u, grid[:, station, k]) for k in range(3)]].T
            sampled = np.c_[[np.interp(dense, pu, reduced[:, station, k]) for k in range(3)]].T
            self.assertLessEqual(np.linalg.norm(original-sampled, axis=1).max(), .002)
        self.assertLess(report['retained_node_count'], len(u))

    def test_forced_coordinates_and_binding_are_preserved(self):
        u = np.arange(7.)
        grid = np.c_[u, 2*u][:, None, :]
        binding = {'source_sha256': 'a'*64, 'selector': 'explicit source fixture'}
        result, _, report = simplify(grid, u, .01, [2, 5], binding)
        np.testing.assert_array_equal(result, grid[[0, 2, 5, 6]])
        self.assertEqual(report['source_binding'], binding)

    def test_invalid_inputs_fail(self):
        g = np.zeros((3, 2, 3));u = np.arange(3.)
        for kwargs in [dict(parameters=[0, 0, 2]), dict(parameters=[0, 1]),
                       dict(max_deviation=0), dict(max_deviation=float('nan')),
                       dict(required_indices=[3]), dict(required_indices=[True]),
                       dict(required_indices=[1.5]), dict(grid=np.zeros((3, 3))),
                       dict(grid=np.full((3, 2, 3), np.nan))]:
            args = dict(grid=g, parameters=u, max_deviation=.01);args.update(kwargs)
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                simplify(**args)


if __name__ == '__main__':
    unittest.main()
