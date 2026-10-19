import unittest
import numpy as np

from guided_mesh_cad.disk_charts import prepare_source_disk_quads


def source_disk():
    count = 12
    angle = np.arange(count)*2*np.pi/count
    outer_radius = 1 + .12*np.sin(3*angle)+.06*np.cos(5*angle)
    xy = np.vstack(([0., 0.], np.c_[.43*np.cos(angle), .43*np.sin(angle)],
                    np.c_[outer_radius*np.cos(angle), outer_radius*np.sin(angle)]))
    z = .19*np.sin(1.3*xy[:, 0])+.17*xy[:, 0]*xy[:, 1]+.11*xy[:, 1]**2
    points = np.c_[xy, z]
    triangles = []
    for i in range(count):
        j = (i+1) % count
        triangles.extend([[0, 1+i, 1+j], [1+i, 13+i, 13+j], [1+i, 13+j, 1+j]])
    return points, np.array(triangles), np.arange(13, 25)


class DiskChartsTests(unittest.TestCase):
    def test_complete_irregular_three_dimensional_source_and_lineage(self):
        p, f, boundary = source_disk()
        before = [x.tobytes() for x in (p, f, boundary)]
        out = prepare_source_disk_quads(p, f, boundary, [0, 4, 8], samples=17)
        self.assertEqual(out['report']['quad_count'], 3)
        self.assertEqual(out['report']['source_triangles'], len(f))
        self.assertFalse(out['report']['native_or_whole_accepted'])
        for grid, lineage in zip(out['grids'], out['lineage']):
            triangles = p[f[lineage['source_triangle_indices']]]
            replay = np.einsum('ijk,ijkl->ijl', lineage['source_barycentric'], triangles)
            np.testing.assert_allclose(grid, replay, rtol=0, atol=2e-12)
            self.assertGreaterEqual(lineage['source_barycentric'].min(), -1e-12)
        self.assertLess(max(out['report']['internal_boundary_sample_gaps']), 1e-12)
        self.assertEqual(before, [x.tobytes() for x in (p, f, boundary)])

    def test_arbitrary_orientation_equivariance(self):
        p, f, b = source_disk()
        rotation = np.array([[.36, -.48, .8], [.8, .6, 0], [-.48, .64, .6]])
        offset = np.array([7.3, -4.1, 2.7])
        first = prepare_source_disk_quads(p, f, b, [0, 4, 8], samples=9)
        second = prepare_source_disk_quads(p @ rotation.T+offset, f, b, [0, 4, 8], samples=9)
        for a, c in zip(first['grids'], second['grids']):
            np.testing.assert_allclose(a @ rotation.T+offset, c, rtol=0, atol=2e-11)

    def test_six_sided_semantics_and_reversed_source(self):
        p, f, b = source_disk()
        out = prepare_source_disk_quads(p, f[:, [0, 2, 1]], b, [0, 2, 4, 6, 8, 10], samples=9)
        self.assertEqual(len(out['grids']), 6)
        self.assertEqual(out['report']['source_orientation_in_chart'], -1)
        self.assertLess(max(out['report']['internal_boundary_sample_gaps']), 1e-12)

    def test_boundary_ears_not_collapsed_to_straight_triangle_edges(self):
        angle = np.arange(9)*2*np.pi/9
        p = np.c_[np.cos(angle), np.sin(angle), .2*np.cos(2*angle)]
        f = np.array([[0, i, i+1] for i in range(1, 8)])
        out = prepare_source_disk_quads(p, f, np.arange(9), [0, 3, 6], samples=9)
        self.assertGreater(out['report']['minimum_all_original_UV_triangle_twice_area'], 0)
        self.assertEqual(out['report']['source_triangles'], 7)

    def test_bad_anchor_order_rejected(self):
        p, f, b = source_disk()
        with self.assertRaises(ValueError):
            prepare_source_disk_quads(p, f, b, [0, 6, 3, 9])

    def test_missing_source_triangle_not_filled(self):
        p, f, b = source_disk()
        with self.assertRaises(ValueError):
            prepare_source_disk_quads(p, f[1:], b, [0, 4, 8])

    def test_duplicate_source_triangle_rejected(self):
        p, f, b = source_disk()
        with self.assertRaises(ValueError):
            prepare_source_disk_quads(p, np.vstack((f, f[0])), b, [0, 4, 8])

    def test_unused_source_vertex_not_silently_dropped(self):
        p, f, b = source_disk()
        with self.assertRaises(ValueError):
            prepare_source_disk_quads(np.vstack((p, [9., 9., 9.])), f, b, [0, 4, 8])

    def test_inconsistent_source_orientation_rejected(self):
        p, f, b = source_disk()
        f[0] = f[0, [0, 2, 1]]
        with self.assertRaises(ValueError):
            prepare_source_disk_quads(p, f, b, [0, 4, 8])

    def test_nonfinite_coordinates_and_bad_binding_rejected(self):
        p, f, b = source_disk()
        with self.assertRaises(ValueError):
            prepare_source_disk_quads(p, f, b, [0, 4, 8], source_binding={'source_sha256': 'a'*64, 'selector': ''})
        p[2, 0] = np.nan
        with self.assertRaises(ValueError):
            prepare_source_disk_quads(p, f, b, [0, 4, 8])

    def test_invalid_sample_counts_rejected(self):
        p, f, b = source_disk()
        for samples in [True, 3, 258, 7.5]:
            with self.subTest(samples=samples), self.assertRaises(ValueError):
                prepare_source_disk_quads(p, f, b, [0, 4, 8], samples=samples)


if __name__ == '__main__':
    unittest.main()
