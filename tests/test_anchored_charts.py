import json
import unittest

import numpy as np

from guided_mesh_cad.charts import resample_closed_contours_by_anchors


def notched_contour(depth, extra_bottom_point=False):
    points = [
        [0.0, 0.0, 0.0],
        [4.0, 0.0, 0.0],
        [4.0, 4.0, 0.0],
        [3.0, 4.0, 0.0],
        [3.0, depth, 0.0],
        [1.0, depth, 0.0],
        [1.0, 4.0, 0.0],
        [0.0, 4.0, 0.0],
    ]
    if extra_bottom_point:
        points.insert(1, [2.0, 0.0, 0.0])
    return np.asarray(points + [points[0]], dtype=float)


class AnchoredChartTests(unittest.TestCase):
    def test_moving_concave_corner_keeps_semantic_chart_endpoint(self):
        contours = [notched_contour(3.9), notched_contour(2.0, True),
                    notched_contour(0.1)]
        anchors = [list(range(8)), [0, 2, 3, 4, 5, 6, 7, 8], list(range(8))]
        charts, report = resample_closed_contours_by_anchors(
            contours, anchors, [5, 4, 4, 7, 5, 4, 4, 6],
            source_binding={'source_sha256':'b'*64,
                            'selector':'three ordered notch-section contours'})
        # Independent full-loop arclength puts this concave corner at changing
        # normalized positions, so one global U grid would shift its semantics.
        fractions = []
        for contour, indices in zip(contours, anchors):
            points = contour[:-1]
            lengths = np.linalg.norm(np.roll(points, -1, axis=0)-points, axis=1)
            fractions.append(float(lengths[:indices[4]].sum()/lengths.sum()))
        self.assertGreater(max(fractions)-min(fractions), 0.015)
        self.assertNotEqual(round(fractions[0]*100), round(fractions[-1]*100))
        reported = [station['anchor_normalized_full_loop_arclength'][4]
                    for station in report['stations']]
        np.testing.assert_allclose(reported, fractions, atol=1e-15)
        for station, (contour, indices) in enumerate(zip(contours, anchors)):
            for segment in range(8):
                np.testing.assert_array_equal(
                    charts[segment][0, station], contour[indices[segment]])
                np.testing.assert_array_equal(
                    charts[segment][-1, station], contour[indices[(segment+1) % 8]])
        self.assertEqual([chart.shape[1] for chart in charts], [3]*8)
        self.assertEqual(report['endpoint_max_deviation'], 0.0)
        self.assertFalse(report['source_points_moved'])
        self.assertEqual(report['source_binding']['source_sha256'], 'b'*64)
        json.dumps(report, allow_nan=False)

    def test_rejects_open_or_wrong_direction_contour(self):
        contour = notched_contour(2.0)
        with self.assertRaisesRegex(ValueError, 'explicitly closed'):
            resample_closed_contours_by_anchors(
                [contour[:-1]], [list(range(8))], 3)
        reversed_points = contour[:-1][::-1]
        reversed_contour = np.vstack([reversed_points, reversed_points[0]])
        with self.assertRaisesRegex(ValueError, 'orientation cw'):
            resample_closed_contours_by_anchors(
                [reversed_contour], [list(range(8))], 3,
                expected_orientation='ccw')

    def test_rejects_duplicate_anchor_and_zero_length_segment(self):
        contour = notched_contour(2.0)
        with self.assertRaisesRegex(ValueError, 'duplicate semantic anchor index'):
            resample_closed_contours_by_anchors(
                [contour], [[0, 1, 2, 3, 4, 5, 6, 6]], 3)
        duplicate = np.insert(contour, 3, contour[2], axis=0)
        with self.assertRaisesRegex(ValueError, 'zero/short contour segment'):
            resample_closed_contours_by_anchors(
                [duplicate], [[0, 1, 2, 4, 5, 6, 7, 8]], 3)

    def test_rejects_inconsistent_semantics_and_anchor_order(self):
        a, b = notched_contour(3.0), notched_contour(1.0)
        with self.assertRaisesRegex(ValueError, 'same semantic anchor count'):
            resample_closed_contours_by_anchors(
                [a, b], [list(range(8)), list(range(7))], 3)
        with self.assertRaisesRegex(ValueError, 'exactly one lap'):
            resample_closed_contours_by_anchors(
                [a], [[0, 2, 1, 3, 4, 5, 6, 7]], 3)
        with self.assertRaisesRegex(ValueError, 'sequence must match'):
            resample_closed_contours_by_anchors(
                [a], [list(range(8))], [3, 3])
        with self.assertRaisesRegex(ValueError, 'integer >= 2'):
            resample_closed_contours_by_anchors(
                [a], [list(range(8))], 1)

    def test_semantic_anchor_order_may_wrap_raw_index_zero(self):
        original = notched_contour(2.0)[:-1]
        shift = 3
        rotated = np.vstack([original[shift:], original[:shift]])
        rotated = np.vstack([rotated, rotated[0]])
        indices = [(semantic-shift) % len(original) for semantic in range(8)]
        charts, _ = resample_closed_contours_by_anchors(
            [rotated], [indices], 3)
        for semantic in range(8):
            np.testing.assert_array_equal(charts[semantic][0, 0], original[semantic])


if __name__ == '__main__':
    unittest.main()
