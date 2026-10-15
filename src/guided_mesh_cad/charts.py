"""Explicit, evidence-bound contour chart preparation."""
import json
import math

import numpy as np


def _binding(value):
    if value is None:
        return None
    if (not isinstance(value, dict) or
            not isinstance(value.get('source_sha256'), str) or
            len(value['source_sha256']) != 64 or
            not isinstance(value.get('selector'), str) or
            not value['selector'].strip()):
        raise ValueError('source_binding requires source_sha256 and non-empty selector')
    try:
        json.dumps(value, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError('source_binding must be finite JSON data') from exc
    return value


def _sample_counts(value, count):
    if isinstance(value, (int, np.integer)) and not isinstance(value, bool):
        values = [int(value)]*count
    else:
        try:
            values = list(value)
        except TypeError as exc:
            raise ValueError('samples_per_segment must be an integer or a sequence') from exc
        if len(values) != count:
            raise ValueError('samples_per_segment sequence must match semantic anchor count')
    if any(isinstance(item, bool) or not isinstance(item, (int, np.integer)) or int(item) < 2
           for item in values):
        raise ValueError('every segment sample count must be an integer >= 2')
    return [int(item) for item in values]


def _segment_indices(start, end, count):
    result = [start]
    while result[-1] != end:
        result.append((result[-1]+1) % count)
        if len(result) > count:
            raise ValueError('anchor segment does not terminate within one contour lap')
    return result


def _resample_segment(points, sample_count):
    lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cumulative = np.r_[0.0, np.cumsum(lengths)]
    targets = np.linspace(0.0, cumulative[-1], sample_count)
    result = np.column_stack([
        np.interp(targets, cumulative, points[:, coordinate])
        for coordinate in range(points.shape[1])
    ])
    # Do not leave source anchors to interpolation roundoff.
    result[0] = points[0]
    result[-1] = points[-1]
    return result, float(cumulative[-1])


def resample_closed_contours_by_anchors(
        contours, anchor_indices, samples_per_segment, tolerance=1e-9,
        expected_orientation='ccw', source_binding=None):
    """Split closed ordered contours at caller-supplied semantic anchors.

    Returns ``(charts, report)``. Each chart has shape
    ``(samples_on_segment, station_count, coordinate_dimension)``. The
    routine never detects or reorders features; anchor position and semantic
    order are entirely caller supplied.
    """
    tolerance = float(tolerance)
    if not math.isfinite(tolerance) or tolerance <= 0:
        raise ValueError('tolerance must be positive and finite')
    if expected_orientation not in ('ccw', 'cw'):
        raise ValueError("expected_orientation must be 'ccw' or 'cw'")
    try:
        contour_list = [np.asarray(contour, dtype=float) for contour in contours]
        anchor_list = [list(indices) for indices in anchor_indices]
    except TypeError as exc:
        raise ValueError('contours and anchor_indices must be station sequences') from exc
    if not contour_list or len(contour_list) != len(anchor_list):
        raise ValueError('contours and anchor_indices must contain the same nonzero station count')
    dimension = contour_list[0].shape[1] if contour_list[0].ndim == 2 else None
    if dimension not in (2, 3):
        raise ValueError('each contour must be an ordered Nx2 or Nx3 array')
    anchor_count = len(anchor_list[0])
    if anchor_count < 2 or any(len(indices) != anchor_count for indices in anchor_list):
        raise ValueError('every station must provide the same semantic anchor count >= 2')
    sample_counts = _sample_counts(samples_per_segment, anchor_count)
    binding = _binding(source_binding)

    normalized, station_reports = [], []
    expected_sign = 1.0 if expected_orientation == 'ccw' else -1.0
    for station, (contour, raw_indices) in enumerate(zip(contour_list, anchor_list)):
        if (contour.ndim != 2 or contour.shape[1] != dimension or len(contour) < 4 or
                not np.isfinite(contour).all()):
            raise ValueError(f'station {station}: contour must be finite Nx{dimension} with N>=4')
        closure_gap = float(np.linalg.norm(contour[-1]-contour[0]))
        if closure_gap > tolerance:
            raise ValueError(f'station {station}: contour is not explicitly closed')
        points = contour[:-1]
        if len(points) < 3:
            raise ValueError(f'station {station}: closed contour needs at least three unique vertices')
        cyclic_lengths = np.linalg.norm(np.roll(points, -1, axis=0)-points, axis=1)
        short = np.flatnonzero(cyclic_lengths <= tolerance).astype(int).tolist()
        if short:
            raise ValueError(f'station {station}: zero/short contour segment at indices {short}')
        area = 0.5*float(np.sum(
            points[:, 0]*np.roll(points[:, 1], -1)-
            points[:, 1]*np.roll(points[:, 0], -1)))
        if abs(area) <= tolerance*tolerance:
            raise ValueError(f'station {station}: contour orientation is degenerate')
        orientation = 'ccw' if area > 0 else 'cw'
        if area*expected_sign <= 0:
            raise ValueError(
                f'station {station}: contour orientation {orientation} does not match '
                f'{expected_orientation}')
        indices = []
        for value in raw_indices:
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
                raise ValueError(f'station {station}: anchor indices must be integers')
            index = int(value)
            if index < 0 or index >= len(points):
                raise ValueError(f'station {station}: anchor index {index} is out of range')
            indices.append(index)
        if len(set(indices)) != anchor_count:
            raise ValueError(f'station {station}: duplicate semantic anchor index')
        anchor_points = points[indices]
        for i in range(anchor_count):
            for j in range(i+1, anchor_count):
                if np.linalg.norm(anchor_points[i]-anchor_points[j]) <= tolerance:
                    raise ValueError(f'station {station}: duplicate semantic anchor coordinates')
        advances = [(indices[(i+1) % anchor_count]-indices[i]) % len(points)
                    for i in range(anchor_count)]
        if any(advance == 0 for advance in advances) or sum(advances) != len(points):
            raise ValueError(
                f'station {station}: semantic anchors must follow contour order for exactly one lap')
        full_cumulative = np.r_[0.0, np.cumsum(cyclic_lengths[:-1])]
        full_length = float(cyclic_lengths.sum())
        normalized.append((points, indices))
        station_reports.append({
            'station':station, 'input_point_count_including_closure':len(contour),
            'unique_point_count':len(points), 'closure_gap':closure_gap,
            'signed_area_first_two_coordinates':area, 'orientation':orientation,
            'anchor_indices':indices, 'anchor_points':anchor_points.tolist(),
            'full_loop_length':full_length,
            'anchor_normalized_full_loop_arclength':
                [float(full_cumulative[index]/full_length) for index in indices],
        })

    per_station_segments = []
    for station, (points, indices) in enumerate(normalized):
        station_segments, lengths, source_counts = [], [], []
        for segment in range(anchor_count):
            source_indices = _segment_indices(
                indices[segment], indices[(segment+1) % anchor_count], len(points))
            sampled, length = _resample_segment(points[source_indices], sample_counts[segment])
            station_segments.append(sampled)
            lengths.append(length)
            source_counts.append(len(source_indices))
        per_station_segments.append(station_segments)
        station_reports[station]['segment_lengths'] = lengths
        station_reports[station]['segment_source_point_counts'] = source_counts

    charts = [np.stack([station[segment] for station in per_station_segments], axis=1)
              for segment in range(anchor_count)]
    endpoint_error = 0.0
    for segment, chart in enumerate(charts):
        for station, (_, indices) in enumerate(normalized):
            points = normalized[station][0]
            endpoint_error = max(
                endpoint_error,
                float(np.linalg.norm(chart[0, station]-points[indices[segment]])),
                float(np.linalg.norm(
                    chart[-1, station]-points[indices[(segment+1) % anchor_count]])))
    report = {
        'method':'caller-supplied semantic anchors; per-segment polyline arclength resampling',
        'station_count':len(normalized), 'coordinate_dimension':dimension,
        'semantic_anchor_count':anchor_count, 'chart_count':len(charts),
        'samples_per_segment':sample_counts,
        'chart_shapes':[list(chart.shape) for chart in charts],
        'expected_orientation':expected_orientation,
        'endpoint_max_deviation':endpoint_error,
        'source_points_moved':False, 'source_binding':binding,
        'stations':station_reports,
        'limitations':[
            'Anchor semantics and cross-station correspondence are caller supplied',
            'Polyline resampling does not prove fitted-surface deviation between samples',
        ],
    }
    return charts, report


def simplify_corresponding_polylines(
        grid, parameters, max_deviation, required_indices=(), source_binding=None):
    """Reduce shared parameter nodes while bounding every supplied polyline.

    ``grid`` has shape (node, station, 2 or 3). Correspondence is supplied by
    the caller. Endpoints and required indices retain their exact coordinates.
    The bound applies to piecewise-linear interpolation at the same parameter,
    not arbitrary subsequent spline fits, feature recognition or missing data.
    """
    points = np.asarray(grid, dtype=float)
    u = np.asarray(parameters, dtype=float)
    if (points.ndim != 3 or points.shape[0] < 2 or points.shape[1] < 1 or
            points.shape[2] not in (2, 3) or not np.isfinite(points).all()):
        raise ValueError('grid must be finite (nodes>=2, stations>=1, dimension=2 or 3)')
    if (u.ndim != 1 or len(u) != len(points) or not np.isfinite(u).all() or
            not np.all(np.diff(u) > 0) or not math.isfinite(float(u[-1]-u[0]))):
        raise ValueError('parameters must be finite, strictly increasing and match grid nodes')
    if isinstance(max_deviation, (bool, np.bool_)):
        raise ValueError('max_deviation must be positive and finite')
    limit = float(max_deviation)
    if not math.isfinite(limit) or limit <= 0:
        raise ValueError('max_deviation must be positive and finite')
    binding = _binding(source_binding)
    required = {0, len(points)-1}
    try:
        raw_required = list(required_indices)
    except TypeError as exc:
        raise ValueError('required_indices must be a sequence of integer node indices') from exc
    for value in raw_required:
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
            raise ValueError('required_indices must contain integer node indices')
        if not 0 <= int(value) < len(points):
            raise ValueError('required node index is out of range')
        required.add(int(value))

    def segment_errors(first, last):
        if last == first+1:
            return np.zeros((0, points.shape[1]))
        fraction = (u[first+1:last]-u[first])/(u[last]-u[first])
        linear = ((1-fraction[:, None, None])*points[first] +
                  fraction[:, None, None]*points[last])
        error = np.linalg.norm(points[first+1:last]-linear, axis=2)
        if not np.isfinite(error).all():
            raise ValueError('Coordinate arithmetic overflow; rescale input coordinates')
        return error

    kept = set(required)
    ordered = sorted(required)
    pending = list(zip(ordered[:-1], ordered[1:]))
    while pending:
        first, last = pending.pop()
        error = segment_errors(first, last)
        if error.size:
            worst = np.max(error, axis=1)
            at = int(np.argmax(worst))
            if worst[at] > limit:
                split = first+1+at
                kept.add(split)
                pending.extend([(first, split), (split, last)])
    indices = np.asarray(sorted(kept), dtype=np.int64)
    per_station = np.zeros(points.shape[1])
    for first, last in zip(indices[:-1], indices[1:]):
        error = segment_errors(first, last)
        if error.size:
            per_station = np.maximum(per_station, error.max(axis=0))
    measured = float(per_station.max())
    if measured > limit:
        raise ValueError('Final correspondence bound exceeds requested maximum deviation')
    report = {
        'method': 'joint multi-station piecewise-linear parameter-domain simplification',
        'input_node_count': len(points), 'retained_node_count': len(indices),
        'station_count': points.shape[1], 'coordinate_dimension': points.shape[2],
        'retained_indices': indices.tolist(), 'required_indices': sorted(required),
        'max_deviation_requested': limit, 'measured_max_deviation': measured,
        'per_station_max_deviation': per_station.tolist(),
        'retained_coordinates_moved': False, 'source_binding': binding,
        'bound_scope': 'same-parameter Euclidean distance for the supplied piecewise-linear station polylines',
        'bound_basis': 'On each original interval the difference of both linear interpolants is affine; its norm is bounded by its endpoint norms.',
        'limitations': [
            'Semantic correspondence and required feature anchors are caller supplied',
            'Does not bound later high-order splines or unprovided source geometry',
            'Distance bounds alone do not establish topology or retain unnamed functional features',
            'Station interpolation inherits this bound only when both models use the same linear station interpolation',
            'Floating-point measurements are not an interval-arithmetic certificate',
        ],
    }
    return points[indices].copy(), u[indices].copy(), report
