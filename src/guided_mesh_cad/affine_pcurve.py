"""Native pullback of one affine UV segment on a B-spline support surface.

Homogeneous tensor Bernstein composition covers every crossed support span.
The continuity-aware coefficient solve is accepted only after a complete
rational Bernstein difference bound. These are floating bounds, not directed
interval arithmetic, a source-geometry fit, or a later face/solid acceptance.
"""
import math
import numpy as np


def _restrict(control, first, last):
    """Blossom restriction, including decreasing affine parameter direction."""
    result = []
    degree = len(control)-1
    for k in range(degree+1):
        work = control.copy()
        for t in [first]*(degree-k)+[last]*k:
            work = (1-t)*work[:-1]+t*work[1:]
        result.append(work[0])
    return np.asarray(result)


def _product(a, b):
    p, q = len(a)-1, len(b)-1
    out = np.zeros((p+q+1,)+np.broadcast_shapes(a.shape[1:], b.shape[1:]))
    for i in range(p+1):
        for j in range(q+1):
            out[i+j] += a[i]*b[j]*(math.comb(p, i)*math.comb(q, j)/math.comb(p+q, i+j))
    return out


def _native_arrays(values, integer=False):
    from OCP.TColStd import TColStd_Array1OfReal, TColStd_Array1OfInteger
    array = (TColStd_Array1OfInteger if integer else TColStd_Array1OfReal)(1, len(values))
    for i, value in enumerate(values, 1):
        array.SetValue(i, int(value) if integer else float(value))
    return array


def _geometry_hash(geometry):
    import io
    import hashlib
    from OCP.GeomTools import GeomTools
    stream = io.BytesIO()
    GeomTools.Write_s(geometry, stream)
    return hashlib.sha256(stream.getvalue()).hexdigest()


def _positive(value, name):
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(name+' must be a positive finite number')
    value = float(value)
    if not np.isfinite(value) or value <= 0:
        raise ValueError(name+' must be a positive finite number')
    return value


def pullback_affine_pcurve(surface, uv_start, uv_end, *, parameter_range=(0., 1.),
                           max_deviation_mm=1e-8, max_spans=4096):
    """Return ``(Geom_BSplineCurve, report, certificate_arrays)``.

    ``surface`` must be a native Geom_BSplineSurface. The affine UV path is
    ``uv_start + (uv_end-uv_start)*(t-first)/(last-first)`` on the exact finite
    increasing ``parameter_range``. Endpoints may run in either UV direction;
    they must be inside the support's actual Bezier decomposition domain.
    A face's trimming wires are not inferred or consulted by this support API.

    Only actually crossed tensor knots constrain the output continuity. A
    real C0 crossing remains a C0 knot, and separate polyline segments must
    be passed separately. Discontinuous crossings, unsupported native curve
    degrees (>25), under-resolved floating parameter spans and nonpositive
    weights fail explicitly; no endpoint clipping or corner smoothing occurs.

    A sparse continuity-aware homogeneous coefficient solve may incur floating
    error. Every entire resulting native span is compared with the composed
    source tensor span using a rational Bernstein control-hull bound. Failure
    to meet the requested bound raises ValueError, rather than returning an
    uncertified approximation. Input native surface data are not modified.
    """
    from scipy.interpolate import BSpline
    from scipy.sparse.linalg import splu
    from OCP.Geom import Geom_BSplineSurface, Geom_BSplineCurve
    from OCP.GeomConvert import GeomConvert_BSplineSurfaceToBezierSurface, GeomConvert_BSplineCurveToBezierCurve
    from OCP.TColgp import TColgp_Array1OfPnt
    from OCP.gp import gp_Pnt

    if not isinstance(surface, Geom_BSplineSurface):
        raise TypeError('surface must be a native Geom_BSplineSurface support')
    q0 = np.array(uv_start, dtype=float, copy=True)
    q1 = np.array(uv_end, dtype=float, copy=True)
    tr = np.array(parameter_range, dtype=float, copy=True)
    if q0.shape != (2,) or q1.shape != (2,) or not np.isfinite([q0, q1]).all():
        raise ValueError('UV endpoints must each contain two finite numbers')
    if np.array_equal(q0, q1):
        raise ValueError('affine UV segment must have distinct endpoints')
    if tr.shape != (2,) or not np.isfinite(tr).all() or not tr[0] < tr[1] or not math.isfinite(float(tr[1])-float(tr[0])):
        raise ValueError('parameter_range must contain two finite increasing values with finite span')
    limit = _positive(max_deviation_mm, 'max_deviation_mm')
    if isinstance(max_spans, (bool, np.bool_)) or int(max_spans) != max_spans or not 1 <= max_spans <= 100000:
        raise ValueError('max_spans must be an integer in [1,100000]')
    before = _geometry_hash(surface)
    converter = GeomConvert_BSplineSurfaceToBezierSurface(surface)
    from OCP.TColStd import TColStd_Array1OfReal
    ku = TColStd_Array1OfReal(1, converter.NbUPatches()+1)
    kv = TColStd_Array1OfReal(1, converter.NbVPatches()+1)
    converter.UKnots(ku)
    converter.VKnots(kv)
    u = np.array([ku.Value(i) for i in range(1, ku.Length()+1)])
    v = np.array([kv.Value(i) for i in range(1, kv.Length()+1)])
    if np.any(np.diff(u) <= 0) or np.any(np.diff(v) <= 0):
        raise ValueError('native surface decomposition has non-increasing domains')
    if np.any(np.minimum(q0, q1) < [u[0], v[0]]) or np.any(np.maximum(q0, q1) > [u[-1], v[-1]]):
        raise ValueError('affine UV path extends outside actual native support domain')
    direction = q1-q0
    crossings = {}
    for axis, knots in enumerate([u, v]):
        if direction[axis] == 0:
            continue
        source_degree = surface.UDegree() if axis == 0 else surface.VDegree()
        original = {surface.UKnot(i): surface.UMultiplicity(i) for i in range(1, surface.NbUKnots()+1)} if axis == 0 else {
            surface.VKnot(i): surface.VMultiplicity(i) for i in range(1, surface.NbVKnots()+1)}
        for knot in knots[1:-1]:
            t = float((knot-q0[axis])/direction[axis])
            if 0 < t < 1:
                continuity = source_degree-original[float(knot)]
                if continuity < 0:
                    raise ValueError('affine path crosses a discontinuous native surface knot')
                crossings.setdefault(t, []).append({'axis': 'U' if axis == 0 else 'V',
                    'native_knot': float(knot), 'source_continuity_order': int(continuity)})
    events = np.array([0., *sorted(crossings), 1.])
    if len(events)-1 > max_spans:
        raise ValueError('affine path exceeds max_spans')
    actual_t = tr[0]+events*(tr[1]-tr[0])
    actual_t[0], actual_t[-1] = tr
    if np.any(np.diff(actual_t) <= 0):
        raise ValueError('native parameter range cannot resolve every crossed knot span')
    constant_u, constant_v = direction == 0
    degree = surface.VDegree() if constant_u else surface.UDegree() if constant_v else surface.UDegree()+surface.VDegree()
    if degree > Geom_BSplineCurve.MaxDegree_s():
        raise ValueError('composed affine pullback exceeds native B-spline maximum degree')
    origin = np.array(surface.Value(*map(float, q0)).Coord())
    # Floating native knot parameters may not represent normalized crossings
    # exactly when parameter_range has a large offset. Account for the entire
    # affine map error through a full rational support derivative hull, rather
    # than certifying only the differently rounded native parameter path.
    derivative_bounds = np.zeros(2)
    for i in range(1, converter.NbUPatches()+1):
        for j in range(1, converter.NbVPatches()+1):
            patch = converter.Patch(i, j)
            pp, qq = patch.UDegree(), patch.VDegree()
            w = np.array([[patch.Weight(a, b) for b in range(1, qq+2)] for a in range(1, pp+2)])
            xyz = np.array([[patch.Pole(a, b).Coord() for b in range(1, qq+2)] for a in range(1, pp+2)])
            if not np.isfinite(w).all() or np.any(w <= 0) or not np.isfinite(xyz).all():
                raise ValueError('finite positive native rational surface weights are required')
            w /= w.max()
            h = (xyz-origin)*w[:, :, None]
            hmax = np.linalg.norm(h, axis=2).max()
            for axis, pd, span in [(0, pp, u[i]-u[i-1]), (1, qq, v[j]-v[j-1])]:
                dh = pd*np.diff(h, axis=axis)
                dw = pd*np.diff(w, axis=axis)
                with np.errstate(over='ignore', divide='ignore', invalid='ignore'):
                    bound = (np.linalg.norm(dh, axis=2).max()*w.max()+hmax*np.abs(dw).max())/(w.min()**2*span)
                if not np.isfinite(bound):
                    raise ValueError('rational support derivative bound exceeds finite numerical range')
                derivative_bounds[axis] = max(derivative_bounds[axis], bound)
    normalized_rounding = np.abs((actual_t-tr[0])/(tr[1]-tr[0])-events).max()
    parameter_rounding_bound = float(derivative_bounds@np.abs(direction)*normalized_rounding)
    pieces = []
    span_rows = []
    for first, last in zip(events[:-1], events[1:]):
        mid = q0+((first+last)/2)*direction
        iu = min(len(u)-2, int(np.searchsorted(u, mid[0], side='right')-1))
        iv = min(len(v)-2, int(np.searchsorted(v, mid[1], side='right')-1))
        if iu < 0 or iv < 0:
            raise ValueError('affine subspan has no native tensor patch')
        patch = converter.Patch(iu+1, iv+1)
        p, q = patch.UDegree(), patch.VDegree()
        weights = np.array([[patch.Weight(i, j) for j in range(1, q+2)] for i in range(1, p+2)])
        poles = np.array([[patch.Pole(i, j).Coord() for j in range(1, q+2)] for i in range(1, p+2)])
        if not np.isfinite(poles).all() or not np.isfinite(weights).all() or np.any(weights <= 0):
            raise ValueError('finite positive native rational surface weights are required')
        weights /= weights.max()
        h = np.concatenate(((poles-origin)*weights[:, :, None], weights[:, :, None]), axis=2)
        a = (q0+first*direction-[u[iu], v[iv]])/[u[iu+1]-u[iu], v[iv+1]-v[iv]]
        b = (q0+last*direction-[u[iu], v[iv]])/[u[iu+1]-u[iu], v[iv+1]-v[iv]]
        # At computed knot events division can round beyond 0/1 by a few ULP.
        # Keep the affine formula (no clamping); reject larger selection errors.
        # Small extrapolation roundoff is covered by the floating allowance.
        if np.any(a < -64*np.finfo(float).eps) or np.any(a > 1+64*np.finfo(float).eps) or np.any(b < -64*np.finfo(float).eps) or np.any(b > 1+64*np.finfo(float).eps):
            raise ValueError('native tensor span selection does not cover affine subspan')
        h = _restrict(h, a[0], b[0])
        h = np.swapaxes(_restrict(np.swapaxes(h, 0, 1), a[1], b[1]), 0, 1)
        if constant_u:
            c = h[0].copy()
        elif constant_v:
            c = h[:, 0].copy()
        else:
            c = np.zeros((degree+1, 4))
            for i in range(p+1):
                for j in range(q+1):
                    c[i+j] += h[i, j]*(math.comb(p, i)*math.comb(q, j)/math.comb(degree, i+j))
        if np.any(c[:, 3] <= 0) or not np.isfinite(c).all():
            raise ValueError('composition has nonpositive or nonfinite homogeneous controls')
        # A constant homogeneous scale does not change a rational patch. Match
        # connected endpoint weights so local Bezier normalizations cannot
        # create artificial discontinuity in the global coefficient system.
        if pieces:
            c *= pieces[-1][-1, 3]/c[0, 3]
        if not np.isfinite(c).all() or np.any(c[:, 3] <= 0):
            raise ValueError('cross-span homogeneous normalization is not numerically resolvable')
        pieces.append(c)
        span_rows.append({'normalized_range': [float(first), float(last)],
                          'native_tensor_patch': [iu+1, iv+1]})
    pieces = np.asarray(pieces)
    mult = np.full(len(events), degree, dtype=int)
    mult[[0, -1]] = degree+1
    continuity_rows = []
    for i, t in enumerate(events[1:-1], 1):
        order = min(degree-1, *(a['source_continuity_order'] for a in crossings[float(t)]))
        mult[i] = degree-order
        continuity_rows.append({'normalized_parameter': float(t), 'native_parameter': float(actual_t[i]),
            'crossed_source_knots': crossings[float(t)], 'preserved_continuity_order': int(order),
            'native_multiplicity': int(mult[i])})
    knots = np.repeat(events, mult)
    nodes = (1-np.cos(np.arange(degree+1)*np.pi/degree))/2
    bern = np.array([[math.comb(degree, k)*x**k*(1-x)**(degree-k) for k in range(degree+1)] for x in nodes])
    parameters = np.concatenate([a+(b-a)*nodes for a, b in zip(events[:-1], events[1:])])
    target = np.concatenate([bern@piece for piece in pieces])
    matrix = BSpline.design_matrix(parameters, knots, degree)
    try:
        controls = splu((matrix.T@matrix).tocsc()).solve(np.asarray(matrix.T@target))
    except (RuntimeError, ValueError) as exc:
        raise ValueError('affine pullback coefficient system is not numerically resolvable') from exc
    if not np.isfinite(controls).all() or np.any(controls[:, 3] <= 0):
        raise ValueError('native pullback controls require finite positive weights')
    # Both actual endpoints are fixed without changing the source surface.
    # Their change, and all other solve error, are covered below on whole spans.
    controls[0, :3] = 0.
    controls[-1, :3] = (np.array(surface.Value(*map(float, q1)).Coord())-origin)*controls[-1, 3]
    poles = TColgp_Array1OfPnt(1, len(controls))
    for i, h in enumerate(controls, 1):
        poles.SetValue(i, gp_Pnt(*map(float, origin+h[:3]/h[3])))
    curve = Geom_BSplineCurve(poles, _native_arrays(controls[:, 3]), _native_arrays(actual_t),
                             _native_arrays(mult, True), degree, False)
    actual = GeomConvert_BSplineCurveToBezierCurve(curve)
    if actual.NbArcs() != len(pieces):
        raise ValueError('native pullback decomposition does not cover source span count')
    native_events = TColStd_Array1OfReal(1, actual.NbArcs()+1)
    actual.Knots(native_events)
    native_events = np.array([native_events.Value(i) for i in range(1, native_events.Length()+1)])
    if not np.array_equal(native_events, actual_t):
        raise ValueError('native rational decomposition changed parameter span boundaries')
    bounds = []
    actual_pieces = []
    for i, source in enumerate(pieces, 1):
        arc = actual.Arc(i)
        h = np.array([np.r_[(np.array(arc.Pole(j).Coord())-origin)*arc.Weight(j), arc.Weight(j)] for j in range(1, arc.NbPoles()+1)])
        if len(h) != degree+1 or not np.isfinite(h).all() or np.any(h[:, 3] <= 0):
            raise ValueError('invalid actual native rational Bernstein span')
        numerator = _product(source[:, :3], h[:, 3, None])-_product(h[:, :3], source[:, 3, None])
        bound = float(np.linalg.norm(numerator, axis=1).max()/(source[:, 3].min()*h[:, 3].min()))
        bounds.append(bound)
        actual_pieces.append(h)
    # Explicit numerical allowance for floating coordinate conversion. This is
    # not an outward-rounded proof; large world coordinates can make a very
    # small requested tolerance unresolvable and are rejected accordingly.
    scale = max(1., float(np.abs(origin).max()), float(np.abs(controls[:, :3]/controls[:, 3, None]).max()))
    allowance = float(64*np.finfo(float).eps*(degree+1)*scale)
    total = max(bounds)+allowance+parameter_rounding_bound
    if not np.isfinite(total) or total > limit:
        raise ValueError(f'complete affine pullback bound {total:.17g} mm exceeds requested {limit:.17g} mm')
    if _geometry_hash(surface) != before:
        raise RuntimeError('input native surface was modified')
    report = {'method': 'native_affine_pcurve_rational_bernstein_pullback.v1', 'units': 'mm',
        'scope': 'One affine support-UV segment. Full same-parameter floating rational Bernstein control-hull bound on every native span, not interval arithmetic, face trim containment, whole-source geometry or solid acceptance.',
        'uv_endpoints': [q0.tolist(), q1.tolist()], 'native_parameter_range': tr.tolist(),
        'degree': degree, 'native_continuity': str(curve.Continuity()),
        'constant_U': bool(constant_u), 'constant_V': bool(constant_v),
        'support_domain': [[float(u[0]), float(v[0])], [float(u[-1]), float(v[-1])]],
        'source_surface_geometry_sha256': before, 'input_geometry_modified': False,
        'native_span_count': len(pieces), 'native_span_boundaries': actual_t.tolist(),
        'source_tensor_spans': span_rows, 'continuity_at_crossings': continuity_rows,
        'whole_span_coefficient_bounds_mm': bounds,
        'whole_curve_coefficient_bound_mm': max(bounds), 'floating_coordinate_allowance_mm': allowance,
        'full_support_derivative_bounds_mm_per_UV': derivative_bounds.tolist(),
        'native_parameter_rounding_normalized_max': float(normalized_rounding),
        'native_parameter_rounding_whole_path_bound_mm': parameter_rounding_bound,
        'whole_curve_bound_with_floating_allowance_mm': total, 'requested_max_deviation_mm': limit,
        'native_min_positive_weight': float(controls[:, 3].min()),
        'source_min_positive_weight': float(pieces[:, :, 3].min()),
        'native_pole_count': len(controls), 'native_knot_multiplicities': mult.tolist(),
        'coordinate_origin_for_conditioning': origin.tolist(), 'genuine_C0_crossings_preserved': True}
    arrays = {'source_homogeneous_Bernstein': pieces, 'actual_native_homogeneous_Bernstein': np.array(actual_pieces),
              'native_span_boundaries': actual_t, 'normalized_span_boundaries': events,
              'conditioned_coordinate_origin': origin}
    return curve, report, arrays


def make_affine_pcurve_edge(surface, uv_start, uv_end, *, parameter_range=(0., 1.),
                            max_deviation_mm=1e-8, tolerance=1e-7, reverse=False,
                            start_vertex=None, end_vertex=None, max_spans=4096):
    """Return ``(CadQuery Edge, report, arrays)`` with its actual affine pcurve.

    The 3D curve and affine 2D curve use the exact same finite native range.
    ``reverse`` changes only TopoDS orientation, not parameterization or the
    endpoints assigned to increasing parameter order. Optional shared vertices
    correspond to uv_start/uv_end, irrespective of ``reverse``; their actual
    coordinate offsets and existing tolerances must fit the requested topology
    tolerance. No vertices are moved, aliases merged, or loops simplified.
    The requested tolerance must be <=0.01 mm and >= the certified floating
    curve/support budget. Each polyline segment needs a separate call to keep
    real corners. Closed/degenerate images are rejected as single-edge inputs.
    """
    import cadquery as cq
    from OCP.BRep import BRep_Builder, BRep_Tool
    from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeEdge, BRepBuilderAPI_MakeVertex
    from OCP.Geom2d import Geom2d_BSplineCurve
    from OCP.TColgp import TColgp_Array1OfPnt2d
    from OCP.TopLoc import TopLoc_Location
    from OCP.TopoDS import TopoDS_Vertex, TopoDS
    from OCP.gp import gp_Pnt2d
    tol = _positive(tolerance, 'tolerance')
    if tol > .01 or not isinstance(reverse, (bool, np.bool_)):
        raise ValueError('tolerance must be <=0.01 mm and reverse must be boolean')
    curve, report, arrays = pullback_affine_pcurve(surface, uv_start, uv_end,
        parameter_range=parameter_range, max_deviation_mm=max_deviation_mm, max_spans=max_spans)
    if report['whole_curve_bound_with_floating_allowance_mm'] > tol:
        raise ValueError('topology tolerance is smaller than certified curve/support budget')
    first, last = report['native_parameter_range']
    source_ends = [surface.Value(*map(float, uv)) for uv in [uv_start, uv_end]]
    if source_ends[0].Distance(source_ends[1]) <= 2*tol:
        raise ValueError('single affine edge image has coincident or unresolved endpoints')
    verts = []
    offsets = []
    for vertex, point in zip([start_vertex, end_vertex], source_ends):
        if vertex is None:
            v = BRepBuilderAPI_MakeVertex(point).Vertex()
        else:
            v = vertex.wrapped if isinstance(vertex, cq.Vertex) else vertex
            if not isinstance(v, TopoDS_Vertex) or v.IsNull():
                raise TypeError('shared endpoint must be a native TopoDS_Vertex or CadQuery Vertex')
        offset = BRep_Tool.Pnt_s(v).Distance(point)
        if offset+report['whole_curve_bound_with_floating_allowance_mm'] > tol or BRep_Tool.Tolerance_s(v) > tol:
            raise ValueError('shared vertex coordinate/tolerance does not fit actual support endpoint budget')
        verts.append(v)
        offsets.append(float(offset))
    maker = BRepBuilderAPI_MakeEdge(curve, verts[0], verts[1], first, last)
    if not maker.IsDone():
        raise ValueError('native affine pullback edge construction failed')
    edge = maker.Edge()
    poles = TColgp_Array1OfPnt2d(1, 2)
    for i, uv in enumerate([uv_start, uv_end], 1):
        poles.SetValue(i, gp_Pnt2d(*map(float, uv)))
    pc = Geom2d_BSplineCurve(poles, _native_arrays([first, last]), _native_arrays([2, 2], True), 1, False)
    builder = BRep_Builder()
    builder.UpdateEdge(edge, pc, surface, TopLoc_Location(), tol)
    builder.Range(edge, surface, TopLoc_Location(), first, last)
    if reverse:
        edge = TopoDS.Edge_s(edge.Reversed())
    result = cq.Edge(edge)
    if not result.isValid() or BRep_Tool.Tolerance_s(edge) > tol:
        raise ValueError('native affine pcurve edge did not pass its actual edge/tolerance gate')
    report.update({'topological_orientation_reversed': bool(reverse), 'actual_edge_tolerance_mm': BRep_Tool.Tolerance_s(edge),
                   'requested_topology_tolerance_mm': tol, 'actual_vertex_endpoint_offsets_mm': offsets,
                   'shared_vertex_inputs': [start_vertex is not None, end_vertex is not None],
                   'affine_pcurve_native_parameter_range': [pc.FirstParameter(), pc.LastParameter()],
                   'face_or_solid_acceptance': 'not-checked'})
    return result, report, arrays
