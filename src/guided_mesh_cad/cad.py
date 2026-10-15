from collections import Counter
from pathlib import Path
import json
import math
import numpy as np


class PatchTopologyError(ValueError):
    """Topology failure with a JSON-serializable diagnostic report."""

    def __init__(self, message, report):
        super().__init__(message)
        self.report = report


def _point(value):
    return np.array([value.x, value.y, value.z], dtype=float)


def _edge_points(edge, samples):
    return np.vstack([_point(edge.positionAt(float(t))) for t in np.linspace(0.0, 1.0, samples)])


def _segment_distance(a0, a1, b0, b1):
    """Shortest distance between finite 3-D segments."""
    u, v, w = a1-a0, b1-b0, a0-b0
    aa, bb, cc = float(u@u), float(u@v), float(v@v)
    dd, ee = float(u@w), float(v@w)
    den = aa*cc-bb*bb
    if aa <= 1e-30 and cc <= 1e-30:
        return float(np.linalg.norm(a0-b0))
    if aa <= 1e-30:
        return float(np.linalg.norm(a0-(b0+np.clip(ee/cc, 0, 1)*v)))
    if cc <= 1e-30:
        return float(np.linalg.norm((a0+np.clip(-dd/aa, 0, 1)*u)-b0))
    s = 0.0 if abs(den) <= 1e-30 else np.clip((bb*ee-cc*dd)/den, 0, 1)
    t = np.clip((bb*s+ee)/cc, 0, 1)
    s = np.clip((bb*t-dd)/aa, 0, 1)
    return float(np.linalg.norm((a0+s*u)-(b0+t*v)))


def analyze_polyline(points, tolerance, sharp_turn_deg=45.0):
    """Diagnose sampled boundary degeneration without changing coordinates."""
    p = np.asarray(points, dtype=float)
    if p.ndim != 2 or p.shape[1] != 3 or len(p) < 2 or not np.isfinite(p).all():
        raise ValueError('boundary points must be finite Nx3 with N>=2')
    seg = np.diff(p, axis=0); lengths = np.linalg.norm(seg, axis=1)
    zero = np.flatnonzero(lengths <= tolerance).astype(int).tolist()
    turns, foldbacks = [], []
    for i in range(1, len(seg)):
        if lengths[i-1] <= tolerance or lengths[i] <= tolerance:
            continue
        dot = float(np.clip((seg[i-1]@seg[i])/(lengths[i-1]*lengths[i]), -1, 1))
        angle = math.degrees(math.acos(dot))
        if angle >= sharp_turn_deg:
            turns.append({'vertex_index':i, 'turn_deg':angle})
        # Consecutive segments normally meet at one endpoint.  A collinear
        # reversal overlaps beyond that endpoint and is therefore a true
        # foldback, rather than merely a sharp but valid corner.
        cross_ratio = float(np.linalg.norm(np.cross(seg[i-1], seg[i])) /
                            (lengths[i-1]*lengths[i]))
        if dot < 0.0 and cross_ratio <= 1e-10:
            foldbacks.append({'vertex_index':i, 'turn_deg':angle})
    crossings = []
    for i in range(len(seg)):
        for j in range(i+2, len(seg)):
            if i == 0 and j == len(seg)-1 and np.linalg.norm(p[0]-p[-1]) <= tolerance:
                continue
            distance = _segment_distance(p[i], p[i+1], p[j], p[j+1])
            if distance <= tolerance:
                crossings.append({'segments':[i,j], 'distance_mm':distance})
    return {'point_count':len(p), 'length_mm':float(lengths.sum()),
            'min_segment_mm':float(lengths.min()), 'zero_or_short_segments':zero,
            'sharp_turns':turns, 'adjacent_foldbacks':foldbacks,
            'self_contacts':crossings}


def canonical_shared_edge(points, tolerance=1e-7, parameters=None, source_binding=None):
    """Interpolate one immutable native edge through source points.

    The points are never snapped or averaged. Callers must reuse this returned
    TopoDS_Edge (or its reversed orientation) in both adjacent face wires.
    """
    import cadquery as cq
    p = np.asarray(points, dtype=float)
    diagnostic = analyze_polyline(p, tolerance)
    if diagnostic['zero_or_short_segments']:
        raise PatchTopologyError('shared edge contains a zero/short source segment', diagnostic)
    if diagnostic['self_contacts'] or diagnostic['adjacent_foldbacks']:
        raise PatchTopologyError('shared edge source polyline folds back or self-contacts', diagnostic)
    if parameters is None:
        lengths = np.linalg.norm(np.diff(p, axis=0), axis=1)
        parameters = np.r_[0.0, np.cumsum(lengths)]
        parameters = (parameters/parameters[-1]).tolist()
    elif len(parameters) != len(p):
        raise ValueError('shared edge parameters must match point count')
    edge = cq.Edge.makeSpline([cq.Vector(*map(float, q)) for q in p],
                              parameters=list(map(float, parameters)), tol=float(tolerance))
    source_deviation = [float(edge.distance(cq.Vertex.makeVertex(*map(float, q)))) for q in p]
    native_sample_count = max(65, 8*(len(p)-1)+1)
    native_diagnostic = analyze_polyline(_edge_points(edge, native_sample_count), tolerance)
    diagnostic.update({'source_point_max_deviation_mm':max(source_deviation),
                       'source_points_moved':False, 'native_edge_valid':edge.isValid(),
                       'native_edge_sample_check':native_diagnostic,
                       'source_binding':source_binding})
    if source_binding is not None:
        if not isinstance(source_binding, dict) or not isinstance(source_binding.get('source_sha256'), str) \
                or len(source_binding['source_sha256']) != 64 or not isinstance(source_binding.get('selector'), str) \
                or not source_binding['selector'].strip():
            raise ValueError('source_binding requires source_sha256 and non-empty selector')
        try:
            json.dumps(source_binding, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise ValueError('source_binding must be finite JSON data') from exc
    if (not edge.isValid() or diagnostic['source_point_max_deviation_mm'] > tolerance or
            native_diagnostic['zero_or_short_segments'] or
            native_diagnostic['adjacent_foldbacks'] or native_diagnostic['self_contacts']):
        raise PatchTopologyError('native shared edge does not interpolate the frozen source points', diagnostic)
    return edge, diagnostic


def _same_topology(a, b):
    return bool(a.wrapped.IsSame(b.wrapped))


def _native_edge_face_owners(faces):
    """Index IsSame edge identities once; retain input face order and locations."""
    from OCP.TopTools import TopTools_IndexedMapOfShape
    identities=TopTools_IndexedMapOfShape();owners={}
    for face_number,face in enumerate(faces,1):
        seen=set()
        for edge in face.Edges():
            index=identities.Add(edge.wrapped)
            if index not in seen:
                owners.setdefault(index,[]).append(face_number);seen.add(index)
    return identities,owners


def validate_shared_boundaries(shared_boundaries, default_tolerance):
    """Validate explicit pairs of shared native edges without moving them."""
    if shared_boundaries is None:
        return []
    if not isinstance(shared_boundaries, (list, tuple)):
        raise ValueError('shared_boundaries must be a list of explicit edge-pair records')
    from OCP.TopTools import TopTools_IndexedMapOfShape
    reports,used_ids,used_labels=[],TopTools_IndexedMapOfShape(),{}
    for number, item in enumerate(shared_boundaries):
        if not isinstance(item, dict) or 'edge_a' not in item or 'edge_b' not in item:
            raise ValueError('each shared boundary requires edge_a and edge_b')
        boundary_id = str(item.get('id', f'boundary-{number+1}'))
        alignment = item.get('alignment')
        if alignment not in ('same', 'reverse'):
            raise ValueError(f'{boundary_id}: explicit alignment must be same or reverse')
        a, b = item['edge_a'], item['edge_b']
        samples = int(item.get('samples', 9))
        if samples < 3 or samples > 10001:
            raise ValueError(f'{boundary_id}: samples must be in [3,10001]')
        limit = float(item.get('max_gap_mm', default_tolerance))
        if not np.isfinite(limit) or limit <= 0:
            raise ValueError(f'{boundary_id}: max_gap_mm must be positive and finite')
        same_topology = _same_topology(a, b)
        pa, pb = _edge_points(a, samples), _edge_points(b, samples)
        # CadQuery evaluates a reversed copy with the underlying curve's native
        # parameter direction. For one reused TopoDS_Edge, orientation is read
        # from topology; for independent diagnostic curves, align samples.
        orientation_a = str(a.wrapped.Orientation())
        orientation_b = str(b.wrapped.Orientation())
        if not same_topology and alignment == 'reverse':
            pb = pb[::-1]
        gap = np.linalg.norm(pa-pb, axis=1)
        edge_a_check = analyze_polyline(pa, min(limit, default_tolerance))
        edge_b_check = analyze_polyline(pb, min(limit, default_tolerance))
        report = {'id':boundary_id, 'alignment':alignment, 'samples':samples,
                  'max_gap_limit_mm':limit, 'max_gap_mm':float(gap.max()),
                  'rms_gap_mm':float(np.sqrt(np.mean(gap*gap))),
                  'same_topods_edge':same_topology,
                  'source_binding':item.get('source_binding'),
                  'orientation_a':orientation_a, 'orientation_b':orientation_b,
                  'edge_a':edge_a_check, 'edge_b_aligned':edge_b_check}
        reports.append(report)
        for label, check in (('edge_a', edge_a_check), ('edge_b', edge_b_check)):
            if (check['zero_or_short_segments'] or check['adjacent_foldbacks'] or
                    check['self_contacts']):
                raise PatchTopologyError(
                    f'{boundary_id}: {label} is degenerate, folded back, or self-contacting',
                    {'shared_boundaries':reports})
        if gap.max() > limit:
            raise PatchTopologyError(f'{boundary_id}: shared edge gap exceeds its frozen limit', {'shared_boundaries':reports})
        if not same_topology:
            raise PatchTopologyError(f'{boundary_id}: curves coincide but do not reuse one TopoDS_Edge', {'shared_boundaries':reports})
        if same_topology:
            same_orientation = orientation_a == orientation_b
            if (alignment == 'reverse') == same_orientation:
                raise PatchTopologyError(f'{boundary_id}: reused edge has the wrong declared orientation', {'shared_boundaries':reports})
        prior_index=used_ids.FindIndex(a.wrapped)
        if prior_index:
            raise PatchTopologyError(f'{boundary_id}: shared edge already belongs to {used_labels[prior_index]}', {'shared_boundaries':reports})
        # The same_topology gate above proves a and b share one IsSame identity.
        used_labels[used_ids.Add(a.wrapped)]=boundary_id
    return reports


def _computed_shell_closure(shell):
    """Compute closure from topology; the cached Closed flag is diagnostic only."""
    from OCP.BRep import BRep_Tool
    from OCP.BRepCheck import BRepCheck_Shell, BRepCheck_NoError
    checker = BRepCheck_Shell(shell.wrapped)
    closed_status = checker.Closed()
    orientation_status = checker.Orientation()
    topologically_closed = bool(BRep_Tool.IsClosed_s(shell.wrapped))
    return {'stored_closed_flag':bool(shell.Closed()),
            'topologically_closed':topologically_closed,
            'closed_status':str(closed_status),
            'orientation_status':str(orientation_status),
            'closed':bool(topologically_closed and closed_status == BRepCheck_NoError
                          and orientation_status == BRepCheck_NoError)}


def shape_check_report(shape):
    """Return exact BRepCheck statuses and native edge degeneration details."""
    from OCP.BRepCheck import BRepCheck_Analyzer
    from OCP.BRep import BRep_Tool
    from OCP.TopAbs import TopAbs_FACE, TopAbs_EDGE, TopAbs_VERTEX
    from OCP.TopExp import TopExp
    from OCP.TopTools import TopTools_IndexedMapOfShape
    analyzer = BRepCheck_Analyzer(shape.wrapped, True)
    face_errors = []
    for i, face in enumerate(shape.Faces(), 1):
        statuses = [str(value) for value in analyzer.Result(face.wrapped).Status()]
        errors = [value for value in statuses if 'NoError' not in value]
        if errors:
            face_errors.append({'face':i, 'surface_type':face.geomType(), 'statuses':errors,
                                'wire_count':len(face.Wires()), 'edge_count':len(face.Edges())})
    edge_errors = []
    for i, edge in enumerate(shape.Edges(), 1):
        degenerated = bool(BRep_Tool.Degenerated_s(edge.wrapped))
        try:
            length = float(edge.Length())
        except Exception:
            length = None
        if degenerated or length is None or length <= 1e-12 or not edge.isValid():
            edge_errors.append({'edge':i, 'degenerated':degenerated, 'valid':edge.isValid(),
                                'length_mm':length})
    native_tolerances = {
        'vertex':max((float(BRep_Tool.Tolerance_s(v.wrapped)) for v in shape.Vertices()), default=0.0),
        'edge':max((float(BRep_Tool.Tolerance_s(e.wrapped)) for e in shape.Edges()), default=0.0),
        'face':max((float(BRep_Tool.Tolerance_s(f.wrapped)) for f in shape.Faces()), default=0.0)}
    unowned = {}
    for name, kind in [('faces', TopAbs_FACE), ('edges', TopAbs_EDGE), ('vertices', TopAbs_VERTEX)]:
        whole, owned = TopTools_IndexedMapOfShape(), TopTools_IndexedMapOfShape()
        TopExp.MapShapes_s(shape.wrapped, kind, whole)
        for solid in shape.Solids():
            TopExp.MapShapes_s(solid.wrapped, kind, owned)
        unowned[name] = sum(not owned.Contains(whole.FindKey(i)) for i in range(1, whole.Extent()+1))
    maximum = max(native_tolerances.values())
    shell_closure = [_computed_shell_closure(shell) for shell in shape.Shells()]
    return {'valid':bool(analyzer.IsValid()), 'shape_type':shape.ShapeType(),
            'solids':len(shape.Solids()), 'shells':len(shape.Shells()),
            'closed_shells':sum(row['closed'] for row in shell_closure),
            'shell_closure_checks':shell_closure,
            'faces':len(shape.Faces()), 'edges':len(shape.Edges()),
            'face_errors':face_errors, 'edge_errors':edge_errors,
            'native_tolerance_max_mm':native_tolerances,
            'native_tolerance_limit_mm':0.01,
            'native_tolerances_within_limit':bool(np.isfinite(list(native_tolerances.values())).all() and maximum <= 0.01+1e-9),
            'unowned_topology':unowned}


def _require_native_integrity(report):
    if not report['native_tolerances_within_limit']:
        raise PatchTopologyError('native vertex/edge/face tolerance exceeds 0.01 mm', report)
    if any(report['unowned_topology'].values()):
        raise PatchTopologyError('native topology exists outside the complete solid', report)


def face_from_surface_and_wires(surface, outer_wire, inner_wires=(), tolerance=1e-7,
                                return_report=False):
    """Trim a supplied surface with caller-owned wires, preserving their edges."""
    import cadquery as cq
    from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeFace
    if not outer_wire.Closed():
        raise ValueError('outer wire must be closed')
    maker = BRepBuilderAPI_MakeFace(surface, outer_wire.wrapped, True)
    for wire in inner_wires:
        if not wire.Closed():
            raise ValueError('inner wire must be closed')
        maker.Add(wire.wrapped)
    if not maker.IsDone():
        raise PatchTopologyError('native surface/wire face construction failed', {'maker_error':str(maker.Error())})
    face = cq.Face(maker.Face())
    report = shape_check_report(face)
    report['tolerance_mm'] = float(tolerance)
    if not report['valid']:
        raise PatchTopologyError('surface/wire face is invalid', report)
    return (face, report) if return_report else face


def _strict_parameters(values, count, name):
    if values is None:
        return np.linspace(0.0, 1.0, count), 'uniform_index'
    result = np.asarray(values, dtype=float)
    if result.shape != (count,) or not np.isfinite(result).all():
        raise ValueError(f'{name} must be a finite vector matching the grid direction')
    if np.any(np.diff(result) <= 0.0):
        raise ValueError(f'{name} must be strictly increasing without duplicates')
    return result, 'explicit'


def _occt_knots(full_knots):
    """Convert SciPy's repeated knot vector to OCCT knots/multiplicities."""
    values, multiplicities = [], []
    for value in map(float, full_knots):
        if values and value == values[-1]:
            multiplicities[-1] += 1
        else:
            values.append(value)
            multiplicities.append(1)
    return values, multiplicities


def _spline_degree(value, count, name):
    maximum = min(3, count-1)
    if value is None:
        return maximum
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f'{name} must be an integer in [1,{maximum}]')
    value = int(value)
    if value < 1 or value > maximum:
        raise ValueError(f'{name} must be an integer in [1,{maximum}]')
    return value


def _explicit_tensor_surface(grid, u_parameters, v_parameters, tolerance,
                             u_degree=None, v_degree=None):
    from scipy.interpolate import make_interp_spline
    from OCP.gp import gp_Pnt
    from OCP.Geom import Geom_BSplineSurface
    from OCP.TColgp import TColgp_Array2OfPnt
    from OCP.TColStd import TColStd_Array1OfReal, TColStd_Array1OfInteger

    nu, nv = grid.shape[:2]
    u, u_mode = _strict_parameters(u_parameters, nu, 'u_parameters')
    v, v_mode = _strict_parameters(v_parameters, nv, 'v_parameters')
    u_degree = _spline_degree(u_degree, nu, 'u_degree')
    v_degree = _spline_degree(v_degree, nv, 'v_degree')

    # First interpolate all v columns in u.  Interpolate the resulting u
    # control rows in v.  Every row uses the same caller-supplied v vector,
    # so adjacent charts can share an identical parameter contract.
    u_spline = make_interp_spline(u, grid, axis=0, k=u_degree)
    u_coefficients = np.asarray(u_spline.c, dtype=float)
    v_splines = [make_interp_spline(v, u_coefficients[i], axis=0, k=v_degree)
                 for i in range(len(u_coefficients))]
    poles_array = np.stack([spline.c for spline in v_splines], axis=0)
    if not np.isfinite(poles_array).all():
        raise PatchTopologyError('explicit tensor interpolation produced non-finite poles',
                                 {'u_mode':u_mode, 'v_mode':v_mode})

    u_knots, u_mults = _occt_knots(u_spline.t)
    v_knots, v_mults = _occt_knots(v_splines[0].t)
    poles = TColgp_Array2OfPnt(1, poles_array.shape[0], 1, poles_array.shape[1])
    for i in range(poles_array.shape[0]):
        for j in range(poles_array.shape[1]):
            poles.SetValue(i+1, j+1, gp_Pnt(*map(float, poles_array[i, j])))

    def real_array(values):
        result = TColStd_Array1OfReal(1, len(values))
        for i, value in enumerate(values, 1):
            result.SetValue(i, value)
        return result

    def integer_array(values):
        result = TColStd_Array1OfInteger(1, len(values))
        for i, value in enumerate(values, 1):
            result.SetValue(i, value)
        return result

    surface = Geom_BSplineSurface(
        poles, real_array(u_knots), real_array(v_knots),
        integer_array(u_mults), integer_array(v_mults),
        u_degree, v_degree, False, False)
    maximum = 0.0
    worst = None
    for i, ui in enumerate(u):
        for j, vj in enumerate(v):
            point = surface.Value(float(ui), float(vj))
            native_point = np.array([point.X(), point.Y(), point.Z()], dtype=float)
            error = float(np.linalg.norm(native_point-grid[i, j]))
            if error > maximum:
                maximum, worst = error, [i, j]
    report = {'mode':'explicit_tensor_bspline', 'u_parameter_mode':u_mode,
              'v_parameter_mode':v_mode, 'u_degree':u_degree,
              'v_degree':v_degree, 'u_count':nu, 'v_count':nv,
              'source_node_max_deviation_mm':maximum,
              'source_node_worst_index':worst,
              'source_points_moved':False}
    if maximum > tolerance:
        raise PatchTopologyError('explicit tensor surface misses a frozen source node', report)
    return surface, report


def interpolate_face(grid, tolerance, u_parameters=None, v_parameters=None,
                     u_degree=None, v_degree=None):
    """Interpolate a native face, optionally with explicit tensor parameters.

    With no vectors or degrees this preserves the 0.1/0.2 OCCT
    Approx_IsoParametric behavior. Supplying any vector or degree selects the
    tensor path; an omitted direction uses normalized data indices. Degrees
    default to at most cubic and may be reduced to 1. Exact node interpolation
    does not bound between-node overshoot, so callers still need full-surface
    source deviation checks.
    """
    import cadquery as cq
    from OCP.gp import gp_Pnt
    from OCP.TColgp import TColgp_Array2OfPnt
    from OCP.GeomAPI import GeomAPI_PointsToBSplineSurface
    from OCP.Approx import Approx_IsoParametric
    from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeFace
    grid = np.asarray(grid, dtype=float)
    if grid.ndim != 3 or grid.shape[2] != 3 or min(grid.shape[:2]) < 2:
        raise ValueError('grid must be finite Nu x Nv x 3 with Nu,Nv >= 2')
    if not np.isfinite(grid).all():
        raise ValueError('non-finite surface points')
    tolerance = float(tolerance)
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError('tolerance must be positive and finite')
    if (u_parameters is not None or v_parameters is not None or
            u_degree is not None or v_degree is not None):
        surface, report = _explicit_tensor_surface(
            grid, u_parameters, v_parameters, tolerance, u_degree, v_degree)
        maker = BRepBuilderAPI_MakeFace(surface, tolerance)
        if not maker.IsDone():
            raise PatchTopologyError('native explicit tensor face construction failed', report)
        face = cq.Face(maker.Face())
        if not face.isValid():
            report['face_check'] = shape_check_report(face)
            raise PatchTopologyError('invalid native explicit tensor surface face', report)
        return face
    pts = TColgp_Array2OfPnt(1, grid.shape[0], 1, grid.shape[1])
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            pts.SetValue(i+1, j+1, gp_Pnt(*map(float, grid[i,j])))
    builder = GeomAPI_PointsToBSplineSurface()
    builder.Interpolate(pts, Approx_IsoParametric, False)
    if not builder.IsDone():
        raise ValueError('B-spline interpolation failed')
    maker = BRepBuilderAPI_MakeFace(builder.Surface(), tolerance)
    if not maker.IsDone():
        raise ValueError('native surface face construction failed')
    face = cq.Face(maker.Face())
    if not face.isValid():
        raise ValueError('invalid native surface face')
    return face


def sew_solid_with_report(faces, tolerance, shared_boundaries=None):
    import cadquery as cq
    from OCP.BRepBuilderAPI import BRepBuilderAPI_Sewing
    from OCP.BRepLib import BRepLib
    if not isinstance(faces, (list, tuple)) or len(faces) < 4:
        raise ValueError('faces must contain at least four native faces')
    boundary_report = validate_shared_boundaries(shared_boundaries, tolerance)
    input_checks = []
    for i, face in enumerate(faces, 1):
        check = shape_check_report(face); check['face'] = i
        input_checks.append(check)
        if not check['valid']:
            raise PatchTopologyError(f'input face {i} is invalid before sewing',
                                     {'stage':'input_faces', 'shared_boundaries':boundary_report,
                                      'input_face_checks':input_checks})
    # A declared shared boundary is evidence only when its native edge occurs
    # in exactly two of the faces that are about to be sewn.  This prevents an
    # unrelated coincident edge pair from making the seam report look valid.
    edge_identities,edge_owners=_native_edge_face_owners(faces) if boundary_report else (None,{})
    for boundary, declared in zip(boundary_report, shared_boundaries or ()):
        identity=edge_identities.FindIndex(declared['edge_a'].wrapped)
        owners=edge_owners.get(identity,[]).copy()
        boundary['owner_faces'] = owners
        if len(owners) != 2:
            raise PatchTopologyError(
                f"{boundary['id']}: declared shared edge must belong to exactly two input faces",
                {'stage':'shared_boundary_ownership',
                 'shared_boundaries':boundary_report,
                 'input_face_checks':input_checks})
    sew = BRepBuilderAPI_Sewing(tolerance, True, True, True, False)
    for face in faces:
        sew.Add(face.wrapped)
    sew.Perform()
    shell_shape = cq.Shape.cast(sew.SewedShape())
    shells = shell_shape.Shells()
    shell_closure = [_computed_shell_closure(shell) for shell in shells]
    report = {'stage':'sewing', 'tolerance_mm':float(tolerance),
              'shared_boundaries':boundary_report, 'input_face_checks':input_checks,
              'free_edges':int(sew.NbFreeEdges()), 'multiple_edges':int(sew.NbMultipleEdges()),
              'shell_count':len(shells), 'closed_shell_count':sum(row['closed'] for row in shell_closure),
              'shell_closure_checks':shell_closure,
              'sewed_shape_check':shape_check_report(shell_shape)}
    if report['free_edges'] or report['multiple_edges']:
        raise PatchTopologyError('unclosed/nonmanifold patch network after sewing', report)
    if len(shells) != 1 or not shell_closure[0]['closed']:
        raise PatchTopologyError('patch network must form exactly one closed shell', report)
    solid = cq.Solid.makeSolid(shells[0])
    if not BRepLib.OrientClosedSolid_s(solid.wrapped):
        report['solid_check'] = shape_check_report(solid)
        raise PatchTopologyError('cannot orient closed solid', report)
    report['solid_check'] = shape_check_report(solid)
    if not report['solid_check']['valid'] or report['solid_check']['solids'] != 1 or report['solid_check']['closed_shells'] != 1:
        raise PatchTopologyError('sewn result is not one valid closed native solid', report)
    if volume(solid)[0] <= 1e-12:
        raise PatchTopologyError('solid volume must be positive', report)
    _require_native_integrity(report['solid_check'])
    report['status'] = 'valid_closed_solid'
    return solid, report


def sew_solid(faces, tolerance):
    """Backward-compatible 0.1 API; use sew_solid_with_report for evidence."""
    return sew_solid_with_report(faces, tolerance)[0]


def volume(shape):
    from OCP.GProp import GProp_GProps
    from OCP.BRepGProp import BRepGProp
    props = GProp_GProps()
    error = BRepGProp.VolumeProperties_s(shape.wrapped, props, 1e-7, True, False)
    return float(props.Mass()), float(error)


def require_solid(shape):
    report = shape_check_report(shape)
    if not report['valid'] or report['solids'] != 1 or report['closed_shells'] != report['shells']:
        raise PatchTopologyError('expected one valid closed native solid', report)
    if volume(shape)[0] <= 1e-12:
        raise PatchTopologyError('solid volume must be positive', report)
    _require_native_integrity(report)


def vector3(value, name):
    x = np.asarray(value, dtype=float)
    if x.shape != (3,) or not np.isfinite(x).all():
        raise ValueError(f'{name} must be a finite 3-vector')
    return x


def apply_operations(shape, operations):
    """Explicit, source-backed feature recipes; no automatic hole discovery."""
    import cadquery as cq
    for op in operations:
        if not isinstance(op.get('evidence'), str) or not op['evidence'].strip():
            raise ValueError('every analytic feature requires a source evidence description')
        if op['kind'] != 'cylinder_cut':
            raise ValueError('only cylinder_cut analytic operation is supported in 0.1')
        origin = vector3(op['origin'], 'origin'); direction = vector3(op['direction'], 'direction')
        r, h = float(op['radius']), float(op['length'])
        if not np.isfinite([r,h]).all() or min(r,h) <= 0 or np.linalg.norm(direction) < 1e-12:
            raise ValueError('invalid cylinder cut dimensions or direction')
        cutter = cq.Solid.makeCylinder(r, h, cq.Vector(*origin), cq.Vector(*(direction/np.linalg.norm(direction))))
        before = volume(shape)[0]
        shape = shape.cut(cutter)
        require_solid(shape)
        if before-volume(shape)[0] <= 1e-9:
            raise ValueError('declared cylinder cut did not remove material')
    return shape


def read_step_unchecked(path):
    import cadquery as cq
    from OCP.STEPControl import STEPControl_Reader
    from OCP.IFSelect import IFSelect_RetDone
    reader = STEPControl_Reader()
    if reader.ReadFile(str(path)) != IFSelect_RetDone or reader.TransferRoots() < 1:
        raise ValueError('STEP read/transfer failed')
    return cq.Shape.cast(reader.OneShape())


def read_native_unchecked(path):
    """Read STEP or BRep without treating one Solid as proof of validity."""
    import cadquery as cq
    suffix = Path(path).suffix.lower()
    if suffix in ('.brep', '.brp'):
        return cq.Shape.importBrep(str(path))
    if suffix in ('.step', '.stp'):
        return read_step_unchecked(path)
    raise ValueError('native diagnostic input must be STEP/STP or BREP/BRP')


def read_step(path):
    shape = read_step_unchecked(path)
    require_solid(shape)
    return shape


def describe(shape):
    vol, integration_error = volume(shape)
    from OCP.Bnd import Bnd_Box
    from OCP.BRepBndLib import BRepBndLib
    bb = Bnd_Box()
    BRepBndLib.AddOptimal_s(shape.wrapped, bb, False, False)
    x0,y0,z0,x1,y1,z1 = bb.Get()
    return {'valid': shape.isValid(), 'solids': len(shape.Solids()), 'faces': len(shape.Faces()),
            'surface_types': dict(Counter(f.geomType() for f in shape.Faces())),
            'volume_mm3': vol, 'volume_integration_error_estimate': integration_error,
            'bbox_mm': [[x0,y0,z0],[x1,y1,z1]]}


def triangulate(shape, deflection):
    # Native OCCT mesher respects trimmed face domains and holes. Never mesh a
    # trimmed surface's entire UV bounding rectangle. Clear cached coarse mesh.
    from OCP.BRepTools import BRepTools
    from OCP.BRepMesh import BRepMesh_IncrementalMesh
    from OCP.BRep import BRep_Tool
    from OCP.TopLoc import TopLoc_Location
    from OCP.TopAbs import TopAbs_REVERSED
    BRepTools.Clean_s(shape.wrapped)
    # CadQuery Shape.mesh uses relative=True; millimetre error budgets require
    # the native absolute=False-for-isRelative overload instead.
    mesher = BRepMesh_IncrementalMesh(shape.wrapped, deflection, False, 0.05, False)
    if not mesher.IsDone():
        raise ValueError('native absolute-deflection mesher failed')
    vertices, faces = [], []
    for face in shape.Faces():
        location = TopLoc_Location()
        mesh = BRep_Tool.Triangulation_s(face.wrapped, location)
        if mesh is None or mesh.NbTriangles() == 0:
            raise ValueError('a native face is missing from triangulation')
        start = len(vertices)
        for i in range(1, mesh.NbNodes()+1):
            p = mesh.Node(i).Transformed(location.Transformation())
            vertices.append([p.X(),p.Y(),p.Z()])
        for i in range(1, mesh.NbTriangles()+1):
            a,b,c = mesh.Triangle(i).Get()
            if face.wrapped.Orientation() == TopAbs_REVERSED:
                b,c = c,b
            faces.append([start+a-1,start+b-1,start+c-1])
    if not vertices or not faces:
        raise ValueError('native solid tessellation returned no triangles')
    return np.asarray(vertices), np.array(faces, dtype=np.int64)

