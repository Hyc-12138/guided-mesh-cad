"""Owned native boundary witnesses; finite certificates, never material equality.

Kernel imports are lazy. An actual point on the requested shape supplies an
upper bound; it need not be a nearest point. Unmatched queries use every face
of that shape, with original periodic trim evaluation as a conservative fallback.
"""
import gzip
import hashlib
import json
import math
from itertools import product
from pathlib import Path


def file_binding(path):
    path = Path(path).resolve()
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return {'path': str(path), 'sha256': digest.hexdigest(), 'bytes': path.stat().st_size}


def write_evidence(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, 'xt', encoding='utf-8') as stream:
        json.dump(value, stream, allow_nan=False, separators=(',', ':'))
    return file_binding(path)


def xyz(point):
    value = [float(v) for v in point]
    if len(value) != 3 or not all(math.isfinite(v) for v in value):
        raise ValueError('native query must be a finite XYZ point')
    return value


def sample_native_boundary(shape):
    """Sample actual trims. Failed edges and unsampled faces stay explicit."""
    from OCP.BRep import BRep_Tool
    from OCP.BRepAdaptor import BRepAdaptor_Curve, BRepAdaptor_Surface
    from OCP.BRepClass import BRepClass_FaceClassifier
    from OCP.BRepTools import BRepTools
    from OCP.TopAbs import TopAbs_IN
    from .native_trim import classify_native_uv
    points, provenance, failed_edges, face_records = [], [], [], []
    def add(point, proof):
        points.append(xyz(point)); provenance.append(proof)
    for i, vertex in enumerate(shape.Vertices()):
        add(vertex.toTuple(), {'kind': 'owned_native_vertex', 'vertex': i})
    for i, edge in enumerate(shape.Edges()):
        if BRep_Tool.Degenerated_s(edge.wrapped):
            # Degenerate edges already contribute their actual owned vertices.
            continue
        try:
            curve = BRepAdaptor_Curve(edge.wrapped)
            a, b = curve.FirstParameter(), curve.LastParameter()
            u = float(edge.paramAt(.5))
            if not a <= u <= b and curve.IsPeriodic():
                u += math.ceil((a-u)/curve.Period())*curve.Period()
            if not all(math.isfinite(v) for v in (a, b, u)) or not a <= u <= b:
                raise ValueError('edge arc-length parameter outside actual trim')
            add(curve.Value(u).Coord(), {'kind': 'owned_trimmed_edge', 'edge': i,
                'parameter': u, 'actual_trim': [a, b], 'arc_length_fraction': .5})
        except Exception as error:
            failed_edges.append({'edge': i, 'error': str(error)})
    for i, face in enumerate(shape.Faces()):
        uv = BRepTools.UVBounds_s(face.wrapped)
        if not all(math.isfinite(v) for v in uv):
            raise ValueError('unbounded/nonfinite native face UV domain')
        surface, classifier = BRepAdaptor_Surface(face.wrapped), BRepClass_FaceClassifier()
        found = 0
        for grid in (3, 9, 27):
            for j in range(grid):
                u = uv[0]+(uv[1]-uv[0])*(j+.5)/grid
                for k in range(grid):
                    v = uv[2]+(uv[3]-uv[2])*(k+.5)/grid
                    if classify_native_uv(classifier, face.wrapped, u, v) == TopAbs_IN:
                        add(surface.Value(u, v).Coord(), {'kind': 'owned_trimmed_face_IN',
                            'face': i, 'uv': [u, v], 'grid': grid,
                            'classifier_tolerance_mm': 1e-9})
                        found += 1
                        if found == 5: break
                if found == 5: break
            if found: break
        face_records.append({'face': i, 'interior_samples': found, 'last_grid': grid})
    if not points:
        raise ValueError('native boundary has no finite actual samples')
    return points, provenance, {'vertices': sum(p['kind'] == 'owned_native_vertex' for p in provenance),
        'edge_samples': sum(p['kind'] == 'owned_trimmed_edge' for p in provenance),
        'face_records': face_records, 'failed_edge_samples': failed_edges,
        'faces_without_interior_hit': [r['face'] for r in face_records if not r['interior_samples']],
        'continuous_surface_coverage': False, 'finite_only': True}


class PeriodicEdgeWitness:
    """Evaluate only parameters mapped into the actual original periodic trim."""
    def __init__(self, shape):
        from OCP.BRep import BRep_Tool
        from OCP.TopLoc import TopLoc_Location
        self.edges = []
        for index, edge in enumerate(shape.Edges()):
            loc = TopLoc_Location()
            curve = BRep_Tool.Curve_s(edge.wrapped, loc, 0., 0.)
            if curve is None or not curve.IsPeriodic(): continue
            a, b = BRep_Tool.Range_s(edge.wrapped)
            period, base = curve.Period(), curve.FirstParameter()
            if not all(math.isfinite(v) for v in (a, b, period, base)) or not a <= b or period <= 0:
                raise ValueError('invalid actual periodic edge domain')
            self.edges.append({'edge': index, 'curve': curve, 'location': loc,
                               'a': a, 'b': b, 'period': period, 'base': base})

    def one_edge(self, entry, point):
        from OCP.GeomAPI import GeomAPI_ProjectPointOnCurve
        from OCP.gp import gp_Pnt
        a, b, period, base = (entry[k] for k in ('a', 'b', 'period', 'base'))
        curve, location = entry['curve'], entry['location']
        query = gp_Pnt(*xyz(point))
        local = query.Transformed(location.Transformation().Inverted())
        lo = base+(a-base) % period; hi = lo+min(b-a, period)
        segments = ([(base, base+period)] if b-a >= period else
                    [(lo, hi)] if hi <= base+period else [(lo, base+period), (base, hi-period)])
        parameters, failures = [a, b], []
        for low, high in segments:
            if high-low <= 1e-14: continue
            try:
                projection = GeomAPI_ProjectPointOnCurve(local, curve, low, high)
                for j in range(1, projection.NbPoints()+1):
                    u = projection.Parameter(j)
                    u += math.ceil((a-u)/period)*period
                    if a <= u <= b: parameters.append(u)
            except Exception as error:
                failures.append(str(error))
        candidates = []
        for u in parameters:
            p = curve.Value(u).Transformed(location.Transformation())
            candidates.append({'distance_mm': query.Distance(p), 'actual_owned_point': list(p.Coord()),
                               'original_trim_parameter': u})
        return dict(min(candidates, key=lambda r: r['distance_mm']), edge=entry['edge'],
            actual_trim=[a, b], period=period, normalized_projection_segments=segments,
            projection_failures=failures, original_trim_evaluated=True)

    def query(self, point):
        return min((self.one_edge(e, point) for e in self.edges),
                   key=lambda r: r['distance_mm'], default=None)


def complete_boundary_distance(shape, point, periodic=None, *, boundary=None, tolerance_mm=.01):
    """DSS is run on a compound of *all faces*, never on the solid interior.

    Evaluate the owned support and recompute XYZ distance. Kernel scalar zeros
    and points on an untrimmed surface are not acceptance evidence.
    """
    import cadquery as cq
    from OCP.BRepExtrema import BRepExtrema_DistShapeShape
    from .memory_validation import owned_extrema_point
    if boundary is None:
        faces = shape.Faces()
        if not faces: raise ValueError('complete native boundary requires native faces')
        boundary = cq.Compound.makeCompound(faces)
    point = xyz(point)
    operation = BRepExtrema_DistShapeShape(boundary.wrapped, cq.Vertex.makeVertex(*point).wrapped)
    candidates, failures = [], []
    if operation.IsDone():
        for i in range(1, operation.NbSolution()+1):
            try:
                actual = xyz(owned_extrema_point(operation, boundary, i).Coord())
                candidates.append({'distance_mm': math.dist(point, actual),
                    'actual_owned_point': actual, 'kind': 'complete_face_boundary_DSS_owned_support',
                    'solution': i, 'raw_kernel_distance_mm': operation.Value()})
            except Exception as error:
                failures.append({'solution': i, 'error': str(error)})
    else: failures.append({'error': 'complete boundary DSS did not complete'})
    # A constructive in-tolerance support already settles this finite query.
    # Expensive periodic projections are required only for an unresolved miss.
    if not candidates or min(c['distance_mm'] for c in candidates) > tolerance_mm:
        periodic = periodic if periodic is not None else PeriodicEdgeWitness(shape)
        witness = periodic.query(point)
        if witness:
            candidates.append(dict(witness, kind='actual_original_periodic_trim_upper_bound'))
    if not candidates:
        raise ValueError('no evaluable owned boundary DSS/periodic witness: '+str(failures))
    result = min(candidates, key=lambda r: r['distance_mm'])
    result.update(unevaluable_DSS_supports=failures, complete_face_boundary=True,
                  value_type='constructive_owned_native_point_upper_bound')
    return result


class ActualNativeWitnessOracle:
    def __init__(self, shape, context, cloud_path, tolerance_mm=.01):
        if not context.get('leaf_name') or not context.get('native_source', {}).get('sha256'):
            raise ValueError('native cloud requires exact leaf ownership and source hash')
        if not math.isfinite(tolerance_mm) or not 0 < tolerance_mm <= .01:
            raise ValueError('native tolerance must be positive and at most .01 mm')
        self.shape, self.context, self.tolerance = shape, context, tolerance_mm
        self.points, provenance, self.coverage = sample_native_boundary(shape)
        self.cloud_ref = write_evidence(cloud_path, {'schema': 'guided-mesh-cad.native-cloud.v1',
            'context': context, 'points': self.points, 'provenance': provenance, 'coverage': self.coverage})
        self.bins = {}
        for i, p in enumerate(self.points):
            self.bins.setdefault(tuple(math.floor(v/tolerance_mm) for v in p), []).append(i)
        self.periodic = None
        self.boundary = None

    def batch(self, points, evidence_path, expected_target_leaf):
        if expected_target_leaf != self.context['leaf_name']:
            raise ValueError('cross-leaf native witness borrowing is forbidden')
        points = [xyz(p) for p in points]
        if not points: raise ValueError('native query batch cannot be empty')
        records, reused, cache = [], 0, {}
        for i, point in enumerate(points):
            key = tuple(point)
            if key in cache:
                records.append({'query_index': i, 'exact_query_reuse_of': cache[key][0],
                                'distance_mm': cache[key][1]}); reused += 1; continue
            cell = tuple(math.floor(v/self.tolerance) for v in point)
            best = None
            for delta in product((-1, 0, 1), repeat=3):
                for index in self.bins.get(tuple(cell[k]+delta[k] for k in range(3)), ()):
                    distance = math.dist(point, self.points[index])
                    if distance <= self.tolerance and (best is None or distance < best['distance_mm']):
                        best = {'kind': 'actual_cloud_point_upper_bound', 'distance_mm': distance,
                                'target_witness_index': index, 'actual_owned_point': self.points[index]}
            if best is None:
                if self.periodic is None: self.periodic = PeriodicEdgeWitness(self.shape)
                if self.boundary is None:
                    import cadquery as cq
                    faces = self.shape.Faces()
                    if not faces: raise ValueError('complete native boundary requires native faces')
                    self.boundary = cq.Compound.makeCompound(faces)
                best = complete_boundary_distance(self.shape, point, self.periodic,
                    boundary=self.boundary, tolerance_mm=self.tolerance)
            record = dict(best, query_index=i)
            records.append(record); cache[key] = i, record['distance_mm']
        worst = max(records, key=lambda r: r['distance_mm'])
        evidence = write_evidence(evidence_path, {'schema': 'guided-mesh-cad.native-certificates.v1',
            'cloud': self.cloud_ref, 'leaf_name': expected_target_leaf, 'query_points': points,
            'records': records, 'value_type': 'owned_actual_point_upper_bounds'})
        return {'samples': len(points), 'max_distance_mm': worst['distance_mm'],
            'over_tolerance': sum(r['distance_mm'] > self.tolerance+1e-9 for r in records),
            'worst_sample_index': worst['query_index'], 'worst_point': points[worst['query_index']],
            'max_distance_value_type': 'constructive_owned_native_boundary_point_upper_bound',
            'certificate_evidence': evidence, 'target_cloud': self.cloud_ref,
            'cloud_certificates': sum(r.get('kind') == 'actual_cloud_point_upper_bound' for r in records),
            'complete_boundary_fallbacks': sum(r.get('complete_face_boundary', False) for r in records),
            'exact_query_reuses': reused, 'continuous_surface_equivalence_proven': False}
