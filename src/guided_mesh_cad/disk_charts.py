"""Evidence-bound charts for a complete triangulated topological disk.

This prepares reconstruction samples, not a solid or a geometry acceptance.
Semantic corners and an ordered complete boundary remain caller supplied.
"""
from collections import Counter, defaultdict
import hashlib
import math

import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve

from .charts import _binding


def _cross2(a, b):
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


def _indices(value, name, ndim):
    raw = np.asarray(value)
    if raw.ndim != ndim or raw.dtype.kind not in 'iu' or raw.dtype.kind == 'b':
        raise ValueError(f'{name} must be an integer array with {ndim} dimensions')
    return raw.astype(np.int64, copy=True)


def _disk_topology(points, triangles, boundary):
    if triangles.shape[1:] != (3,) or not len(triangles):
        raise ValueError('triangles must be nonempty Nx3')
    if triangles.min() < 0 or triangles.max() >= len(points):
        raise ValueError('triangle index out of range')
    if len(np.unique(triangles)) != len(points):
        raise ValueError('all supplied source vertices must be used; select the intended domain explicitly')
    if any(len(set(map(int, t))) != 3 for t in triangles):
        raise ValueError('triangle repeats a vertex')
    if len({tuple(sorted(map(int, t))) for t in triangles}) != len(triangles):
        raise ValueError('duplicate source triangle')
    raw = points[triangles]
    areas = np.linalg.norm(np.cross(raw[:, 1]-raw[:, 0], raw[:, 2]-raw[:, 0]), axis=1)
    if not np.isfinite(areas).all() or np.any(areas <= 0):
        raise ValueError('zero-area or nonfinite source triangle')
    owners = defaultdict(list)
    adjacency = [set() for _ in points]
    links = [[] for _ in points]
    for ti, tri in enumerate(triangles):
        for i, j in ((0, 1), (1, 2), (2, 0)):
            a, b = int(tri[i]), int(tri[j])
            owners[tuple(sorted((a, b)))].append((ti, a, b))
            adjacency[a].add(b)
            adjacency[b].add(a)
        for i in range(3):
            links[int(tri[i])].append((int(tri[(i+1) % 3]), int(tri[(i+2) % 3])))
    if any(len(x) > 2 for x in owners.values()):
        raise ValueError('nonmanifold source edge')
    if any(len(x) == 2 and x[0][1:] == x[1][1:] for x in owners.values()):
        raise ValueError('inconsistent source triangle orientation')
    edges = {e for e, x in owners.items() if len(x) == 1}
    if (len(boundary) < 3 or len(set(map(int, boundary))) != len(boundary) or
            boundary.min() < 0 or boundary.max() >= len(points)):
        raise ValueError('boundary_cycle must contain unique valid vertex indices without a repeated closure')
    declared = {tuple(sorted((int(a), int(b)))) for a, b in zip(boundary, np.roll(boundary, -1))}
    if len(declared) != len(boundary) or declared != edges:
        raise ValueError('boundary_cycle must cover every actual boundary edge exactly once')
    if len(points)-len(owners)+len(triangles) != 1:
        raise ValueError('source domain must be a topological disk (Euler characteristic 1)')
    visited, todo = set(), [0]
    while todo:
        node = todo.pop()
        if node not in visited:
            visited.add(node)
            todo.extend(adjacency[node]-visited)
    if len(visited) != len(points):
        raise ValueError('disconnected source domain')
    boundary_set = set(map(int, boundary))
    for vi, link_edges in enumerate(links):
        link_adj = defaultdict(list)
        for a, b in link_edges:
            link_adj[a].append(b)
            link_adj[b].append(a)
        seen, queue = set(), [next(iter(link_adj))]
        while queue:
            node = queue.pop()
            if node not in seen:
                seen.add(node)
                queue.extend(x for x in link_adj[node] if x not in seen)
        degrees = Counter(map(len, link_adj.values()))
        valid = (degrees.get(1) == 2 and set(degrees) <= {1, 2}) if vi in boundary_set else set(degrees) == {2}
        if len(seen) != len(link_adj) or not valid:
            raise ValueError(f'nonmanifold source vertex link at {vi}')
    return adjacency, owners


class _TriangleLocator:
    """Complete original-triangle AABB bins; never retriangulate the source."""
    def __init__(self, uv, triangles):
        self.uv, self.triangles = uv, triangles
        self.t = uv[triangles]
        self.lo, self.hi = uv.min(axis=0), uv.max(axis=0)
        self.width = self.hi-self.lo
        self.count = max(1, min(128, int(math.sqrt(len(triangles)))))
        self.bins = defaultdict(list)
        lo = self._cell(self.t.min(axis=1))
        hi = self._cell(self.t.max(axis=1))
        for ti, (a, b) in enumerate(zip(lo, hi)):
            for x in range(a[0], b[0]+1):
                for y in range(a[1], b[1]+1):
                    self.bins[(x, y)].append(ti)

    def _cell(self, point):
        result = np.floor((point-self.lo)/self.width*self.count).astype(int)
        return np.clip(result, 0, self.count-1)

    def locate(self, point):
        if np.any(point < self.lo-1e-12) or np.any(point > self.hi+1e-12):
            raise ValueError('quad parameter point is outside the original UV domain')
        candidates = self.bins[tuple(self._cell(point))]
        for ti in candidates:
            t = self.t[ti]
            matrix = np.column_stack((t[1]-t[0], t[2]-t[0]))
            b = np.linalg.solve(matrix, point-t[0])
            weights = np.r_[1-b.sum(), b]
            if weights.min() >= -1e-12 and weights.max() <= 1+1e-12:
                return ti, weights
        raise ValueError('no original triangle contains the quad parameter point; no domain fill or UV clamp')


def prepare_source_disk_quads(vertices, triangles, boundary_cycle, anchor_indices,
                              *, samples=33, source_binding=None):
    """Turn one complete source disk into one quad chart per semantic corner.

    ``boundary_cycle`` lists all actual boundary vertex indices, without a
    repeated closing vertex. ``anchor_indices`` indexes that cycle in cyclic
    order. Three or more anchors are required. No axis, section, or height-graph
    assumption is made. Inputs are never welded, reordered, or modified.

    Returns arrays under ``grids``, ``source_uv`` and ``lineage`` plus a JSON
    ``report``. Every grid point retains an original triangle and barycentric
    coordinates. Native interpolation can overshoot; actual CAD fidelity,
    regularity, inter-patch boundary coordination and solid checks are separate.
    """
    points = np.array(vertices, dtype=float, copy=True)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 3 or not np.isfinite(points).all():
        raise ValueError('vertices must be finite Nx3')
    tri = _indices(triangles, 'triangles', 2)
    boundary = _indices(boundary_cycle, 'boundary_cycle', 1)
    anchors = _indices(anchor_indices, 'anchor_indices', 1)
    if (isinstance(samples, bool) or not isinstance(samples, (int, np.integer)) or
            not 4 <= samples <= 257):
        raise ValueError('samples must be an integer in [4,257]')
    if (len(anchors) < 3 or len(np.unique(anchors)) != len(anchors) or
            anchors.min() < 0 or anchors.max() >= len(boundary)):
        raise ValueError('three or more distinct valid semantic boundary anchors are required')
    advances = (np.roll(anchors, -1)-anchors) % len(boundary)
    if np.any(advances == 0) or advances.sum() != len(boundary):
        raise ValueError('anchors must follow the boundary for exactly one lap')
    binding = _binding(source_binding)
    adjacency, edge_owners = _disk_topology(points, tri, boundary)
    count = len(anchors)
    uv = np.full((len(points), 2), np.nan)
    chains = []
    for ci, (first, advance) in enumerate(zip(anchors, advances)):
        positions = (int(first)+np.arange(int(advance)+1)) % len(boundary)
        ids = boundary[positions]
        lengths = np.linalg.norm(np.diff(points[ids], axis=0), axis=1)
        if np.any(lengths <= 0):
            raise ValueError('zero-length source boundary segment')
        t = np.r_[0., np.cumsum(lengths)]
        t /= t[-1]
        angles = (ci+t)*(2*np.pi/count)
        uv[ids] = np.column_stack((np.cos(angles), np.sin(angles)))
        chains.append((ids, t))
    boundary_set = set(map(int, boundary))
    interior = np.array([i for i in range(len(points)) if i not in boundary_set], dtype=int)
    lookup = {int(v): i for i, v in enumerate(interior)}
    matrix = lil_matrix((len(interior), len(interior)))
    rhs = np.zeros((len(interior), 2))
    for row, vi in enumerate(interior):
        matrix[row, row] = len(adjacency[vi])
        for other in adjacency[vi]:
            if other in lookup:
                matrix[row, lookup[other]] = -1
            else:
                rhs[row] += uv[other]
    if len(interior):
        uv[interior] = spsolve(matrix.tocsr(), rhs)
    signed = _cross2(uv[tri[:, 1]]-uv[tri[:, 0]], uv[tri[:, 2]]-uv[tri[:, 0]])
    if not np.isfinite(uv).all() or not (np.all(signed > 1e-14) or np.all(signed < -1e-14)):
        raise ValueError('harmonic UV map contains a collapsed or reversed source triangle')
    locator = _TriangleLocator(uv, tri)

    def chain_value(ci, parameter):
        ids, t = chains[ci]
        interval = min(len(t)-2, max(0, int(np.searchsorted(t, parameter, side='right')-1)))
        fraction = (parameter-t[interval])/(t[interval+1]-t[interval])
        a, b = int(ids[interval]), int(ids[interval+1])
        ti = edge_owners[tuple(sorted((a, b)))][0][0]
        bary = np.zeros(3)
        bary[np.flatnonzero(tri[ti] == a)[0]] = 1-fraction
        bary[np.flatnonzero(tri[ti] == b)[0]] = fraction
        return bary @ uv[tri[ti]], bary @ points[tri[ti]], ti, bary

    midpoints = [chain_value(ci, .5)[0] for ci in range(count)]
    parameter = np.linspace(0., 1., int(samples))
    grids, lineage, domain_grids = [], [], []
    for ci in range(count):
        corner = chain_value(ci, 0.)[0]
        next_mid, previous_mid = midpoints[ci], midpoints[(ci-1) % count]
        grid = np.empty((samples, samples, 3))
        domain = np.empty((samples, samples, 2))
        tids = np.empty((samples, samples), dtype=np.int64)
        barycentric = np.empty((samples, samples, 3))
        for i, u in enumerate(parameter):
            bottom_uv, bottom_p, bottom_ti, bottom_bc = chain_value(ci, .5*u)
            top = (1-u)*previous_mid
            for j, v in enumerate(parameter):
                left_uv, left_p, left_ti, left_bc = chain_value((ci-1) % count, 1-.5*v)
                right = (1-v)*next_mid
                bilinear = (1-u)*(1-v)*corner + u*(1-v)*next_mid + (1-u)*v*previous_mid
                query = (1-v)*bottom_uv + v*top + (1-u)*left_uv + u*right-bilinear
                domain[i, j] = query
                if j == 0:
                    grid[i, j], tids[i, j], barycentric[i, j] = bottom_p, bottom_ti, bottom_bc
                elif i == 0:
                    grid[i, j], tids[i, j], barycentric[i, j] = left_p, left_ti, left_bc
                else:
                    ti, bc = locator.locate(query)
                    grid[i, j], tids[i, j], barycentric[i, j] = bc @ points[tri[ti]], ti, bc
        grids.append(grid)
        domain_grids.append(domain)
        lineage.append({'source_triangle_indices': tids, 'source_barycentric': barycentric})
    gaps = [float(np.linalg.norm(grids[ci][-1]-grids[(ci+1) % count][:, -1], axis=1).max())
            for ci in range(count)]
    scale = max(1., float(np.linalg.norm(np.ptp(points, axis=0))))
    if max(gaps) > 1e-11*scale:
        raise ValueError('common internal chart boundaries disagree; no sewing tolerance change')
    report = {
        'source_binding': binding, 'source_vertices': len(points), 'source_triangles': len(tri),
        'source_arrays_sha256': hashlib.sha256(points.tobytes()+tri.tobytes()).hexdigest(),
        'boundary_vertices': len(boundary), 'semantic_anchors': anchors.tolist(),
        'quad_count': count, 'grid_shape': [int(samples), int(samples)],
        'source_orientation_in_chart': 1 if signed[0] > 0 else -1,
        'minimum_all_original_UV_triangle_twice_area': float(np.abs(signed).min()),
        'all_original_triangles_retained': True, 'internal_boundary_sample_gaps': gaps,
        'native_or_whole_accepted': False,
        'scope': ('Complete manifold source disk, positive-weight harmonic convex-polygon UV map, '
                  'Coons-domain quads with original triangle/barycentric lineage. No source welding, '
                  'retriangulation, domain filling, feature detection, CAD-history recovery, '
                  'native accuracy or 3D/global injectivity claim.')}
    return {'grids': grids, 'source_uv': uv, 'quad_source_uv': domain_grids,
            'lineage': lineage, 'boundary_chains': chains, 'report': report}
