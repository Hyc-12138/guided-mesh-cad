from pathlib import Path
import hashlib
import numpy as np


def sha256(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):
            digest.update(chunk)
    return digest.hexdigest()


def load_mesh(path):
    with np.load(path, allow_pickle=False) as data:
        v, f = data['vertices'], data['faces']
    if v.ndim != 2 or v.shape[1] != 3 or len(v) < 4 or not np.isfinite(v).all():
        raise ValueError('vertices must be finite Nx3, with at least four vertices')
    if f.ndim != 2 or f.shape[1] != 3 or len(f) < 4 or f.dtype.kind not in 'iu':
        raise ValueError('faces must be an integer Mx3 triangle array')
    if f.min() < 0 or f.max() >= len(v):
        raise ValueError('triangle index out of bounds')
    v, f = np.asarray(v, dtype=float), np.asarray(f, dtype=np.int64)
    areas = np.linalg.norm(np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]]), axis=1)
    if np.any(areas <= 1e-14):
        raise ValueError('degenerate triangles are not silently removed')
    return v, f


def polydata(v, f):
    import vtk
    from vtk.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray
    points = vtk.vtkPoints()
    points.SetData(numpy_to_vtk(np.ascontiguousarray(v, dtype=float), deep=True))
    cells = vtk.vtkCellArray()
    cells.SetData(numpy_to_vtkIdTypeArray(np.arange(0, (len(f)+1)*3, 3, dtype=np.int64), deep=True),
                  numpy_to_vtkIdTypeArray(np.asarray(f, dtype=np.int64).reshape(-1), deep=True))
    p = vtk.vtkPolyData(); p.SetPoints(points); p.SetPolys(cells)
    return p


def sample_triangles(v, f, n, seed):
    rng = np.random.default_rng(seed)
    t = v[f]
    areas = np.linalg.norm(np.cross(t[:, 1]-t[:, 0], t[:, 2]-t[:, 0]), axis=1)
    ids = rng.choice(len(f), n, p=areas/areas.sum())
    uv = rng.random((n, 2)); mask = uv.sum(axis=1) > 1; uv[mask] = 1-uv[mask]
    return t[ids, 0] + uv[:, :1]*(t[ids, 1]-t[ids, 0]) + uv[:, 1:]*(t[ids, 2]-t[ids, 0])


def mesh_distances(points, v, f):
    import vtk
    locator = vtk.vtkStaticCellLocator(); locator.SetDataSet(polydata(v, f)); locator.BuildLocator()
    closest = [0., 0., 0.]; cell = vtk.reference(0); sub = vtk.reference(0); distance = vtk.reference(0.)
    out = []
    for pt in points:
        locator.FindClosestPoint(pt, closest, cell, sub, distance)
        out.append(float(distance)**0.5)
    return np.asarray(out)


def chart_grid(v, f, patch):
    """Resample an explicitly provided UV triangulation, never its convex hull.

    The actual source triangles determine coverage; samples outside the patch,
    in holes, or in overlapping UV triangles are rejected. Charts are supplied
    by the operator/LLM; this function does not discover them.
    """
    ids = np.asarray(patch['vertex_ids'])
    uv = np.asarray(patch['uv'], dtype=float)
    tri_ids = np.asarray(patch['triangle_ids'])
    if ids.dtype.kind not in 'iu' or ids.ndim != 1 or len(set(ids.tolist())) != len(ids):
        raise ValueError('chart vertex_ids must be unique integers')
    if ids.min() < 0 or ids.max() >= len(v) or uv.shape != (len(ids), 2) or not np.isfinite(uv).all():
        raise ValueError('invalid chart UV / vertex indices')
    if tri_ids.ndim != 1 or tri_ids.dtype.kind not in 'iu' or len(tri_ids) == 0 or tri_ids.min() < 0 or tri_ids.max() >= len(f):
        raise ValueError('invalid chart triangle indices')
    lookup = {int(idx): n for n, idx in enumerate(ids)}
    try:
        local = np.array([[lookup[int(i)] for i in t] for t in f[tri_ids]])
    except KeyError as exc:
        raise ValueError('chart triangle refers to a vertex outside the chart') from exc
    t = uv[local]; a = t[:, 1]-t[:, 0]; b = t[:, 2]-t[:, 0]
    den = a[:, 0]*b[:, 1]-a[:, 1]*b[:, 0]
    if np.any(abs(den) < 1e-14):
        raise ValueError('degenerate UV triangle')
    nu, nv = patch['grid_shape']
    if not isinstance(nu, int) or not isinstance(nv, int) or min(nu, nv) < 4 or max(nu, nv) > 256:
        raise ValueError('grid_shape must have two integers in [4,256]')
    bounds = np.asarray(patch.get('uv_bounds', [[0., 1.], [0., 1.]]), dtype=float)
    if bounds.shape != (2,2) or not np.isfinite(bounds).all() or np.any(bounds[:,1] <= bounds[:,0]):
        raise ValueError('invalid UV bounds')
    out = np.empty((nu, nv, 3))
    for i, u in enumerate(np.linspace(*bounds[0], nu)):
        for j, w in enumerate(np.linspace(*bounds[1], nv)):
            q = [u, w]-t[:, 0]
            s = (q[:,0]*b[:,1]-q[:,1]*b[:,0])/den
            r = (a[:,0]*q[:,1]-a[:,1]*q[:,0])/den
            inside = (s >= -1e-9) & (r >= -1e-9) & (s+r <= 1+1e-9)
            matches = np.flatnonzero(inside)
            if not len(matches):
                raise ValueError(f'UV grid point {(u,w)} lies outside actual patch triangles')
            positions = ((1-s[matches]-r[matches])[:,None]*v[ids[local[matches,0]]] +
                         s[matches,None]*v[ids[local[matches,1]]] + r[matches,None]*v[ids[local[matches,2]]])
            if np.max(np.linalg.norm(positions-positions[0], axis=1)) > 1e-8:
                raise ValueError('overlapping/folded UV chart has ambiguous mesh positions')
            out[i,j] = positions[0]
    return out


def patch_grid(v, f, patch):
    if patch['kind'] == 'uv_chart':
        return chart_grid(v, f, patch)
    if patch['kind'] != 'indexed_grid':
        raise ValueError('patch kind must be indexed_grid or uv_chart')
    ids = np.asarray(patch['indices'])
    if ids.ndim != 2 or ids.dtype.kind not in 'iu' or min(ids.shape) < 4 or max(ids.shape) > 256:
        raise ValueError('indexed_grid requires an integer 4..256 by 4..256 array')
    if ids.min() < 0 or ids.max() >= len(v):
        raise ValueError('grid index out of bounds')
    return v[ids]
