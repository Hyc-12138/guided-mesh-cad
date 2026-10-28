"""Indexed signed winding of an unchanged triangle soup.

The result describes the threshold ``abs(w) > 0.5``. An open, self-intersecting,
or otherwise incomplete soup is NOT thereby certified as a physical solid.
The analytic far-field error bound has floating-point implementation padding;
this is not formal interval arithmetic for arbitrary ill-conditioned inputs.
No vertices move, no triangles disappear, and no holes are filled.
"""
from __future__ import annotations

from numbers import Integral
import hashlib
import numpy as np


def _positive_integer(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _float64_without_loss(raw, name):
    """Reject implicit coordinate movement at the public numeric boundary."""
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        result = np.array(raw, dtype=np.float64, order="C", copy=True)
        restored = result.astype(raw.dtype)
    # Comparing int64 directly with float64 promotes the integer and can hide
    # loss at 2**53 and above. The explicit round trip keeps integer semantics.
    same = np.array_equal(restored, raw, equal_nan=True) if raw.dtype.kind == "f" else np.array_equal(restored, raw)
    if not same:
        raise ValueError(f"{name} cannot be represented losslessly in float64")
    return result


class BoundedWinding:
    """Build a spatial hierarchy while retaining every original triangle.

    NumPy is sufficient. If Numba is installed, the same implementation is JIT
    compiled; ``backend`` reports which path actually loaded. Mesh vertices
    must be finite real Nx3 values and faces must be nonempty integer Mx3
    indices. Empty meshes and silently truncated face indices are rejected.
    Degenerate or repeated triangles are retained, not treated as proof of a
    valid solid. Callers must assess topology separately when that matters.
    """

    def __init__(self, vertices, faces, *, leaf_size=24):
        leaf_size = _positive_integer(leaf_size, "leaf_size")
        raw_v = np.asarray(vertices)
        raw_f = np.asarray(faces)
        if raw_v.ndim != 2 or raw_v.shape[1] != 3 or len(raw_v) < 3:
            raise ValueError("vertices must be a nonempty Nx3 array with at least three vertices")
        if raw_v.dtype.kind not in "iuf" or not np.all(np.isfinite(raw_v)):
            raise ValueError("vertices must contain finite real numeric coordinates")
        if raw_f.ndim != 2 or raw_f.shape[1] != 3 or len(raw_f) == 0:
            raise ValueError("faces must be a nonempty Mx3 array")
        if raw_f.dtype.kind not in "iu":
            raise ValueError("face indices must have an integer dtype; no truncation is performed")
        if np.any(raw_f < 0) or np.any(raw_f >= len(raw_v)):
            raise ValueError("face indices must lie within the vertex array")
        v = _float64_without_loss(raw_v, "vertices")
        f = np.array(raw_f, dtype=np.int64, order="C", copy=True)
        # A finite input can still overflow an intermediate subtraction/cross
        # product. Reject that case explicitly rather than publishing intervals.
        with np.errstate(over="ignore", invalid="ignore"):
            tri = v[f]
            e1, e2 = tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]
            cross = np.cross(e1, e2)
            scale = np.maximum(np.max(np.abs(e1), axis=1), np.max(np.abs(e2), axis=1))
            # Area-vector norms square quadratic quantities. Explicitly reject
            # unsupported scales before an underflowed zero can become OUT.
            machine = np.finfo(np.float64)
            scale_supported = (scale == 0) | ((scale >= machine.tiny ** 0.25) & (scale <= machine.max ** 0.25))
            cross_norm = np.linalg.norm(cross, axis=1)
            lost_nonzero_area = np.any(cross != 0, axis=1) & (cross_norm == 0)
            numeric_ok = np.all(np.isfinite(e1)) and np.all(np.isfinite(e2)) and np.all(np.isfinite(cross))
            numeric_ok = numeric_ok and np.all(np.isfinite(cross_norm)) and np.all(scale_supported) and not np.any(lost_nonzero_area)
        if not numeric_ok:
            raise ValueError("mesh coordinates exceed the supported finite arithmetic range")
        from ._winding_core import _BoundedWindingCore, ACCELERATION_BACKEND
        self._index = _BoundedWindingCore(v, f, leaf_size=leaf_size)
        if any(not np.all(np.isfinite(arr)) for arr in self._index.arrays[:5]):
            raise ValueError("mesh hierarchy exceeds the supported finite arithmetic range")
        self.backend = ACCELERATION_BACKEND
        self.vertex_count = len(v)
        self.triangle_count = len(f)
        self.node_count = self._index.nodes
        self.vertex_sha256 = hashlib.sha256(v.tobytes()).hexdigest()
        self.face_sha256 = hashlib.sha256(f.tobytes()).hexdigest()

    @property
    def original_face_order(self):
        """Read-only original face index for every indexed triangle."""
        result = self._index.original_face_order.copy()
        result.flags.writeable = False
        return result

    def query(self, points, *, chunk_size=2000):
        """Return a winding interval and threshold state for every input point.

        ``state`` is +1 when the entire interval has abs(w)>0.5, -1 when it
        lies within (-0.5,0.5), and 0 when unresolved. Nonfinite query points
        remain unresolved with NaN bounds; they are never silently omitted.
        Values on mesh boundaries need separate distance/topology treatment.
        ``full_atan2`` indicates expansion to every original triangle, not a
        declaration of physical occupancy or exact real arithmetic.
        """
        chunk_size = _positive_integer(chunk_size, "chunk_size")
        raw = np.asarray(points)
        if raw.ndim != 2 or raw.shape[1] != 3 or raw.dtype.kind not in "iuf":
            raise ValueError("query points must be a real numeric Nx3 array")
        p = _float64_without_loss(raw, "query points")
        result = self._index.query(p, chunk=chunk_size)
        if any(len(result[key]) != len(p) for key in result):
            raise RuntimeError("winding query did not retain every input point")
        result["backend"] = self.backend
        result["input_point_count"] = len(p)
        result["triangle_count"] = self.triangle_count
        result["classification_semantics"] = "absolute_signed_winding_threshold_0.5_not_solid_certification"
        return result
