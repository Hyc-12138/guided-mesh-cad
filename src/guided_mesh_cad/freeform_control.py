"""Explicit native approximation: samples used as poles, never called interpolation."""
import math
import numpy as np


def make_control_face(control_grid, *, tolerance=1e-7):
    """Make a C2 open-uniform cubic native B-spline face from explicit poles.

    A positive polynomial basis prevents control-hull overshoot, but does not
    guarantee source accuracy, local regularity or global non-self-intersection.
    Coordinate data are reconstruction control poles, not points promised to
    lie on the surface. Existing ``interpolate_face`` behavior is unchanged.
    """
    import cadquery as cq
    from OCP.Geom import Geom_BSplineSurface
    from OCP.TColgp import TColgp_Array2OfPnt
    from OCP.TColStd import TColStd_Array1OfReal, TColStd_Array1OfInteger
    from OCP.gp import gp_Pnt
    from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeFace
    from .cad import shape_check_report, PatchTopologyError

    grid = np.array(control_grid, dtype=float, copy=True)
    if (grid.ndim != 3 or grid.shape[2] != 3 or min(grid.shape[:2]) < 4 or
            not np.isfinite(grid).all()):
        raise ValueError('control_grid must be finite Nu x Nv x 3 with Nu,Nv >=4')
    tolerance = float(tolerance)
    if not math.isfinite(tolerance) or tolerance <= 0:
        raise ValueError('tolerance must be positive and finite')
    nu, nv = grid.shape[:2]
    poles = TColgp_Array2OfPnt(1, nu, 1, nv)
    for i in range(nu):
        for j in range(nv):
            poles.SetValue(i+1, j+1, gp_Pnt(*map(float, grid[i, j])))

    def knots(count):
        values = np.linspace(0., 1., count-2)
        kk = TColStd_Array1OfReal(1, len(values))
        mm = TColStd_Array1OfInteger(1, len(values))
        for i, value in enumerate(values, 1):
            kk.SetValue(i, float(value))
            mm.SetValue(i, 4 if i in (1, len(values)) else 1)
        return kk, mm

    ku, mu = knots(nu)
    kv, mv = knots(nv)
    surface = Geom_BSplineSurface(poles, ku, kv, mu, mv, 3, 3, False, False)
    maker = BRepBuilderAPI_MakeFace(surface, tolerance)
    if not maker.IsDone():
        raise ValueError('native control-surface face construction failed')
    face = cq.Face(maker.Face())
    if not face.isValid():
        raise PatchTopologyError('invalid native control-surface face', shape_check_report(face))
    return face
