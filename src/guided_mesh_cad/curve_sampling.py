"""Bounded sampling of actual trimmed native curves, without changing them.

For each parameter interval, linear interpolation error is at most
sup(norm(C''))*h*h/8. Bezier derivative control hulls supply the bound;
positive rational weights use an explicit quotient-derivative bound.
These are floating computations of mathematical bounds, not interval
arithmetic certificates or evidence about a subsequently fitted surface.
"""
import json
import numpy as np


def _point(p):
    return np.array([p.X(),p.Y(),p.Z()],dtype=float)


def _split_homogeneous(poles):
    work=poles.copy();left=[work[0].copy()];right=[work[-1].copy()]
    while len(work)>1:
        work=(work[:-1]+work[1:])/2
        left.append(work[0].copy());right.append(work[-1].copy())
    return np.asarray(left),np.asarray(right[::-1])


def _rational_bezier_d2_bound(poles):
    """Bound the second derivative in this segment's local [0,1] parameter."""
    weights=poles[:,3]
    if not np.isfinite(poles).all() or np.any(weights<=0):
        raise ValueError('finite positive rational weights are required')
    weights=weights/weights.max()
    points=poles[:,:3]/poles[:,3,None]
    # Translation does not change C''. This keeps bounds independent of a
    # distant assembly origin and avoids unnecessary quotient cancellation.
    a=(points-points[0])*weights[:,None]
    degree=len(poles)-1;wmin=float(weights.min())
    if degree<2:
        a2=0.;w2=0.
    else:
        a2=float(np.linalg.norm(degree*(degree-1)*np.diff(a,n=2,axis=0),axis=1).max())
        w2=float(np.max(abs(degree*(degree-1)*np.diff(weights,n=2))))
    a1=float(np.linalg.norm(degree*np.diff(a,axis=0),axis=1).max()) if degree else 0.
    w1=float(np.max(abs(degree*np.diff(weights)))) if degree else 0.
    a0=float(np.linalg.norm(a,axis=1).max())
    with np.errstate(over='ignore',divide='ignore',invalid='ignore'):
        wmin=np.float64(wmin)
        value=a2/wmin+2*(a1/wmin)*(w1/wmin)+(a0/wmin)*(w2/wmin)+2*(a0/wmin)*np.square(w1/wmin)
    if not np.isfinite(value):
        raise ValueError('rational derivative bound exceeds finite numerical range')
    return float(value)


def _bezier_poles(curve):
    points=np.array([_point(curve.Pole(i)) for i in range(1,curve.NbPoles()+1)])
    weights=np.array([curve.Weight(i) for i in range(1,curve.NbPoles()+1)],dtype=float)
    if not np.isfinite(points).all() or not np.isfinite(weights).all() or np.any(weights<=0):
        raise ValueError('native Bezier poles/weights are invalid')
    weights/=weights.max()
    return np.c_[points*weights[:,None],weights]


def sample_native_edge_bounded(edge,max_deviation_mm=.0001,*,max_segments=200000,
                               max_depth=30,source_binding=None):
    """Return (points, native_parameters, report) for one native mm edge.

    Supports lines, circles, ellipses, Bezier curves and rational/nonrational
    B-splines, including periodic and trimmed B-splines. Native parameter order
    is increasing; TopoDS orientation is recorded separately. The input curve
    is never fitted, healed, moved, reparameterized or modified.

    Each retained segment has a derivative bound for the full interval, not
    just an endpoint/midpoint test. Same-parameter interpolation gives both
    curve-to-polyline and polyline-to-curve distance upper bounds. Topological
    vertex offsets and pcurves are outside that geometric-curve bound. The
    report exposes endpoint offsets and does not certify a whole solid, a
    later interpolation/loft, assembly fit or manufacturing intent.
    """
    from OCP.BRepAdaptor import BRepAdaptor_Curve
    from OCP.BRep import BRep_Tool
    from OCP.GeomAbs import (GeomAbs_Line,GeomAbs_Circle,GeomAbs_Ellipse,
                             GeomAbs_BezierCurve,GeomAbs_BSplineCurve)
    from OCP.GeomConvert import GeomConvert_BSplineCurveToBezierCurve
    from OCP.TColStd import TColStd_Array1OfReal
    limit=float(max_deviation_mm)
    if not np.isfinite(limit) or limit<=0:
        raise ValueError('max_deviation_mm must be finite and positive')
    if isinstance(max_segments,bool) or not isinstance(max_segments,int) or max_segments<1:
        raise ValueError('max_segments must be a positive integer')
    if isinstance(max_depth,bool) or not isinstance(max_depth,int) or not 1<=max_depth<=50:
        raise ValueError('max_depth must be an integer in [1,50]')
    if source_binding is not None:
        if not isinstance(source_binding,dict) or not isinstance(source_binding.get('source_sha256'),str) or len(source_binding['source_sha256'])!=64 or not isinstance(source_binding.get('selector'),str) or not source_binding['selector'].strip():
            raise ValueError('source_binding requires a frozen native input source_sha256 and selector')
        if any(c not in '0123456789abcdefABCDEF' for c in source_binding['source_sha256']):
            raise ValueError('source_sha256 must contain 64 hexadecimal characters')
        json.dumps(source_binding,allow_nan=False)
    if edge.ShapeType()!='Edge' or not edge.isValid() or BRep_Tool.Degenerated_s(edge.wrapped):
        raise ValueError('one valid nondegenerate native edge is required')
    native_edge_tol=float(BRep_Tool.Tolerance_s(edge.wrapped))
    native_vertex_tol=max((float(BRep_Tool.Tolerance_s(v.wrapped)) for v in edge.Vertices()),default=0.)
    if not np.isfinite([native_edge_tol,native_vertex_tol]).all() or max(native_edge_tol,native_vertex_tol)>.01+1e-9:
        raise ValueError('native edge/vertex tolerance exceeds .01 mm')
    adaptor=BRepAdaptor_Curve(edge.wrapped);first=float(adaptor.FirstParameter());last=float(adaptor.LastParameter())
    if not np.isfinite([first,last]).all() or last<=first:
        raise ValueError('native trimmed parameter range must be finite and increasing')
    kind=adaptor.GetType();arcs=[];conversion_residual=0.
    if kind in (GeomAbs_Line,GeomAbs_Circle,GeomAbs_Ellipse):
        d2=0. if kind==GeomAbs_Line else float(adaptor.Circle().Radius()) if kind==GeomAbs_Circle else float(adaptor.Ellipse().MajorRadius())
        arcs=[(first,last,None,d2)]
    elif kind==GeomAbs_BezierCurve:
        curve=adaptor.Bezier()
        # A trimmed Bezier edge requires its exact trimmed Bezier control hull.
        curve=curve.Copy();curve.Segment(first,last)
        arcs=[(first,last,_bezier_poles(curve),None)]
    elif kind==GeomAbs_BSplineCurve:
        spline=adaptor.BSpline()  # OCP applies the edge location to this copy.
        converter=GeomConvert_BSplineCurveToBezierCurve(spline,first,last,1e-10)
        knots=TColStd_Array1OfReal(1,converter.NbArcs()+1);converter.Knots(knots)
        arcs=[(float(knots.Value(i)),float(knots.Value(i+1)),_bezier_poles(converter.Arc(i)),None) for i in range(1,converter.NbArcs()+1)]
    else:
        raise ValueError('native curve type is unsupported; no approximate conversion substituted')
    if len(arcs)>max_segments:
        raise ValueError('native knot spans already exceed max_segments')
    coordinate_scale=1.
    for ua,ub,poles,_ in arcs:
        ends=[_point(adaptor.Value(ua)),_point(adaptor.Value(ub))]
        coordinate_scale=max(coordinate_scale,*[float(np.linalg.norm(p)) for p in ends])
        if poles is not None:
            ends2=[poles[0,:3]/poles[0,3],poles[-1,:3]/poles[-1,3]]
            conversion_residual=max(conversion_residual,*[float(np.linalg.norm(a-b)) for a,b in zip(ends,ends2)])
    if conversion_residual>1e-8:
        raise ValueError('native Bezier decomposition coordinate/parameter check failed')
    numerical_budget=max(1e-12,128*np.finfo(float).eps*coordinate_scale,4*conversion_residual)
    if limit<=numerical_budget:
        raise ValueError('requested deviation is below the reported numerical budget')
    stack=[(ua,ub,poles,d2,0) for ua,ub,poles,d2 in reversed(arcs)];leaves=[]
    while stack:
        ua,ub,poles,d2,depth=stack.pop()
        local_d2=_rational_bezier_d2_bound(poles) if poles is not None else float(d2)*(ub-ua)**2
        bound=local_d2/8*(1+1e-12)+numerical_budget
        if bound<=limit:
            leaves.append({'first_parameter':ua,'last_parameter':ub,'local_unit_parameter_second_derivative_bound_mm':local_d2,'interpolation_error_upper_bound_mm':bound,'depth':depth})
            continue
        if depth>=max_depth or len(leaves)+len(stack)+2>max_segments:
            raise ValueError('requested native curve bound exceeds subdivision budget; tolerance unchanged')
        mid=ua+(ub-ua)/2
        if not ua<mid<ub:
            raise ValueError('native parameter precision prevents safe subdivision')
        left,right=(None,None) if poles is None else _split_homogeneous(poles)
        stack.append((mid,ub,right,d2,depth+1));stack.append((ua,mid,left,d2,depth+1))
    parameters=np.array([row['first_parameter'] for row in leaves]+[leaves[-1]['last_parameter']],dtype=float)
    if np.any(np.diff(parameters)<=0) or parameters[0]!=first or parameters[-1]!=last:
        raise ValueError('native interval coverage is incomplete or out of order')
    points=np.vstack([_point(adaptor.Value(float(t))) for t in parameters])
    if not np.isfinite(points).all():raise ValueError('native curve evaluation returned nonfinite coordinates')
    endpoint_offsets=[]
    for vertex in edge.Vertices():
        point=np.array(vertex.toTuple(),dtype=float);endpoint_offsets.append(float(np.linalg.norm(points[[0,-1]]-point,axis=1).min()))
    report={'method':'native_curve_derivative_control_bound.v1','units':'mm','curve_type':str(kind),'native_parameter_range':[first,last],'native_topods_orientation':str(edge.wrapped.Orientation()),'segments':len(leaves),'samples':len(points),'max_deviation_limit_mm':limit,'whole_curve_polyline_error_upper_bound_mm':max(row['interpolation_error_upper_bound_mm'] for row in leaves),'native_bezier_spans':len(arcs),'native_decomposition_endpoint_max_residual_mm':conversion_residual,'floating_numerical_budget_mm':numerical_budget,'native_edge_tolerance_mm':native_edge_tol,'native_vertex_tolerance_max_mm':native_vertex_tol,'topological_vertex_endpoint_offsets_mm':endpoint_offsets,'source_binding':source_binding,'intervals':leaves,'input_geometry_modified':False,'bound_scope':'Actual trimmed geometric curve and its same-parameter piecewise linear interpolant, in both directions. Includes every native knot span; no endpoint/midpoint-only acceptance. Floating derivative-control bounds are not interval arithmetic certificates, topology/pcurve bounds, later-surface error bounds or whole-source/solid acceptance.'}
    return points,parameters,report
