"""Read-only diagnostics, detached native restrictions and integration estimates.

Representation hashes, finite boundary evidence, integration estimates and
strict material equality are different claims. No source/BRep is repaired here.
"""
import hashlib
import io
import json
import math


def representation_hash(shape):
    from OCP.BRepTools import BRepTools
    stream = io.BytesIO()
    BRepTools.Write_s(shape.wrapped, stream)
    return hashlib.sha256(stream.getvalue()).hexdigest()


def diagnose_representation(shape):
    from OCP.BRepAdaptor import BRepAdaptor_Surface
    from OCP.BRepTools import BRepTools
    from OCP.GeomAbs import GeomAbs_BSplineSurface
    before = representation_hash(shape)
    records = []
    for index, face in enumerate(shape.Faces()):
        adaptor = BRepAdaptor_Surface(face.wrapped)
        bounds = list(BRepTools.UVBounds_s(face.wrapped))
        row = {'face': index, 'surface_type': str(adaptor.GetType()), 'active_UV_bounds': bounds,
               'C0_knots': [], 'support_C0_knots_outside_active_domain': []}
        if not all(math.isfinite(v) for v in bounds):
            raise ValueError('nonfinite active face domain')
        if adaptor.GetType() == GeomAbs_BSplineSurface:
            surface = adaptor.BSpline()
            row.update(U_degree=surface.UDegree(), V_degree=surface.VDegree(),
                       U_periodic=surface.IsUPeriodic(), V_periodic=surface.IsVPeriodic())
            for axis, degree, count, knot, multiplicity, low, high in (
                    ('U', surface.UDegree(), surface.NbUKnots(), surface.UKnot, surface.UMultiplicity, bounds[0], bounds[1]),
                    ('V', surface.VDegree(), surface.NbVKnots(), surface.VKnot, surface.VMultiplicity, bounds[2], bounds[3])):
                for i in range(2, count):
                    value, mult = float(knot(i)), int(multiplicity(i))
                    if mult >= degree:
                        item = {'axis': axis, 'knot_index': i, 'parameter': value,
                                'multiplicity': mult, 'degree': degree,
                                'crosses_active_domain': low < value < high}
                        row['C0_knots' if item['crosses_active_domain'] else
                            'support_C0_knots_outside_active_domain'].append(item)
        records.append(row)
    after = representation_hash(shape)
    if before != after: raise ValueError('read-only representation diagnostic changed native geometry')
    return {'representation_sha256': before, 'faces': records, 'geometry_changed': False,
            'active_domain_rule': 'strictly inside actual face UV bounds; trim occupancy not inferred',
            'C0_knots_prove_defect': False, 'automatic_repair': 'not-implemented',
            'continuous_geometry_equivalence_proven': False, 'strict_material_equality_proven': False}


def adaptive_mass_properties(shape, epsilons=(1e-5, 1e-7, 1e-9)):
    from OCP.GProp import GProp_GProps
    from OCP.BRepGProp import BRepGProp
    rows = []
    for epsilon in epsilons:
        if isinstance(epsilon, bool) or not math.isfinite(epsilon) or not 0 < epsilon < .001:
            raise ValueError('adaptive integration epsilon must be finite, positive and below .001')
        volume, area = GProp_GProps(), GProp_GProps()
        ve = BRepGProp.VolumeProperties_s(shape.wrapped, volume, epsilon, True, False)
        ae = BRepGProp.SurfaceProperties_s(shape.wrapped, area, epsilon, False)
        values = [float(volume.Mass()), float(area.Mass()), float(ve), float(ae)]
        if not all(math.isfinite(v) for v in values): raise ValueError('nonfinite native mass integral')
        rows.append({'epsilon': epsilon, 'volume_mm3': values[0], 'area_mm2': values[1],
                     'volume_relative_error_estimate': values[2], 'area_relative_error_estimate': values[3]})
    if not rows: raise ValueError('adaptive mass comparison requires integration levels')
    last = rows[-1]
    uncertainty = {}
    for name in ('volume', 'area'):
        key = name+('_mm3' if name == 'volume' else '_mm2')
        drift = abs(last[key]-rows[-2][key]) if len(rows) > 1 else 0.
        uncertainty[name] = max(drift, abs(last[key]*last[name+'_relative_error_estimate']))
    return {'levels': rows, 'reported_integration_uncertainty': uncertainty,
            'uncertainty_is_rigorous_bound': False}


def compare_mass_properties(source, target, epsilons=(1e-5, 1e-7, 1e-9)):
    left, right = adaptive_mass_properties(source, epsilons), adaptive_mass_properties(target, epsilons)
    return {'source': left, 'target': right,
        'volume_difference_mm3': abs(left['levels'][-1]['volume_mm3']-right['levels'][-1]['volume_mm3']),
        'area_difference_mm2': abs(left['levels'][-1]['area_mm2']-right['levels'][-1]['area_mm2']),
        'strict_material_equality_proven': False,
        'interpretation': 'adaptive integration estimates are representation-sensitive diagnostics, not material equality'}


def _bspline_control_data(geometry):
    """Full native basis/control data, including rational weights and all knots."""
    from OCP.Geom import Geom_BSplineCurve, Geom_BSplineSurface
    def point(p):
        values = [float(v) for v in p.Coord()]
        if not all(math.isfinite(v) for v in values):
            raise ValueError('nonfinite native BSpline control point')
        return values
    if isinstance(geometry, Geom_BSplineCurve):
        data = {'type': 'Geom_BSplineCurve', 'degree': geometry.Degree(),
                'periodic': geometry.IsPeriodic(), 'rational': geometry.IsRational(),
                'bounds': [geometry.FirstParameter(), geometry.LastParameter()],
                'knots': [geometry.Knot(i) for i in range(1, geometry.NbKnots()+1)],
                'multiplicities': [geometry.Multiplicity(i) for i in range(1, geometry.NbKnots()+1)],
                'poles': [point(geometry.Pole(i)) for i in range(1, geometry.NbPoles()+1)],
                'weights': [geometry.Weight(i) for i in range(1, geometry.NbPoles()+1)]}
    elif isinstance(geometry, Geom_BSplineSurface):
        data = {'type': 'Geom_BSplineSurface', 'U_degree': geometry.UDegree(), 'V_degree': geometry.VDegree(),
                'U_periodic': geometry.IsUPeriodic(), 'V_periodic': geometry.IsVPeriodic(),
                'U_rational': geometry.IsURational(), 'V_rational': geometry.IsVRational(),
                'bounds': list(geometry.Bounds()),
                'U_knots': [geometry.UKnot(i) for i in range(1, geometry.NbUKnots()+1)],
                'V_knots': [geometry.VKnot(i) for i in range(1, geometry.NbVKnots()+1)],
                'U_multiplicities': [geometry.UMultiplicity(i) for i in range(1, geometry.NbUKnots()+1)],
                'V_multiplicities': [geometry.VMultiplicity(i) for i in range(1, geometry.NbVKnots()+1)],
                'poles': [[point(geometry.Pole(i,j)) for j in range(1, geometry.NbVPoles()+1)]
                          for i in range(1, geometry.NbUPoles()+1)],
                'weights': [[geometry.Weight(i,j) for j in range(1, geometry.NbVPoles()+1)]
                            for i in range(1, geometry.NbUPoles()+1)]}
    else:
        raise TypeError('only native Geom_BSplineCurve or Geom_BSplineSurface geometry is supported')
    # allow_nan=False verifies finite parameters and weights as well as XYZ.
    serialized = json.dumps(data, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    return data, hashlib.sha256(serialized).hexdigest()


def restrict_active_bspline(geometry, bounds, tolerance_mm=.01):
    """Return ``(restricted_copy, report)`` using native Copy + Segment only.

    Bounds must be the caller's explicit complete intended domain: ``(u0,u1)``
    for a curve or ``(u0,u1,v0,v1)`` for a surface. Completeness of actual BRep
    usage is *not* inferred or checked. Periodic geometry, extrapolation and
    bounds snapping/clamping are rejected. The source and all BRep topology
    remain untouched; this primitive never installs the returned geometry.
    """
    from itertools import product
    from pathlib import Path
    import OCP
    from OCP.Geom import Geom_BSplineCurve, Geom_BSplineSurface
    if isinstance(tolerance_mm, bool) or not isinstance(tolerance_mm, (int, float)) or not math.isfinite(tolerance_mm) or not 0 < tolerance_mm <= .01:
        raise ValueError('tolerance_mm must be finite, positive and at most .01')
    original, original_hash = _bspline_control_data(geometry)
    curve = isinstance(geometry, Geom_BSplineCurve)
    if (geometry.IsPeriodic() if curve else geometry.IsUPeriodic() or geometry.IsVPeriodic()):
        raise ValueError('periodic BSpline restriction is not supported')
    count = 2 if curve else 4
    if not isinstance(bounds, (tuple, list)) or len(bounds) != count or any(
            isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in bounds):
        raise ValueError('explicit complete bounds require '+str(count)+' finite numeric parameters')
    bounds = list(map(float, bounds))
    for axis in range(0, count, 2):
        low, high = bounds[axis:axis+2]
        source_low, source_high = original['bounds'][axis:axis+2]
        if not source_low <= low < high <= source_high:
            raise ValueError('bounds must increase strictly within the original native parameter domain; no clamping/extrapolation')
    restricted = geometry.Copy()
    if not isinstance(restricted, Geom_BSplineCurve if curve else Geom_BSplineSurface):
        raise RuntimeError('native Copy did not retain the concrete BSpline type')
    # Parameter tolerance is deliberately zero. A physical XYZ tolerance must
    # never be silently substituted for a parameter tolerance or used to snap
    # a requested domain to a neighbouring knot.
    try:
        if curve: restricted.Segment(*bounds, 0.)
        else: restricted.Segment(*bounds, 0., 0.)
    finally:
        _, after_hash = _bspline_control_data(geometry)
        if after_hash != original_hash:
            raise RuntimeError('native copy/segmentation unexpectedly mutated the input BSpline')
    result, result_hash = _bspline_control_data(restricted)
    if result['bounds'] != bounds:
        raise RuntimeError('native Segment did not retain the exact requested bounds; no snapped result is accepted')
    axes = [[low+(high-low)*i/32 for i in range(33)]
            for low, high in zip(bounds[::2], bounds[1::2])]
    # Use the supplied endpoints exactly (rather than rely on arithmetic
    # reproducing them), including for very narrow or badly conditioned spans.
    for axis, low, high in zip(axes, bounds[::2], bounds[1::2]):
        axis[0], axis[-1] = low, high
    checks = []
    for parameters in product(*axes):
        before = [float(v) for v in geometry.Value(*parameters).Coord()]
        after = [float(v) for v in restricted.Value(*parameters).Coord()]
        if not all(math.isfinite(v) for v in before+after):
            raise ValueError('nonfinite actual native evaluation after parameter restriction')
        checks.append({'parameters': list(parameters), 'original_point_mm': before,
                       'restricted_point_mm': after, 'distance_mm': math.dist(before, after)})
    worst = max(checks, key=lambda row: row['distance_mm'])
    if worst['distance_mm'] > tolerance_mm:
        raise ValueError('native Segment finite point deviation exceeds tolerance_mm: '+str(worst['distance_mm']))
    report = {'schema': 'guided-mesh-cad.native-bspline-restriction.v1',
        'algorithm': 'OCCT '+original['type']+'.Copy + Segment',
        'native_parameter_tolerance': 0., 'OCP_version': OCP.__version__,
        'implementation_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'original_control_data_sha256': original_hash, 'restricted_control_data_sha256': result_hash,
        'original_bounds': original['bounds'], 'requested_bounds': bounds, 'restricted_bounds': result['bounds'],
        'original_poles': len(original['poles']) if curve else [len(original['poles']), len(original['poles'][0])],
        'restricted_poles': len(result['poles']) if curve else [len(result['poles']), len(result['poles'][0])],
        'finite_point_check': {'parameters_per_axis': 33, 'samples': len(checks),
            'maximum_distance_mm': worst['distance_mm'], 'tolerance_mm': float(tolerance_mm),
            'worst_parameters': worst['parameters'], 'evaluations': checks},
        'input_unchanged': True, 'fitting_performed': False, 'bounds_clamped': False,
        'BRep_topology_replaced': False, 'caller_complete_usage_domain_verified': False,
        'continuous_geometry_equivalence_proven': False, 'strict_material_equality_proven': False,
        'limitations': ['Only nonperiodic native BSpline geometry is supported.',
            'Caller must identify the complete intended parameter domain across every use.',
            'Finite actual point checks are not a proof over the continuous domain.',
            'Returned geometry is detached; BRep topology, pcurves and tolerances are not replaced or validated.']}
    return restricted, report


def repair_representation(*args, **kwargs):
    raise NotImplementedError('automatic BRep representation repair is not implemented; no refit/heal or topology replacement is performed')
