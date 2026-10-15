from pathlib import Path
import json
import platform
import numpy as np
from . import __version__
from .mesh import load_mesh, patch_grid, sha256, sample_triangles, mesh_distances


def dump(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def positive(value, name):
    value = float(value)
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f'{name} must be positive and finite')
    return value


def _failure_record(exc):
    """Serialize failures without breaking the 0.1 ValueError contract."""
    serialized_type = 'ValueError' if isinstance(exc, ValueError) else type(exc).__name__
    failure = {'status':'failed', 'error':str(exc), 'type':serialized_type}
    if hasattr(exc, 'report'):
        failure['diagnostic_type'] = type(exc).__name__
        failure['error_code'] = 'patch_topology_error'
        failure['topology_report'] = exc.report
    return failure


def reconstruct(plan_path, output_dir):
    """Execute a supplied patch plan, bind inputs, export, read back, and verify."""
    from .cad import interpolate_face, sew_solid, apply_operations, describe, read_step
    import cadquery as cq
    plan_path, out = Path(plan_path).resolve(), Path(output_dir).resolve()
    if out.exists() and any(out.iterdir()):
        raise ValueError('output directory must be empty; use a new run directory')
    out.mkdir(parents=True, exist_ok=True)
    try:
        plan = json.loads(plan_path.read_text(encoding='utf-8-sig'))
        if plan.get('schema_version') != 1 or plan.get('units') != 'mm':
            raise ValueError('schema_version=1 and explicit units=mm are required')
        if not isinstance(plan.get('scope'), str) or not plan['scope'].strip():
            raise ValueError('an explicit reconstruction scope is required')
        source = (plan_path.parent / plan['mesh']).resolve()
        digest = sha256(source)
        if digest != plan.get('mesh_sha256'):
            raise ValueError('source hash mismatch; bind plan to the actual input mesh')
        v, f = load_mesh(source)
        construction = None
        if 'construction_mesh' in plan:
            construction = (plan_path.parent / plan['construction_mesh']).resolve()
            if sha256(construction) != plan.get('construction_mesh_sha256'):
                raise ValueError('construction mesh hash mismatch')
            if not isinstance(plan.get('construction_evidence'), str) or not plan['construction_evidence'].strip():
                raise ValueError('separate construction mesh requires explicit inference/provenance evidence')
            v, f = load_mesh(construction)
        tolerance = positive(plan.get('sewing_tolerance_mm', 1e-6), 'sewing tolerance')
        limit = positive(plan.get('verification', {}).get('tolerance_mm', .01), 'verification tolerance')
        if tolerance > min(.001, limit/10):
            raise ValueError('sewing tolerance exceeds the strict geometric limit')
        patches = plan['patches']
        if not isinstance(patches, list) or len(patches) < 4:
            raise ValueError('at least four explicit quadrilateral patches are required')
        names = [p['id'] for p in patches]
        if len(set(names)) != len(names):
            raise ValueError('patch ids must be unique')
        grids, faces = {}, []
        for patch in patches:
            grid = patch_grid(v, f, patch)
            grids[patch['id']] = grid
            faces.append(interpolate_face(grid, tolerance))
        np.savez_compressed(out/'fitted_grids.npz', **grids)
        shape = apply_operations(sew_solid(faces, tolerance), plan.get('operations', []))
        cq.exporters.export(shape, str(out/'model.step'))
        shape.exportBrep(str(out/'model.brep'))
        reloaded = read_step(out/'model.step')
        dump(out/'plan.json', plan)
        manifest = {'package_version': __version__, 'python': platform.python_version(),
                    'source_path': str(source), 'source_sha256': digest, 'plan_sha256': sha256(plan_path),
                    'construction_mesh_path': str(construction) if construction else None,
                    'construction_mesh_sha256': sha256(construction) if construction else None,
                    'construction_evidence': plan.get('construction_evidence'),
                    'saved_plan_sha256': sha256(out/'plan.json'), 'step_sha256': sha256(out/'model.step'),
                    'brep_sha256': sha256(out/'model.brep'), 'scope': plan['scope'],
                    'geometry': describe(reloaded), 'recognition': 'operator-supplied patch chart and feature plan',
                    'status': 'geometry_built_not_yet_verified',
                    'limitations': ['No automatic segmentation or original feature-history recovery',
                                    'Patch seams are position-continuous; tangent/curvature continuity not guaranteed',
                                    'No full-part acceptance from a partial input mesh',
                                    'Finite-sample deviation is not a global Hausdorff bound']}
        dump(out/'manifest.json', manifest)
        result = validate(source, out/'model.step', out/'verification.json', **plan.get('verification', {}))
        manifest['status'] = 'sampled_geometry_pass' if result['passed'] else 'sampled_geometry_fail'
        manifest['verification_sha256'] = sha256(out/'verification.json')
        dump(out/'manifest.json', manifest)
        return manifest
    except Exception as exc:
        # PatchTopologyError is a ValueError subtype.  Keep the 0.1 on-disk
        # failure contract stable while retaining the richer 0.2 subtype and
        # structured topology evidence in additive fields.
        failure = _failure_record(exc)
        dump(out/'failure.json', failure)
        raise


def validate(source_path, step_path, report_path, tolerance_mm=.01, samples=3000, seed=16449,
             diagnostics_dir=None):
    """Full input mesh vs all trimmed native faces; no cap/window exclusions."""
    import cadquery as cq
    from .cad import read_step, triangulate, describe
    tolerance_mm = positive(tolerance_mm, 'verification tolerance')
    if not isinstance(samples, int) or samples < 100 or samples > 1000000:
        raise ValueError('samples must be an integer in [100,1000000]')
    diagnostic_out = Path(diagnostics_dir) if diagnostics_dir is not None else None
    if diagnostic_out is not None:
        if diagnostic_out.exists() and any(diagnostic_out.iterdir()):
            raise ValueError('diagnostics directory must be empty; preserve frozen prior evidence')
        diagnostic_out.mkdir(parents=True, exist_ok=True)
    v, f = load_mesh(source_path); shape = read_step(step_path)
    # Deflection is a requested mesher setting, not an independently certified bound.
    mv, mf = triangulate(shape, tolerance_mm/10)
    source_points = np.vstack([v, sample_triangles(v,f,samples,seed)])
    model_points = np.vstack([mv, sample_triangles(mv,mf,samples,seed+1)])
    forward = mesh_distances(source_points, mv,mf)
    reverse = mesh_distances(model_points, v,f)
    boundary = cq.Compound.makeCompound(shape.Faces())
    worst = np.argsort(forward)[-min(8,len(forward)):]
    native = [float(boundary.distance(cq.Vertex.makeVertex(*map(float, source_points[i])))) for i in worst]
    def stats(a):
        return {'count':len(a), 'max_mm':float(a.max()), 'p95_mm':float(np.percentile(a,95)),
                'rms_mm':float(np.sqrt(np.mean(a*a))), 'over_tolerance':int((a>tolerance_mm).sum())}
    maximum = max(float(forward.max()),float(reverse.max()),max(native))
    def worst_records(points, distances):
        count = min(8, len(distances))
        indices = np.argpartition(distances, len(distances)-count)[-count:]
        indices = indices[np.argsort(distances[indices])[::-1]]
        return [{'sample_index':int(i), 'point_mm':points[i].tolist(),
                 'mesh_distance_mm':float(distances[i])} for i in indices]
    worst_forward = worst_records(source_points, forward)
    worst_reverse = worst_records(model_points, reverse)
    diagnostics = None
    if diagnostic_out is not None:
        # These are the already measured native triangles, not another mesh or
        # a relaxed recheck. Retaining them avoids repeating an expensive native
        # mesher merely to locate a failed sample. Source data remain untouched.
        cache = diagnostic_out/'native_tessellation.npz'
        np.savez_compressed(cache, vertices=mv, faces=mf)
        from OCP.BRepExtrema import BRepExtrema_DistShapeShape
        actual_points, projections = [], []
        for record in worst_reverse:
            vertex = cq.Vertex.makeVertex(*record['point_mm'])
            distance = BRepExtrema_DistShapeShape(boundary.wrapped, vertex.wrapped)
            if not distance.IsDone() or distance.NbSolution() < 1:
                raise ValueError('worst native-mesh sample could not be projected to the true boundary')
            point = distance.PointOnShape1(1)
            xyz = [float(point.X()),float(point.Y()),float(point.Z())]
            actual_points.append(xyz)
            projections.append({**record, 'actual_native_boundary_point_mm':xyz,
                                'projection_distance_mm':float(distance.Value())})
        exact_to_source = mesh_distances(np.asarray(actual_points), v, f)
        for row, measured in zip(projections, exact_to_source):
            row['actual_native_point_to_source_mesh_mm'] = float(measured)
        diagnostics = {'native_tessellation':str(cache.resolve()),
                       'native_tessellation_sha256':sha256(cache),
                       'native_mesh_vertices':len(mv), 'native_mesh_triangles':len(mf),
                       'worst_reverse_actual_boundary_rechecks':projections,
                       'acceptance_unchanged':True,
                       'scope':'Supplemental exact-boundary projection of eight finite mesh samples; does not replace the original maximum or certify a global bound.'}
    result = {'passed': maximum <= tolerance_mm, 'sampled_max_mm': maximum, 'tolerance_mm':tolerance_mm,
              'source_to_model':stats(forward), 'model_to_source':stats(reverse),
              'native_boundary_recheck_worst8_mm':native, 'seed':seed, 'random_samples_each_direction':samples,
              'model_tessellation_requested_deflection_mm':tolerance_mm/10,
              'model_tessellation_relative':False,
              'source_sha256':sha256(source_path), 'step_sha256':sha256(step_path),
              'worst_source_to_model_samples':worst_forward,
              'worst_model_to_source_samples':worst_reverse,
              'diagnostics':diagnostics,
              'geometry':describe(shape),
              'scope':'All supplied mesh and all native boundary faces; finite sampling only, not a certified global maximum; source mesh may itself be partial or noisy.'}
    dump(report_path, result)
    return result

