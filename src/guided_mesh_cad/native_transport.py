"""Generic, scope-preserving native transport and named assembly readback.

Plan discovery is stdlib-only. Native operations import CAD lazily and never
fuse, refit, heal, waive validity, or promote finite evidence to material equality.
"""
from functools import wraps
import gc
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import re
import sys
import uuid

from .native_witness import file_binding, sample_native_boundary, ActualNativeWitnessOracle


def _read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def _write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
    try:
        with temporary.open('x', encoding='utf-8') as stream:
            json.dump(data, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists(): temporary.unlink()
    return str(path.resolve())


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False,
                         separators=(',', ':')).encode()).hexdigest()


def _progress(out, phase, completed, total, current_instance=None):
    """Small read-only status snapshot; deliberately not acceptance evidence."""
    _write(Path(out)/'progress.json', {'schema': 'guided-mesh-cad.progress.v1',
        'phase': phase, 'completed': completed, 'total': total,
        'current_instance': current_instance, 'snapshot_only': True})


def _operation(function):
    @wraps(function)
    def run(*args, **kwargs):
        # Public methods all name out_dir; keep errors on disk as well.
        import inspect
        bound = inspect.signature(function).bind(*args, **kwargs)
        out = Path(bound.arguments['out_dir']).resolve()
        out.mkdir(parents=True, exist_ok=True)
        try:
            report, artifacts = function(*args, **kwargs)
        except Exception as error:
            report = {'schema': 'guided-mesh-cad.native-rejection.v1', 'operation': function.__name__,
                      'ok': False, 'errors': [{'code': 'NATIVE_TRANSPORT_REJECTED',
                                              'type': type(error).__name__, 'message': str(error)}],
                      'strict_material_equality_proven': False}
            artifacts = {}
        report_path = _write(out/'report.json', report)
        artifacts = dict(artifacts, report=report_path)
        return {'ok': bool(report['ok']), 'acceptance': report.get('acceptance', 'rejected'),
                'report': report, 'artifacts': artifacts}
    return run


def _tolerance(value):
    if isinstance(value, bool) or not math.isfinite(value) or not 0 < value <= .01:
        raise ValueError('tolerance_mm must be positive and at most .01; physical gates cannot be relaxed')
    return float(value)


def _parameters(params, allowed):
    unknown = set(params)-set(allowed)
    if unknown:
        raise ValueError('unsupported native parameter(s): '+', '.join(sorted(unknown)))


def identity_matrix():
    return [[float(i == j) for j in range(4)] for i in range(4)]


def _matrix(value):
    if value is None: return identity_matrix()
    if not isinstance(value, (list, tuple)) or len(value) != 4 or any(len(row) != 4 for row in value):
        raise ValueError('placement must be a finite rigid 4x4 matrix')
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
           for row in value for v in row): raise ValueError('placement matrix contains nonfinite/non-numeric values')
    m = [list(map(float, row)) for row in value]
    if max(abs(m[3][j]-float(j == 3)) for j in range(4)) > 1e-12:
        raise ValueError('placement matrix has a non-affine final row')
    orth = max(abs(sum(m[k][i]*m[k][j] for k in range(3))-float(i == j))
               for i in range(3) for j in range(3))
    r = m
    determinant = (r[0][0]*(r[1][1]*r[2][2]-r[1][2]*r[2][1])
                   -r[0][1]*(r[1][0]*r[2][2]-r[1][2]*r[2][0])
                   +r[0][2]*(r[1][0]*r[2][1]-r[1][1]*r[2][0]))
    if orth > 1e-6 or abs(determinant-1.) > 1e-6:
        raise ValueError('placement is not a proper rigid frame; no SVD, scale or reflection repair')
    return m


def _point(matrix, point):
    return [sum(matrix[i][j]*point[j] for j in range(3))+matrix[i][3] for i in range(3)]


def _location(cq, matrix):
    from OCP.gp import gp_Trsf
    matrix = _matrix(matrix)
    transform = gp_Trsf()
    transform.SetValues(*[matrix[i][j] for i in range(3) for j in range(4)])
    actual = [[transform.Value(i+1, j+1) for j in range(4)] for i in range(3)]+[[0., 0., 0., 1.]]
    return cq.Location(transform), actual


def _bbox(shape):
    from OCP.Bnd import Bnd_Box
    from OCP.BRepBndLib import BRepBndLib
    box = Bnd_Box()
    BRepBndLib.AddOptimal_s(shape.wrapped, box, False, True)
    if box.IsVoid(): raise ValueError('native shape has no finite bounding box')
    values = list(box.Get())
    if not all(math.isfinite(v) for v in values): raise ValueError('nonfinite native bounding box')
    return values


def _pose_displacement(shape, left, right):
    from itertools import product
    bounds = _bbox(shape)
    return max(math.dist(_point(left, p), _point(right, p))
               for p in product(*[(bounds[i], bounds[i+3]) for i in range(3)]))


def _placed(cq, shape, matrix, tolerance):
    from itertools import product
    requested = _matrix(matrix)
    location, actual = _location(cq, requested)
    bounds = _bbox(shape)
    corners = list(product(*[(bounds[i], bounds[i+3]) for i in range(3)]))
    displacement = max(math.dist(_point(actual, p), _point(requested, p)) for p in corners)
    if displacement > tolerance+1e-9:
        raise ValueError('actual native transform coercion exceeds point-displacement tolerance')
    return shape.moved(location), location, {'requested_matrix': requested,
        'actual_gp_Trsf_matrix': actual, 'bbox_corner_displacement_upper_mm': displacement,
        'matrix_max_difference': max(abs(actual[i][j]-requested[i][j]) for i in range(4) for j in range(4)),
        'pose_repair_performed': False, 'kernel_coercion_explicitly_measured': True}


def _load(cq, path):
    path = Path(path).resolve()
    if path.suffix.lower() in ('.brep', '.brp'):
        shape = cq.Shape.importBrep(str(path))
    elif path.suffix.lower() in ('.step', '.stp'):
        values = [s for s in cq.importers.importStep(str(path), unit='MM').vals() if isinstance(s, cq.Shape)]
        if not values: raise ValueError('STEP has no shape roots')
        shape = values[0] if len(values) == 1 else cq.Compound.makeCompound(values)
    else: raise ValueError('native transport accepts STEP or BRep only')
    if shape.wrapped.IsNull(): raise ValueError('null native shape')
    return shape


def _stats(shape, tolerance):
    from .cad import shape_check_report, _require_native_integrity
    report = shape_check_report(shape)
    _require_native_integrity(report)
    if not report['valid'] or report['solids'] < 1 or not report['shells'] or report['closed_shells'] != report['shells']:
        raise ValueError('native boundary requires valid closed solids: '+str(report))
    if max(report['native_tolerance_max_mm'].values()) > tolerance+1e-9:
        raise ValueError('native topological tolerance exceeds requested distance tolerance')
    volumes = [float(solid.Volume()) for solid in shape.Solids()]
    if not all(math.isfinite(v) and v > 0 for v in volumes):
        raise ValueError('native solids must have finite positive oriented volumes')
    area = float(shape.Area())
    if not math.isfinite(area) or area <= 0: raise ValueError('native boundary must have finite positive area')
    report.update(bbox_mm=_bbox(shape), volume_mm3=sum(volumes), area_mm2=area,
                  full_BOP_self_intersection='not-checked; inherited scope retained')
    return report


def _leaves(assembly):
    leaves = {}
    for obj, full_name, world_location, _ in assembly:
        name = full_name.split('/')[-1]
        if name in leaves: raise ValueError('duplicate STEP leaf identity: '+name)
        leaves[name] = (obj, world_location, full_name)
    return leaves


def _safe_name(value):
    if not isinstance(value, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,95}', value):
        raise ValueError('IDs must be unique safe nonempty ASCII names (at most 96 characters)')
    if value.upper().split('.')[0] in {'CON', 'PRN', 'AUX', 'NUL', *('COM'+str(i) for i in range(1,10)), *('LPT'+str(i) for i in range(1,10))}:
        raise ValueError('reserved filesystem ID')
    return value


def _plan(path, *, verify_hashes=True):
    path = Path(path).resolve(); plan = _read(path)
    if plan.get('schema') != 'guided-mesh-cad.assembly.v1' or plan.get('units') != 'mm':
        raise ValueError('assembly requires schema guided-mesh-cad.assembly.v1 and explicit units=mm')
    resources, instances = {}, []
    for row in plan.get('resources', []):
        key = _safe_name(row['id'])
        if key in resources: raise ValueError('duplicate resource ID: '+key)
        resource = Path(row['path'])
        if not resource.is_absolute(): resource = path.parent/resource
        resource = resource.resolve()
        if not resource.is_file(): raise ValueError('missing assembly resource: '+str(resource))
        if not isinstance(row.get('sha256'), str) or not re.fullmatch('[a-fA-F0-9]{64}', row['sha256']):
            raise ValueError('every assembly resource requires a SHA256')
        if not row.get('scope'): raise ValueError('every assembly resource requires its inherited acceptance scope')
        binding = file_binding(resource) if verify_hashes else {'path': str(resource), 'sha256': row['sha256'].lower()}
        if verify_hashes and binding['sha256'] != row['sha256'].lower(): raise ValueError('assembly resource SHA256 mismatch: '+key)
        resources[key] = {'id': key, 'source': binding, 'scope': row['scope']}
    used = set()
    for row in plan.get('instances', []):
        key = _safe_name(row['id'])
        if key in used: raise ValueError('duplicate instance ID: '+key)
        used.add(key)
        if row['resource_id'] not in resources: raise ValueError('instance refers to absent resource')
        if 'matrix' not in row: raise ValueError('instance requires an explicit rigid matrix')
        instances.append({'id': key, 'resource_id': row['resource_id'], 'matrix': _matrix(row['matrix'])})
    if not resources or not instances: raise ValueError('assembly resources and instances cannot be empty')
    if not isinstance(plan.get('excluded', []), list) or any(not isinstance(x, dict) or not x for x in plan.get('excluded', [])):
        raise ValueError('excluded must be a list of explicit nonempty records')
    return plan, resources, instances


def referenced_inputs(operation, inputs):
    """Discover transitive files for runner fingerprints without importing CAD."""
    paths = []
    def include(value):
        p = Path(value).resolve()
        if not p.is_file(): raise ValueError('missing referenced input: '+str(p))
        if str(p) not in paths: paths.append(str(p))
    if operation in ('assemble', 'verify-assembly'):
        plan_path = inputs['plan']
        _, resources, _ = _plan(plan_path, verify_hashes=False)
        paths.extend(row['source']['path'] for row in resources.values())
        if operation == 'verify-assembly':
            manifest_path = Path(inputs.get('build_manifest', inputs.get('manifest', ''))).resolve()
            include(manifest_path)
            manifest = _read(manifest_path)
            step = Path(manifest['step']['path'])
            if not step.is_absolute(): step = manifest_path.parent/step
            include(step)
    return list(dict.fromkeys(paths))


def _implementation_binding():
    import cadquery as cq
    import OCP
    package = Path(__file__).parent
    return {'python': sys.version, 'cadquery': cq.__version__, 'OCP': OCP.__version__,
            'implementation': {str(path.relative_to(package)): file_binding(path)['sha256']
                               for path in sorted(package.rglob('*.py'))}}


def _compare_shapes(source, target, out, source_ref, target_ref, leaf, tolerance,
                    source_pose, target_pose, scope=None, progress_out=None):
    from .representation import compare_mass_properties
    if progress_out is not None: _progress(progress_out, 'compare-native.sampling', 0, 2)
    source_stats, target_stats = _stats(source, tolerance), _stats(target, tolerance)
    points, _, source_coverage = sample_native_boundary(source)
    reverse_points, _, target_coverage = sample_native_boundary(target)
    if source_coverage['failed_edge_samples'] or target_coverage['failed_edge_samples']:
        raise ValueError('required native edge sampling failed; no coverage is silently dropped')
    out.mkdir(parents=True, exist_ok=True)
    target_oracle = ActualNativeWitnessOracle(target, {'leaf_name': leaf, 'native_source': target_ref,
        'pose': target_pose, 'role': 'actual_target_boundary'}, out/'target_cloud.json.gz', tolerance)
    if progress_out is not None: _progress(progress_out, 'compare-native.forward', 0, 2)
    forward = target_oracle.batch(points, out/'forward.json.gz', leaf)
    del target_oracle
    source_oracle = ActualNativeWitnessOracle(source, {'leaf_name': leaf, 'native_source': source_ref,
        'pose': source_pose, 'role': 'actual_source_boundary'}, out/'source_cloud.json.gz', tolerance)
    if progress_out is not None: _progress(progress_out, 'compare-native.reverse', 1, 2)
    reverse = source_oracle.batch(reverse_points, out/'reverse.json.gz', leaf)
    if progress_out is not None: _progress(progress_out, 'compare-native.mass-diagnostics', 2, 2)
    del source_oracle
    passed = (not forward['over_tolerance'] and not reverse['over_tolerance']
              and source_stats['solids'] == target_stats['solids'])
    return {'schema': 'guided-mesh-cad.native-comparison.v1', 'ok': passed,
        'acceptance': 'finite-native-boundary-transport' if passed else 'rejected',
        'source': source_ref, 'target': target_ref, 'target_leaf': leaf,
        'source_stats': source_stats, 'target_stats': target_stats,
        'source_pose': source_pose, 'target_pose': target_pose,
        'forward': forward, 'reverse': reverse, 'source_sampling_coverage': source_coverage,
        'target_sampling_coverage': target_coverage, 'tolerance_mm': tolerance,
        'mass_diagnostics': compare_mass_properties(source, target), 'inherited_scope': scope,
        'strict_material_equality_proven': False, 'continuous_surface_equivalence_proven': False,
        'full_assembly_clearance_checked': False, 'continuous_motion_checked': False,
        'native_warning_policy': 'positive_union unchanged; any native boolean warning remains unresolved'}


@_operation
def compare_native(source, target, out_dir, **params):
    _parameters(params, {'tolerance_mm', 'source_matrix', 'target_matrix', 'target_instance', 'scope'})
    import cadquery as cq
    tolerance = _tolerance(params.get('tolerance_mm', .01)); out = Path(out_dir).resolve()
    source_ref, target_ref = file_binding(source), file_binding(target)
    raw_left = _load(cq, source)
    left, _, left_pose = _placed(cq, raw_left, params.get('source_matrix'), tolerance)
    leaf = params.get('target_instance')
    if leaf is not None:
        _safe_name(leaf)
        leaves = _leaves(cq.Assembly.importStep(str(target), unit='MM'))
        if leaf not in leaves: raise ValueError('requested exact target leaf is absent: '+leaf)
        obj, location, full_name = leaves[leaf]
        raw_right = obj.moved(location)
    else:
        leaf, raw_right = 'target', _load(cq, target)
    right, _, right_pose = _placed(cq, raw_right, params.get('target_matrix'), tolerance)
    report = _compare_shapes(left, right, out/'evidence', source_ref, target_ref, leaf,
                             tolerance, left_pose, right_pose, params.get('scope'), progress_out=out)
    # Pose coercion and exchange distance share the requested physical budget.
    # Reuse is allowed only for exact equality to the *actually queried* lists.
    for label, raw, placed, opposite, reference, opposite_pose, existing in (
            ('requested_source_to_target', raw_left, left, right, target_ref, right_pose, 'forward'),
            ('requested_target_to_source', raw_right, right, left, source_ref, left_pose, 'reverse')):
        requested_matrix = left_pose['requested_matrix'] if existing == 'forward' else right_pose['requested_matrix']
        requested_points = [_point(requested_matrix, p) for p in sample_native_boundary(raw)[0]]
        queried_points = sample_native_boundary(placed)[0]
        exact = requested_points == queried_points
        if exact:
            result = report[existing]
        else:
            oracle = ActualNativeWitnessOracle(opposite, {'leaf_name': leaf, 'native_source': reference,
                'pose': opposite_pose, 'role': label}, out/(label+'_cloud.json.gz'), tolerance)
            result = oracle.batch(requested_points, out/(label+'.json.gz'), leaf)
            del oracle
        report[label] = result
        report[label+'_exact_query_reuse'] = exact
        report['ok'] = report['ok'] and not result['over_tolerance']
    if not report['ok']: report['acceptance'] = 'rejected'
    if source_ref != file_binding(source) or target_ref != file_binding(target):
        raise ValueError('native input changed during comparison')
    return report, {}


@_operation
def assemble(plan, out_dir, **params):
    _parameters(params, {'tolerance_mm'})
    plan_path = Path(plan).resolve(); initial = file_binding(plan_path)
    document, resources, instances = _plan(plan_path)
    import cadquery as cq
    tolerance = _tolerance(params.get('tolerance_mm', .01)); out = Path(out_dir).resolve()
    assembly, records = cq.Assembly(name='assembly'), []
    _progress(out, 'assemble', 0, len(instances))
    # Only the assembly retains placed instances; no cross-resource witness pool.
    for index, instance in enumerate(instances):
        _progress(out, 'assemble', index, len(instances), instance['id'])
        resource = resources[instance['resource_id']]
        shape = _load(cq, resource['source']['path']); stats = _stats(shape, tolerance)
        world, location, pose = _placed(cq, shape, instance['matrix'], tolerance)
        assembly.add(shape, name=instance['id'], loc=location)
        records.append(dict(instance, source=resource['source'], scope=resource['scope'],
                            source_stats=stats, transform=pose))
        _progress(out, 'assemble', index+1, len(instances), instance['id'])
        del world, shape
    step = out/'assembly.step'
    _progress(out, 'assemble.export', len(instances), len(instances))
    assembly.export(str(step), 'STEP', mode='default', unit='MM')
    if initial != file_binding(plan_path): raise ValueError('assembly plan changed during export')
    _, current_resources, _ = _plan(plan_path)
    if current_resources != resources: raise ValueError('assembly source changed during export')
    manifest = {'schema': 'guided-mesh-cad.assembly-build.v1', 'plan': initial, 'step': file_binding(step),
        'units': 'mm', 'resources': resources, 'records': records, 'excluded': document.get('excluded', []),
        'tolerance_mm': tolerance, 'implementation': _implementation_binding(),
        'assembly_mode': 'separate named STEP leaves; no fuse', 'fresh_STEP_readback': False,
        'strict_material_equality_proven': False, 'independent_source_acceptance': 'inherited per resource'}
    manifest_path = _write(out/'build_manifest.json', manifest)
    return {'schema': 'guided-mesh-cad.assembly-export.v1', 'ok': True,
        'acceptance': 'named-STEP-export-only', 'instances': len(instances), 'resources': len(resources),
        'excluded': document.get('excluded', []), 'fresh_STEP_readback': False,
        'whole_assembly_accepted': False, 'manifest': file_binding(manifest_path)}, {
            'step': str(step), 'manifest': manifest_path}


def _runner_context(out):
    """Recognize the fixed worker layout without caller-selected owner roots."""
    request = out.parent/'request.json'
    if out.name != 'output' or not re.fullmatch('attempt-[a-f0-9]{32}', out.parent.name) or not request.is_file():
        return None
    document = _read(request)
    if document.get('operation') != 'verify-assembly': return None
    return {'root': out.parent.parent/'verification-checkpoints',
            'binding': document.get('checkpoint_binding')}


def _checkpoint_area(out, explicit, binding):
    checkpoint = Path(explicit).resolve() if explicit else out/'instance_checkpoints'
    runner = _runner_context(out)
    runner_owned = runner and checkpoint.is_relative_to(runner['root'].resolve()) and checkpoint != runner['root'].resolve()
    if not runner_owned and (checkpoint == out.parent or not checkpoint.is_relative_to(out.parent)):
        raise ValueError('checkpoint_dir must be an owned child within output parent')
    checkpoint.mkdir(parents=True, exist_ok=True)
    owner = checkpoint/'owner.json'
    expected = {'schema': 'guided-mesh-cad.instance-checkpoint-owner.v1', 'binding': binding,
                'checkpoint_root': str(checkpoint)}
    if owner.exists():
        if _read(owner) != expected: raise ValueError('checkpoint ownership/binding mismatch; old receipts cannot be re-signed')
    else:
        if any(checkpoint.iterdir()): raise ValueError('unowned checkpoint directory is not empty')
        _write(owner, expected)
    return checkpoint


def _checkpoint_load(checkpoint, name, binding):
    index = checkpoint/(name+'.checkpoint.json')
    if not index.is_file(): return None
    try:
        entry = _read(index)
        if entry['binding'] != binding: return None
        for ref in [entry['record'], *entry['evidence']]:
            path = Path(ref['path']).resolve()
            if not path.is_relative_to(checkpoint) or ref != file_binding(path): return None
        report = _read(entry['record']['path'])
        if not report['ok'] or report['target_leaf'] != name: return None
        return report, entry['record']
    except (OSError, ValueError, KeyError, TypeError): return None


@_operation
def verify_assembly(plan, build_manifest, out_dir, **params):
    _parameters(params, {'tolerance_mm', 'checkpoint_dir'})
    plan_path, manifest_path = Path(plan).resolve(), Path(build_manifest).resolve()
    initial_plan, initial_manifest = file_binding(plan_path), file_binding(manifest_path)
    document, resources, instances = _plan(plan_path)
    manifest = _read(manifest_path)
    if manifest.get('schema') != 'guided-mesh-cad.assembly-build.v1' or manifest.get('units') != 'mm':
        raise ValueError('invalid assembly build manifest or units')
    if manifest['plan'] != initial_plan or manifest['resources'] != resources:
        raise ValueError('manifest plan/source bindings differ')
    if manifest.get('excluded', []) != document.get('excluded', []): raise ValueError('excluded scope changed')
    step_ref = manifest['step']
    if file_binding(step_ref['path']) != step_ref: raise ValueError('built STEP hash mismatch')
    tolerance = _tolerance(params.get('tolerance_mm', manifest.get('tolerance_mm', .01)))
    if tolerance > manifest['tolerance_mm']: raise ValueError('verification cannot relax build tolerance')
    import cadquery as cq
    implementation = _implementation_binding()
    if implementation != manifest['implementation']:
        raise ValueError('build implementation/runtime changed; rebuild required')
    records = {r['id']: r for r in manifest['records']}
    if len(records) != len(manifest['records']) or set(records) != {r['id'] for r in instances}:
        raise ValueError('build manifest instance inventory differs')
    for instance in instances:
        rec, resource = records[instance['id']], resources[instance['resource_id']]
        if any(rec.get(k) != instance[k] for k in ('id', 'resource_id', 'matrix')) or rec['source'] != resource['source'] or rec['scope'] != resource['scope']:
            raise ValueError('build instance identity/pose/scope differs')
    out = Path(out_dir).resolve()
    binding = {'plan': initial_plan, 'manifest': initial_manifest, 'step': step_ref,
               'resources': resources, 'tolerance_mm': tolerance, 'implementation': implementation,
               'params': {k: v for k, v in params.items() if k != 'checkpoint_dir'}}
    runner = _runner_context(out)
    if runner: binding['workflow_checkpoint_binding'] = runner['binding']
    checkpoint = _checkpoint_area(out, params.get('checkpoint_dir'), binding)
    _progress(out, 'verify-assembly.import', 0, len(instances))
    # This full STEP import happens even if every per-instance receipt is reusable.
    assembly = cq.Assembly.importStep(step_ref['path'], unit='MM'); leaves = _leaves(assembly)
    if set(leaves) != {r['id'] for r in instances}:
        raise ValueError('fresh full STEP leaf identities differ from complete plan')
    accepted, reused, freshly_checked = [], 0, 0
    for index, instance in enumerate(instances):
        name = instance['id']; resource = resources[instance['resource_id']]
        _progress(out, 'verify-assembly', index, len(instances), name)
        cached = _checkpoint_load(checkpoint, name, binding)
        if cached:
            accepted.append({'id': name, 'ok': True, 'scope': resource['scope'],
                             'resumed': True, 'record': cached[1]}); reused += 1
            _progress(out, 'verify-assembly', index+1, len(instances), name)
            continue
        attempt = checkpoint/name/uuid.uuid4().hex; attempt.mkdir(parents=True)
        source = _load(cq, resource['source']['path'])
        world, _, pose = _placed(cq, source, instance['matrix'], tolerance)
        if pose != records[name]['transform']: raise ValueError('actual native placement differs from build')
        obj, world_location, full_name = leaves[name]
        target = obj.moved(world_location)
        transform = world_location.wrapped.Transformation()
        imported_pose = [[transform.Value(i+1,j+1) for j in range(4)] for i in range(3)]+[[0.,0.,0.,1.]]
        pose_residual = _pose_displacement(source, pose['actual_gp_Trsf_matrix'], imported_pose)
        if pose_residual > tolerance+1e-9:
            raise ValueError('fresh STEP imported leaf pose differs beyond point-displacement tolerance: '+name)
        report = _compare_shapes(world, target, attempt/'evidence', resource['source'], step_ref,
                                 name, tolerance, pose, imported_pose, resource['scope'])
        report['imported_full_name'] = full_name
        report['resource_id'] = instance['resource_id']
        report['imported_pose_point_displacement_upper_mm'] = pose_residual
        report['requested_original_pose_to_readback'] = report['forward']
        # Kernel coercion and STEP exchange must share the same .01 budget.
        raw_points, _, _ = sample_native_boundary(source)
        actual_points = sample_native_boundary(world)[0]
        requested_points = [_point(instance['matrix'], p) for p in raw_points]
        exact = actual_points == requested_points
        if not exact:
            oracle = ActualNativeWitnessOracle(target, {'leaf_name': name, 'native_source': step_ref,
                'pose': imported_pose, 'role': 'requested_original_pose_target'}, attempt/'requested_cloud.json.gz', tolerance)
            requested = oracle.batch(requested_points, attempt/'requested_forward.json.gz', name)
            report['requested_original_pose_to_readback'] = requested
            report['ok'] = report['ok'] and not requested['over_tolerance']
            del oracle
        report['requested_forward_exact_query_reuse'] = exact
        record_path = _write(attempt/'instance_report.json', report)
        record_ref = file_binding(record_path)
        accepted.append({'id': name, 'ok': report['ok'], 'scope': resource['scope'],
                         'resumed': False, 'record': record_ref}); freshly_checked += 1
        if report['ok']:
            evidence = [file_binding(p) for p in sorted(attempt.rglob('*')) if p.is_file() and p != Path(record_path)]
            _write(checkpoint/(name+'.checkpoint.json'), {'binding': binding, 'record': record_ref, 'evidence': evidence})
        _progress(out, 'verify-assembly' if report['ok'] else 'verify-assembly.instance-rejected',
                  index+1, len(instances), name)
        del source, world, target, report
        gc.collect()
    if initial_plan != file_binding(plan_path) or initial_manifest != file_binding(manifest_path) or step_ref != file_binding(step_ref['path']):
        raise ValueError('verification inputs changed during native checks')
    _, current_resources, _ = _plan(plan_path)
    if resources != current_resources: raise ValueError('assembly source changed during verification')
    # A successful workflow receipt depends on the per-instance native evidence,
    # not just the small report that points at it. Give the runner an explicit,
    # bounded-schema transitive output manifest to hash on acceptance and reuse.
    refs = {str(checkpoint/'owner.json'): file_binding(checkpoint/'owner.json')}
    for row in accepted:
        record_path = Path(row['record']['path'])
        if row['record'] != file_binding(record_path): raise ValueError('instance report changed during verification')
        if row['ok']:
            cached = _checkpoint_load(checkpoint, row['id'], binding)
            if not cached or cached[1] != row['record']:
                raise ValueError('instance checkpoint/evidence changed during verification')
            index = checkpoint/(row['id']+'.checkpoint.json')
            refs[str(index)] = file_binding(index)
        for path in sorted(record_path.parent.rglob('*')):
            if path.is_file(): refs[str(path)] = file_binding(path)
    evidence_manifest = _write(out/'evidence_manifest.json', {
        'schema': 'guided-mesh-cad.evidence-manifest.v1', 'refs': list(refs.values()),
        'checkpoint_root': str(checkpoint),
        'checkpoint_binding': runner['binding'] if runner else None})
    passed = all(r['ok'] for r in accepted)
    return {'schema': 'guided-mesh-cad.assembly-verification.v1', 'ok': passed,
        'acceptance': 'finite-native-named-STEP-transport' if passed else 'rejected',
        'plan': initial_plan, 'manifest': initial_manifest, 'step': step_ref, 'units': 'mm',
        'fresh_complete_STEP_import': True, 'identity_names_pass': True, 'instances': len(instances),
        'reused_instances': reused, 'freshly_checked_instances': freshly_checked, 'records': accepted,
        'excluded': document.get('excluded', []), 'tolerance_mm': tolerance,
        'per_instance_checkpoint_dir': str(checkpoint), 'strict_material_equality_proven': False,
        'continuous_surface_equivalence_proven': False, 'whole_assembly_accepted': False,
        'assembly_clearance_checked': False, 'continuous_motion_checked': False,
        'inherited_scopes_preserved': True,
        'evidence_manifest': file_binding(evidence_manifest)}, {'evidence_manifest': evidence_manifest}
