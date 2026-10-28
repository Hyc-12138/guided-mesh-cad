"""Private isolated operation dispatcher. No shell or user-selected callables."""
import json
from pathlib import Path
import sys
import traceback


OPERATIONS = frozenset(('inspect-input', 'reconstruct', 'validate', 'diagnose-native',
                        'compare-native', 'assemble', 'verify-assembly', 'audit-domains'))
INPUT_KEYS = {'reconstruct': {'plan'}, 'validate': {'mesh', 'step'}, 'diagnose-native': {'native'},
              'compare-native': {'source', 'target'}, 'assemble': {'plan'},
              'verify-assembly': {'plan', 'build_manifest'}, 'audit-domains': {'source', 'domains'}}


def _required(inputs, *names):
    for name in names:
        if name not in inputs:
            raise ValueError('missing input: ' + name)
    return [inputs[name] for name in names]


def dispatch(operation, inputs, params, out_dir, scope, checkpoint_root=None):
    """Return the public envelope; CAD imports occur only in selected branches."""
    out = Path(out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    if operation in INPUT_KEYS and set(inputs) != INPUT_KEYS[operation]:
        raise ValueError('required input keys for ' + operation + ': ' + ', '.join(sorted(INPUT_KEYS[operation])))
    if operation in ('reconstruct', 'diagnose-native', 'audit-domains') and params:
        raise ValueError(operation + ' accepts no parameters')
    if operation == 'inspect-input':
        from .workflow import file_hash
        if set(params) - {'expected_sha256'}:
            raise ValueError('inspect-input accepts only expected_sha256')
        expected = params.get('expected_sha256', {})
        if not isinstance(expected, dict) or set(expected) - set(inputs):
            raise ValueError('expected_sha256 must map existing input names to SHA256 values')
        records = {key: {'path': str(Path(value).resolve()),
                         'bytes': Path(value).stat().st_size, 'sha256': file_hash(value)}
                   for key, value in sorted(inputs.items())}
        mismatches = [key for key, digest in expected.items() if records[key]['sha256'] != digest]
        return {'ok': not mismatches,
                'acceptance': 'input_evidence_only' if not mismatches else 'input_hash_rejected',
                'report': {'scope': scope, 'inputs': records, 'hash_mismatches': mismatches,
                           'geometry_acceptance': False,
                           'limitations': ['File identity evidence only; geometry is not evaluated.']},
                'artifacts': {}}
    if operation == 'reconstruct':
        from .pipeline import reconstruct
        report = reconstruct(_required(inputs, 'plan')[0], out / 'reconstruction')
        artifacts = {key: str(out / 'reconstruction' / name) for key, name in
                     [('step', 'model.step'), ('brep', 'model.brep'), ('manifest', 'manifest.json'),
                      ('verification', 'verification.json')]}
        return {'ok': report.get('status') == 'sampled_geometry_pass',
                'acceptance': report.get('status', 'geometry_rejected'),
                'report': report, 'artifacts': artifacts}
    if operation == 'validate':
        if set(params) - {'tolerance_mm', 'samples', 'seed'}:
            raise ValueError('unknown validate parameter')
        from .pipeline import validate
        mesh, step = _required(inputs, 'mesh', 'step')
        report = validate(mesh, step, out / 'verification.json', **params)
        return {'ok': report['passed'], 'acceptance': 'finite_sample_geometry_pass' if report['passed'] else 'finite_sample_geometry_rejected',
                'report': report, 'artifacts': {'verification': str(out / 'verification.json')}}
    if operation == 'diagnose-native':
        from .cad import read_native_unchecked, shape_check_report, require_solid, PatchTopologyError
        shape = read_native_unchecked(_required(inputs, 'native')[0])
        report = shape_check_report(shape)
        try:
            require_solid(shape)
            ok = True
        except PatchTopologyError as exc:
            report['formal_solid_gate_error'] = str(exc)
            ok = False
        report['formal_solid_gate_passed'] = ok
        return {'ok': ok, 'acceptance': 'native_topology_only' if ok else 'native_topology_rejected',
                'report': report, 'artifacts': {}}
    if operation in ('compare-native', 'assemble', 'verify-assembly'):
        from . import native_transport
        if operation == 'compare-native':
            return native_transport.compare_native(*_required(inputs, 'source', 'target'), out, **params)
        if operation == 'assemble':
            return native_transport.assemble(_required(inputs, 'plan')[0], out, **params)
        root = Path(checkpoint_root).resolve() if checkpoint_root else out.parent / 'verification-checkpoints'
        checkpoint = Path(params.get('checkpoint_dir', root))
        checkpoint = (root / checkpoint).resolve() if not checkpoint.is_absolute() else checkpoint.resolve()
        if not checkpoint.is_relative_to(root):
            raise ValueError('checkpoint_dir must remain within the bound task checkpoint namespace')
        return native_transport.verify_assembly(*_required(inputs, 'plan', 'build_manifest'), out,
                                                **{**params, 'checkpoint_dir': str(checkpoint)})
    if operation == 'audit-domains':
        from .source_domains import audit_domains
        return audit_domains(*_required(inputs, 'source', 'domains'), out, **params)
    raise ValueError('unsupported operation: ' + str(operation))


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 1:
        raise ValueError('private worker expects a request path')
    request_path = Path(argv[0]).resolve()
    request = json.loads(request_path.read_text(encoding='utf-8'))
    # Parent attaches the process to its hard-limit job before releasing this gate.
    if sys.stdin.buffer.readline() != b'GO\n':
        return 1
    from .resources import ResourceLimits, apply_child_limit
    from .workflow import _atomic_json, file_hash
    out = request_path.parent / 'output'
    try:
        apply_child_limit(ResourceLimits.from_mapping(request['resources']))
        out.mkdir(parents=True, exist_ok=True)
        _atomic_json(out / 'progress.json', {'schema': 'guided-mesh-cad.progress.v1',
                     'phase': 'dispatch_started', 'completed': 0, 'total': 1,
                     'current_instance': None, 'snapshot_only': True})
        result = dispatch(request['operation'], request['inputs'], request['params'], out, request['scope'],
                          request.get('checkpoint_root'))
        if not isinstance(result, dict) or type(result.get('ok')) is not bool or not isinstance(result.get('report'), dict) or not isinstance(result.get('acceptance'), str) or not isinstance(result.get('artifacts'), dict):
            raise ValueError('operation returned an invalid result envelope')
        for key, value in result['artifacts'].items():
            path = Path(value).resolve()
            if not isinstance(key, str) or not path.is_relative_to(out.resolve()) or not path.is_file():
                raise ValueError('operation artifact is missing or outside the attempt output')
            result['artifacts'][key] = str(path)
        result['report']['workflow_scope'] = request['scope']
        report_path = out / 'report.json'
        _atomic_json(report_path, result['report'])
        result['artifacts']['report'] = str(report_path.resolve())
        result['error_code'] = None if result['ok'] else 'operation_rejected'
        result['exit_code'] = 0 if result['ok'] else 2
    except Exception as exc:
        out.mkdir(parents=True, exist_ok=True)
        traceback.print_exc()
        report = {'error_code': 'memory_limit_exceeded' if isinstance(exc, MemoryError) else ('invalid_input' if isinstance(exc, (ValueError, FileNotFoundError, KeyError, TypeError)) else 'operation_error'),
                  'type': type(exc).__name__, 'error': str(exc), 'scope': request.get('scope')}
        report_path = out / 'report.json'
        _atomic_json(report_path, report)
        result = {'ok': False, 'acceptance': 'execution_error', 'report': report,
                  'error_code': report['error_code'], 'exit_code': 1,
                  'artifacts': {'report': str(report_path.resolve())},
                  'artifact_hashes': {'report': file_hash(report_path)}}
    # Finish progress before freezing the attempt's complete output manifest.
    # Dispatch completion alone is never a geometry acceptance decision.
    _atomic_json(out / 'progress.json', {'schema': 'guided-mesh-cad.progress.v1',
                 'phase': 'dispatch_finished', 'completed': 1, 'total': 1,
                 'current_instance': None, 'snapshot_only': True})
    # Bind complete successful and rejected local evidence, including detailed
    # files referenced only inside a native report rather than artifact keys.
    output_manifest = out / 'output_files.json'
    refs = [{'path': str(path.resolve()), 'sha256': file_hash(path), 'bytes': path.stat().st_size}
            for path in sorted(out.rglob('*')) if path.is_file() and path != output_manifest]
    _atomic_json(output_manifest, {'schema': 'guided-mesh-cad.attempt-output-manifest.v1',
                                   'root': str(out.resolve()), 'refs': refs})
    result['artifacts']['output_manifest'] = str(output_manifest.resolve())
    result['artifact_hashes'] = {key: file_hash(value) for key, value in result['artifacts'].items()}
    # Never dump native reports/logs into parent stdout.
    _atomic_json(request_path.parent / 'result.json', result)
    return result['exit_code']


if __name__ == '__main__':
    raise SystemExit(main())
