"""Hash-bound serial DAG runner with immutable attempts and bounded read APIs.

Completion describes operation execution. Each operation retains its own evidence
scope; an inspection or assembly export never implies whole-model certification.
"""
from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import uuid

from ._worker import OPERATIONS, INPUT_KEYS
from .resources import ResourceError, ResourceLimits, ProcessGuard, check_reserves, wait_process

SCHEMA = 'guided-mesh-cad.workflow.v1'
STATE_SCHEMA = 'guided-mesh-cad.checkpoint.v1'
_SAFE = re.compile(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,79}\Z')


class WorkflowError(ValueError):
    def __init__(self, code, message):
        super().__init__(message)
        self.code = code


def _now():
    return datetime.now(timezone.utc).isoformat()


def _json_bytes(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(',', ':')).encode('utf-8')


def _digest(value):
    return hashlib.sha256(_json_bytes(value)).hexdigest()


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path, value):
    """A partial JSON write is never visible as a checkpoint/result."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name('.' + path.name + '.' + uuid.uuid4().hex + '.tmp')
    try:
        with temp.open('xb') as handle:
            handle.write(_json_bytes(value))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, path)
        if os.name != 'nt':
            descriptor = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
    finally:
        if temp.exists():
            temp.unlink()


def _read_json(path):
    def invalid(value):
        raise ValueError('non-finite JSON number: ' + value)
    return json.loads(Path(path).read_text(encoding='utf-8-sig'), parse_constant=invalid)


@contextmanager
def _run_lock(run_dir):
    """OS ownership releases after a crash; PID text is only diagnostic."""
    path = Path(run_dir) / '.run.lock'
    handle = path.open('a+b')
    locked = False
    try:
        handle.seek(0, os.SEEK_END)
        if handle.tell() == 0:
            handle.write(b'\0')
            handle.flush()
        handle.seek(0)
        try:
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            locked = True
        except OSError as exc:
            raise WorkflowError('run_locked', 'another process owns this run') from exc
        handle.seek(1)
        handle.write(_json_bytes({'pid': os.getpid(), 'acquired': _now()}))
        handle.truncate()
        handle.flush()
        yield
    finally:
        if locked:
            handle.seek(0)
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()


def runtime_binding():
    """Versions and actual Python implementation bytes, without importing CAD."""
    package = Path(__file__).parent
    code = {str(path.relative_to(package)): file_hash(path) for path in sorted(package.rglob('*.py'))}
    dependencies = {}
    for name in ('numpy', 'scipy', 'cadquery', 'cadquery-ocp', 'vtk', 'numba'):
        try:
            dependencies[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            dependencies[name] = None
    return {'python': sys.version, 'implementation': platform.python_implementation(),
            'executable': str(Path(sys.executable).resolve()), 'platform': platform.platform(),
            'dependencies': dependencies, 'code_sha256': _digest(code)}


def _safe_id(value):
    if not isinstance(value, str) or not _SAFE.fullmatch(value) or value.endswith('.') or value.split('.')[0].upper() in {'CON', 'PRN', 'AUX', 'NUL', *('COM' + str(i) for i in range(1, 10)), *('LPT' + str(i) for i in range(1, 10))}:
        raise ValueError('task ids must be unique safe names of at most 80 characters')
    return value


def load_plan(path):
    plan = _read_json(path)
    if not isinstance(plan, dict) or plan.get('schema') != SCHEMA:
        raise ValueError('workflow schema must be ' + SCHEMA)
    if set(plan) - {'schema', 'tasks', 'resources'}:
        raise ValueError('unknown workflow plan keys')
    resources = ResourceLimits.from_mapping(plan.get('resources')).to_dict()
    tasks = plan.get('tasks')
    if not isinstance(tasks, list) or not tasks or len(tasks) > 10000:
        raise ValueError('tasks must contain between 1 and 10000 tasks')
    ids, normalized = set(), []
    for task in tasks:
        if not isinstance(task, dict) or set(task) - {'id', 'operation', 'inputs', 'params', 'depends_on', 'scope', 'resources'}:
            raise ValueError('invalid task object or unknown task keys')
        identity = _safe_id(task.get('id'))
        if identity in ids:
            raise ValueError('duplicate task id: ' + identity)
        ids.add(identity)
        if task.get('operation') not in OPERATIONS:
            raise ValueError('unsupported operation: ' + str(task.get('operation')))
        if not isinstance(task.get('scope'), str) or not task['scope'].strip():
            raise ValueError('every task requires an explicit nonempty scope')
        inputs, params, deps = task.get('inputs', {}), task.get('params', {}), task.get('depends_on', [])
        if not isinstance(inputs, dict) or not isinstance(params, dict) or not isinstance(deps, list) or any(not isinstance(dep, str) for dep in deps):
            raise ValueError('inputs/params must be objects and depends_on a list of ids')
        if task['operation'] in INPUT_KEYS and set(inputs) != INPUT_KEYS[task['operation']]:
            raise ValueError('incorrect input keys for ' + task['operation'])
        allowed_params = {'inspect-input': {'expected_sha256'}, 'validate': {'tolerance_mm', 'samples', 'seed'},
                          'reconstruct': set(), 'diagnose-native': set(), 'audit-domains': set()}
        if task['operation'] in allowed_params and set(params) - allowed_params[task['operation']]:
            raise ValueError('unknown parameters for ' + task['operation'])
        deps = list(dict.fromkeys(deps))
        for name, value in inputs.items():
            if not isinstance(name, str) or not name:
                raise ValueError('input names must be nonempty strings')
            if isinstance(value, str):
                if not value:
                    raise ValueError('input paths must be nonempty')
            elif isinstance(value, dict) and set(value) == {'task', 'artifact'} and all(isinstance(v, str) and v for v in value.values()):
                if value['task'] not in deps:
                    deps.append(value['task'])
            else:
                raise ValueError('input must be a path or a task/artifact reference')
        merged_resources = dict(resources)
        if not isinstance(task.get('resources', {}), dict):
            raise ValueError('task resources must be an object')
        merged_resources.update(task.get('resources', {}))
        normalized.append({**task, 'inputs': inputs, 'params': params, 'depends_on': deps,
                           'resources': ResourceLimits.from_mapping(merged_resources).to_dict()})
    order, pending = [], list(normalized)
    for task in normalized:
        if any(dep not in ids or dep == task['id'] for dep in task['depends_on']):
            raise ValueError('missing or self dependency for ' + task['id'])
    ready = set()
    while pending:
        batch = [task for task in pending if set(task['depends_on']) <= ready]
        if not batch:
            raise ValueError('workflow dependency cycle')
        for task in batch:
            order.append(task)
            ready.add(task['id'])
            pending.remove(task)
    return {'schema': SCHEMA, 'tasks': order, 'resources': resources}


def _resolved_inputs(task, base, state):
    result = {}
    for name, value in task['inputs'].items():
        if isinstance(value, str):
            path = (base / value).resolve()
        else:
            dependency = state['tasks'][value['task']]
            try:
                path = Path(dependency['artifacts'][value['artifact']]).resolve()
            except KeyError as exc:
                raise ValueError('dependency artifact is unavailable: ' + value['artifact']) from exc
        if not path.is_file():
            raise FileNotFoundError('input file does not exist: ' + str(path))
        result[name] = str(path)
    return result


def referenced_inputs(operation, inputs):
    """Transitive source bindings; native_transport's hook must remain stdlib."""
    result = []
    if operation == 'reconstruct':
        path = Path(inputs['plan'])
        plan = _read_json(path)
        for name in ('mesh', 'construction_mesh'):
            if name in plan:
                result.append(str((path.parent / plan[name]).resolve()))
    if operation in ('compare-native', 'assemble', 'verify-assembly'):
        from .native_transport import referenced_inputs as native_refs
        result.extend(native_refs(operation, inputs))
    return sorted(set(str(Path(path).resolve()) for path in result))


def _fingerprint(task, inputs, state, binding):
    paths = sorted(set(inputs.values()) | set(referenced_inputs(task['operation'], inputs)))
    sources = {path: file_hash(path) for path in paths}
    dependencies = {name: {'fingerprint': state['tasks'][name]['fingerprint'],
                            'artifact_hashes': state['tasks'][name].get('artifact_hashes', {})}
                    for name in task['depends_on']}
    return _digest({'task': task, 'inputs': sources, 'dependencies': dependencies, 'runtime': binding}), sources


def _artifacts_valid(record, run_dir):
    try:
        artifacts, hashes = record['artifacts'], record['artifact_hashes']
        if not artifacts or set(artifacts) != set(hashes):
            return False
        for key, name in artifacts.items():
            path = Path(name).resolve()
            if not path.is_relative_to(run_dir) or not path.is_file() or file_hash(path) != hashes[key]:
                return False
        if 'output_manifest' in artifacts:
            manifest_path = Path(artifacts['output_manifest']).resolve()
            manifest = _read_json(manifest_path)
            root = Path(manifest['root']).resolve()
            if manifest['schema'] != 'guided-mesh-cad.attempt-output-manifest.v1' or root != manifest_path.parent:
                return False
            if not _evidence_refs_valid(manifest['refs'], root):
                return False
            expected = {str(path.resolve()) for path in root.rglob('*') if path.is_file() and path.resolve() != manifest_path}
            if expected != {ref['path'] for ref in manifest['refs']}:
                return False
        if 'evidence_manifest' in artifacts:
            manifest = _read_json(artifacts['evidence_manifest'])
            if manifest['schema'] != 'guided-mesh-cad.evidence-manifest.v1':
                return False
            attempt = Path(artifacts['report']).resolve().parent.parent
            if not re.fullmatch(r'attempt-[0-9a-f]{32}', attempt.name):
                return False
            request = _read_json(attempt / 'request.json')
            if request.get('operation') != 'verify-assembly' or not request.get('checkpoint_binding'):
                return False
            root = Path(manifest['checkpoint_root']).resolve()
            allowed = Path(request['checkpoint_root']).resolve()
            task_root = attempt.parent / 'verification-checkpoints'
            if not allowed.is_relative_to(task_root) or allowed == task_root or not root.is_relative_to(allowed):
                return False
            if manifest['checkpoint_binding'] != request['checkpoint_binding'] or not _evidence_refs_valid(manifest['refs'], root):
                return False
        return True
    except (KeyError, OSError, TypeError, ValueError):
        return False


def _evidence_refs_valid(refs, root):
    if not isinstance(refs, list) or not refs:
        return False
    seen = set()
    for ref in refs:
        path = Path(ref['path']).resolve()
        if str(path) in seen or not path.is_relative_to(root) or not path.is_file():
            return False
        seen.add(str(path))
        if path.stat().st_size != ref['bytes'] or file_hash(path) != ref['sha256']:
            return False
    return True


def _error(code, message, exit_code=1):
    return {'error_code': code, 'error': str(message)[:600], 'exit_code': exit_code}


def _save(run_dir, state):
    state['updated'] = _now()
    _atomic_json(run_dir / 'checkpoint.json', state)


def _execute_task(task, inputs, attempt, limits, fingerprint=None):
    """Spawn only the fixed internal module; shell execution is never enabled."""
    request = {'operation': task['operation'], 'inputs': inputs, 'params': task['params'],
               'scope': task['scope'], 'resources': limits.to_dict()}
    if task['operation'] == 'verify-assembly':
        request['checkpoint_binding'] = {'task_fingerprint': fingerprint}
        request['checkpoint_root'] = str(attempt.parent / 'verification-checkpoints' / fingerprint)
    _atomic_json(attempt / 'request.json', request)
    try:
        check_reserves(limits)
    except ResourceError as exc:
        return _error(exc.code, exc, 2)
    env = os.environ.copy()
    # Include the exact package location in both installed and source-tree runs.
    env['PYTHONPATH'] = str(Path(__file__).resolve().parent.parent) + (os.pathsep + env['PYTHONPATH'] if env.get('PYTHONPATH') else '')
    env['PYTHONUNBUFFERED'] = '1'
    guard, process = None, None
    try:
        with (attempt / 'worker.log').open('xb') as log:
            process = subprocess.Popen([sys.executable, '-m', 'guided_mesh_cad._worker', str(attempt / 'request.json')],
                                       stdin=subprocess.PIPE, stdout=log, stderr=subprocess.STDOUT,
                                       cwd=str(attempt), env=env, shell=False,
                                       creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0,
                                       start_new_session=os.name != 'nt')
            guard = ProcessGuard(process, limits)
            process.stdin.write(b'GO\n')
            process.stdin.flush()
            process.stdin.close()
            stop = wait_process(process, limits, guard)
        if stop:
            return _error(stop, 'worker stopped by configured resource policy', 2)
        result_path = attempt / 'result.json'
        if not result_path.is_file():
            return _error('worker_crashed', 'worker exited without an atomic result (return code ' + str(process.returncode) + ')', 1)
        result = _read_json(result_path)
        if type(result.get('ok')) is not bool or not isinstance(result.get('report'), dict) or not isinstance(result.get('acceptance'), str):
            return _error('invalid_worker_result', 'worker result envelope is invalid')
        if process.returncode != result.get('exit_code'):
            return _error('worker_exit_mismatch', 'worker result and process return code disagree')
        if not _artifacts_valid(result, attempt.resolve()):
            return _error('invalid_worker_artifacts', 'worker artifacts do not match saved hashes')
        log_path = attempt / 'worker.log'
        result['artifacts']['log'] = str(log_path)
        result['artifact_hashes']['log'] = file_hash(log_path)
        result['artifact_bytes'] = sum(Path(path).stat().st_size for path in set(result['artifacts'].values()))
        report = result['report']
        result['metrics'] = {key: report[key] for key in ('sampled_max_mm', 'tolerance_mm')
                             if isinstance(report.get(key), (int, float)) and not isinstance(report[key], bool)}
        result.pop('report', None)  # Full evidence belongs to immutable report.json.
        return result
    except (ResourceError, OSError) as exc:
        if process is not None and process.poll() is None:
            if guard is not None:
                guard.terminate()
            else:
                process.kill()
                process.wait()
        return _error(getattr(exc, 'code', 'worker_start_failed'), exc)
    finally:
        if process is not None and process.poll() is None and guard is not None:
            guard.terminate()
        if guard is not None:
            guard.close()


def _execute(run_dir, plan_path, state, retry_failed):
    plan = load_plan(plan_path)
    binding = runtime_binding()
    state['plan_path'] = str(plan_path)
    state['runtime'] = binding
    revision = _digest(plan)
    revision_path = run_dir / 'plans' / (revision + '.json')
    if not revision_path.exists():
        _atomic_json(revision_path, plan)
    state['plan_sha256'] = revision
    state['current_tasks'] = [task['id'] for task in plan['tasks']]
    state['reused_tasks'] = 0
    state['execution_complete'] = False
    _save(run_dir, state)
    for task in plan['tasks']:
        identity = task['id']
        old = state['tasks'].get(identity, {})
        blocked = [name for name in task['depends_on'] if state['tasks'][name].get('status') != 'succeeded']
        if blocked:
            state['tasks'][identity] = {'status': 'blocked', 'operation': task['operation'],
                                        'error_code': 'dependency_failed', 'error': 'failed dependencies: ' + ', '.join(blocked),
                                        'exit_code': 2, 'previous_attempt': old.get('attempt')}
            _save(run_dir, state)
            continue
        try:
            inputs = _resolved_inputs(task, plan_path.parent, state)
            fingerprint, sources = _fingerprint(task, inputs, state, binding)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            state['tasks'][identity] = {'status': 'failed', 'operation': task['operation'],
                                        **_error('input_binding_failed', exc), 'previous_attempt': old.get('attempt')}
            _save(run_dir, state)
            continue
        same = old.get('fingerprint') == fingerprint
        if same and old.get('status') == 'succeeded' and _artifacts_valid(old, run_dir):
            state['reused_tasks'] += 1
            continue
        if (same and old.get('status') == 'failed' and not retry_failed
                and old.get('error_code') in ('operation_rejected', 'invalid_input', 'operation_error')
                and _artifacts_valid(old, run_dir)):
            continue
        attempt = run_dir / 'tasks' / identity / ('attempt-' + uuid.uuid4().hex)
        attempt.mkdir(parents=True, exist_ok=False)
        record = {'status': 'running', 'operation': task['operation'], 'fingerprint': fingerprint,
                  'sources': sources, 'attempt': str(attempt), 'started': _now(),
                  'previous_attempt': old.get('attempt')}
        state['tasks'][identity] = record
        _save(run_dir, state)  # A crash after here makes a fresh attempt on resume.
        result = _execute_task(task, inputs, attempt, ResourceLimits.from_mapping(task['resources']), fingerprint)
        record.update(result)
        # Recheck all sources after execution to reject changes during a task.
        try:
            after_fingerprint, _ = _fingerprint(task, inputs, state, runtime_binding())
            if after_fingerprint != fingerprint:
                record.update(_error('input_changed_during_execution', 'input, implementation or runtime changed during execution'))
                record['ok'] = False
        except (OSError, ValueError, KeyError) as exc:
            record.update(_error('input_changed_during_execution', exc))
            record['ok'] = False
        record['status'] = 'succeeded' if record.get('ok') is True else 'failed'
        record['finished'] = _now()
        record['log'] = str(attempt / 'worker.log')
        _atomic_json(attempt / 'receipt.json', record)
        _save(run_dir, state)
    state['execution_complete'] = True
    _save(run_dir, state)
    return _summary(state, run_dir)


def _summary(state, run_dir):
    tasks = [state['tasks'].get(identity, {'status': 'pending'}) for identity in state.get('current_tasks', [])]
    counts = dict(sorted(Counter(task['status'] for task in tasks).items()))
    errors = []
    for identity in state.get('current_tasks', []):
        task = state['tasks'].get(identity, {})
        if task.get('status') in ('failed', 'blocked') and len(errors) < 5:
            errors.append({'task': identity, 'code': task.get('error_code', 'task_failed'),
                           'message': task.get('error', task.get('acceptance', 'task rejected'))[:240],
                           'report': task.get('artifacts', {}).get('report')})
    complete = state.get('execution_complete', False)
    success = bool(tasks) and complete and all(task.get('status') == 'succeeded' for task in tasks)
    exit_code = 0 if success else (1 if any(task.get('exit_code') == 1 for task in tasks) else 2)
    return _bounded_summary({'ok': success, 'exit_code': exit_code, 'execution_complete': complete,
            'acceptance': 'execution_complete_scoped_results' if complete else 'execution_incomplete',
            'counts': counts, 'task_count': len(tasks), 'reused_tasks': state.get('reused_tasks', 0),
            'artifact_bytes': sum(task.get('artifact_bytes', 0) for task in tasks),
            'errors': errors, 'checkpoint': str(run_dir / 'checkpoint.json'),
            'scope': 'Execution completion only; individual reports define geometry, material and motion evidence.'})


def _bounded_summary(value):
    """Keep fixed summary fields and progress within the 4096-byte wire budget."""
    while len(_json_bytes(value)) > 4096 and value.get('errors'):
        value['errors'].pop()
        value['additional_errors'] = value.get('additional_errors', 0) + 1
    if len(_json_bytes(value)) > 4096:
        value['checkpoint'] = value.get('checkpoint', '')[:500]
        value['path_display_truncated'] = True
    return value


def _progress_snapshot(state, run_dir):
    """Read a small advisory file only; never import CAD or rehash evidence."""
    for identity in state.get('current_tasks', []):
        task = state['tasks'].get(identity, {})
        if task.get('status') != 'running':
            continue
        try:
            attempt = Path(task['attempt']).resolve()
            if attempt.parent != (run_dir / 'tasks' / _safe_id(identity)).resolve():
                continue
            path = (attempt / 'output' / 'progress.json').resolve()
            if not path.is_relative_to(attempt):
                continue
            with path.open('rb') as handle:
                payload = handle.read(8193)
            if len(payload) > 8192:
                continue
            def invalid_constant(value):
                raise ValueError('non-finite progress number')
            document = json.loads(payload.decode('utf-8'), parse_constant=invalid_constant)
            if not isinstance(document, dict) or document.get('schema') != 'guided-mesh-cad.progress.v1':
                continue
            phase, completed, total = document.get('phase'), document.get('completed'), document.get('total')
            current = document.get('current_instance')
            if not isinstance(phase, str) or not phase.strip() or type(completed) is not int or not 0 <= completed <= 2**53 - 1:
                continue
            if total is not None and (type(total) is not int or not completed <= total <= 2**53 - 1):
                continue
            if current is not None and not isinstance(current, str):
                continue
            phase.encode('utf-8')
            if current is not None:
                current.encode('utf-8')
            return {'task': identity, 'phase': phase[:80], 'completed': completed, 'total': total,
                    'current_instance': current[:120] if current is not None else None,
                    'snapshot_only': True, 'acceptance': 'not_evaluated'}
        except (OSError, ValueError, KeyError, TypeError, RecursionError):
            continue
    return None


def _public_failure(exc, run_dir):
    return {'ok': False, 'execution_complete': False, 'exit_code': 1,
            'error_code': getattr(exc, 'code', 'invalid_workflow'), 'error': str(exc)[:600],
            'checkpoint': str(Path(run_dir).resolve() / 'checkpoint.json')}


def run(plan_path, out_dir):
    plan_path, run_dir = Path(plan_path).resolve(), Path(out_dir).resolve()
    try:
        load_plan(plan_path)
        run_dir.mkdir(parents=True, exist_ok=True)
        with _run_lock(run_dir):
            if any(path.name != '.run.lock' for path in run_dir.iterdir()):
                raise ValueError('run output must be empty; use resume for an existing run')
            state = {'schema': STATE_SCHEMA, 'created': _now(), 'tasks': {}}
            return _execute(run_dir, plan_path, state, False)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return _public_failure(exc, run_dir)


def resume(run_dir, retry_failed=False):
    run_dir = Path(run_dir).resolve()
    try:
        with _run_lock(run_dir):
            state = _read_json(run_dir / 'checkpoint.json')
            if state.get('schema') != STATE_SCHEMA:
                raise ValueError('unsupported checkpoint schema')
            return _execute(run_dir, Path(state['plan_path']).resolve(), state, retry_failed)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return _public_failure(exc, run_dir)


def status(run_dir):
    run_dir = Path(run_dir).resolve()
    try:
        state = _read_json(run_dir / 'checkpoint.json')
        if state.get('schema') != STATE_SCHEMA:
            raise ValueError('unsupported checkpoint schema')
        result = _summary(state, run_dir)
        progress = _progress_snapshot(state, run_dir)
        if progress is not None:
            result['progress'] = progress
        return _bounded_summary(result)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return _public_failure(exc, run_dir)


def diagnose(run_dir, task_id, view='report', offset=0, limit=5, pointer=None):
    """Bounded stored-evidence reader: no process launch and no native imports."""
    run_dir = Path(run_dir).resolve()
    try:
        _safe_id(task_id)
        if view not in ('errors', 'metrics', 'evidence', 'log', 'report'):
            raise ValueError('invalid diagnosis view')
        if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0 or isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 100:
            raise ValueError('offset must be nonnegative and limit between 1 and 100')
        state = _read_json(run_dir / 'checkpoint.json')
        task = state['tasks'][task_id]
        path = Path(task.get('log', '') if view == 'log' else task.get('artifacts', {}).get('report', '')).resolve()
        checkpoint_errors = view == 'errors' and not task.get('artifacts', {}).get('report')
        if checkpoint_errors:
            path = run_dir / 'checkpoint.json'
        if not path.is_relative_to(run_dir) or not path.is_file():
            raise ValueError('requested task has no stored ' + view)
        if view == 'log':
            rows, total = [], 0
            with path.open(encoding='utf-8', errors='replace') as handle:
                # A worker may produce a very long physical line; bounded readline
                # avoids reading that line into memory. Pagination counts chunks.
                while True:
                    line = handle.readline(1024)
                    if not line:
                        break
                    if offset <= total < offset + limit:
                        rows.append(line.rstrip('\r\n'))
                    total += 1
            value = rows
        else:
            value = dict(task) if checkpoint_errors else _read_json(path)
            if pointer:
                if not isinstance(pointer, str) or not pointer.startswith('/'):
                    raise ValueError('JSON pointer must start with /')
                for part in pointer[1:].split('/'):
                    part = part.replace('~1', '/').replace('~0', '~')
                    value = value[int(part)] if isinstance(value, list) else value[part]
            elif view == 'errors':
                value = {key: val for key, val in value.items() if 'error' in key or 'fail' in key or 'reject' in key}
                if not value:
                    value = {key: task.get(key) for key in ('error_code', 'error', 'acceptance') if task.get(key)}
            elif view == 'metrics':
                value = {key: val for key, val in value.items() if isinstance(val, (int, float, bool)) or any(word in key for word in ('metric', 'distance', 'deviation', 'max_mm', 'coverage', 'tolerance'))}
            elif view == 'evidence':
                value = {'scope': value.get('scope'), 'artifacts': task.get('artifacts', {}),
                         'artifact_hashes': task.get('artifact_hashes', {}), 'sources': task.get('sources', {})}
            if isinstance(value, dict):
                rows = list(value.items())
                total = len(rows)
                value = dict(rows[offset:offset + limit])
            elif isinstance(value, list):
                total = len(value)
                value = value[offset:offset + limit]
            else:
                total = 1
                value = value if offset == 0 else None
        # Bound every returned value as well as the complete wire response.
        def bounded(item, depth=0):
            if depth >= 4:
                return {'omitted': 'nested value; use --pointer'}
            if isinstance(item, str):
                return item[:500] + (' [truncated]' if len(item) > 500 else '')
            if isinstance(item, dict):
                return {str(k)[:100]: bounded(v, depth + 1) for k, v in list(item.items())[:20]}
            if isinstance(item, list):
                return [bounded(v, depth + 1) for v in item[:20]]
            return item
        result = {'ok': True, 'exit_code': 0, 'task': task_id, 'view': view,
                  'offset': offset, 'limit': limit, 'total': total,
                  'next_offset': offset + limit if offset + limit < total else None,
                  'source': str(path), 'value': bounded(value)}
        if len(_json_bytes(result)) > 3900:
            result['value'] = {'omitted': 'page exceeds response budget; reduce --limit or use --pointer'}
        return result
    except (OSError, ValueError, KeyError, TypeError, IndexError) as exc:
        return _public_failure(exc, run_dir)
