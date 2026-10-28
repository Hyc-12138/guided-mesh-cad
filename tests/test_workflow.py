"""Pure runner tests. These never import or execute CAD/native geometry."""
import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import uuid
from unittest.mock import patch

from guided_mesh_cad import workflow as wf
from guided_mesh_cad.resources import ResourceError, ResourceLimits, ProcessGuard, check_reserves


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='workflow-', dir=Path(__file__).parent)
        self.base = Path(self.temp.name)
        self.source = self.base / 'source.txt'
        self.source.write_text('original input', encoding='utf-8')
        self.plan_path = self.base / 'plan.json'
        self.run_dir = self.base / 'run'

    def tearDown(self):
        self.temp.cleanup()

    def task(self, name='inspect', **overrides):
        return {'id': name, 'operation': 'inspect-input', 'inputs': {'source': 'source.txt'},
                'params': {}, 'scope': 'identity evidence only', **overrides}

    def plan(self, tasks=None, **overrides):
        plan = {'schema': wf.SCHEMA, 'tasks': tasks or [self.task()], **overrides}
        self.plan_path.write_text(json.dumps(plan), encoding='utf-8')
        return plan

    def state(self):
        return json.loads((self.run_dir / 'checkpoint.json').read_text(encoding='utf-8'))

    def run_plan(self, tasks=None, **overrides):
        self.plan(tasks, **overrides)
        result = wf.run(self.plan_path, self.run_dir)
        self.assertTrue(result['execution_complete'], result)
        return result

    def test_isolated_inspection_is_evidence_only_and_unchanged_resume_reuses(self):
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            result = self.run_plan()
        self.assertEqual(stdout.getvalue(), '')
        self.assertEqual(result['exit_code'], 0)
        state = self.state()
        task = state['tasks']['inspect']
        self.assertEqual(task['acceptance'], 'input_evidence_only')
        report = json.loads(Path(task['artifacts']['report']).read_text(encoding='utf-8'))
        self.assertIs(report['geometry_acceptance'], False)
        resumed = wf.resume(self.run_dir)
        self.assertEqual(resumed['reused_tasks'], 1)
        self.assertEqual(self.state()['tasks']['inspect']['attempt'], task['attempt'])
        self.assertLess(len(json.dumps(resumed)), 4000)

    def test_worker_final_progress_is_frozen_in_output_manifest(self):
        self.run_plan()
        task = self.state()['tasks']['inspect']
        progress_path = Path(task['attempt']) / 'output' / 'progress.json'
        progress = json.loads(progress_path.read_text(encoding='utf-8'))
        self.assertEqual(progress['phase'], 'dispatch_finished')
        self.assertEqual((progress['completed'], progress['total']), (1, 1))
        self.assertTrue(progress['snapshot_only'])
        manifest = json.loads(Path(task['artifacts']['output_manifest']).read_text(encoding='utf-8'))
        binding = next(ref for ref in manifest['refs'] if ref['path'] == str(progress_path))
        self.assertEqual(binding['sha256'], wf.file_hash(progress_path))
        self.assertEqual(wf.resume(self.run_dir)['reused_tasks'], 1)

    def running_snapshot_fixture(self):
        self.run_plan()
        state = self.state()
        state['execution_complete'] = False
        state['tasks']['inspect']['status'] = 'running'
        wf._atomic_json(self.run_dir / 'checkpoint.json', state)
        return Path(state['tasks']['inspect']['attempt']) / 'output' / 'progress.json'

    def test_status_reads_only_bounded_advisory_running_progress(self):
        path = self.running_snapshot_fixture()
        wf._atomic_json(path, {'schema': 'guided-mesh-cad.progress.v1', 'phase': 'verify-assembly',
                              'completed': 2, 'total': 5, 'current_instance': 'part-A',
                              'snapshot_only': False, 'passed': True, 'unused': 'x' * 1000})
        with patch.object(wf, 'file_hash', side_effect=AssertionError('status must not hash')), \
             patch.object(wf, 'runtime_binding', side_effect=AssertionError('status must not inspect runtime')), \
             patch.object(wf.subprocess, 'Popen', side_effect=AssertionError('status must not spawn')):
            result = wf.status(self.run_dir)
        self.assertEqual(result['counts'], {'running': 1})
        self.assertFalse(result['ok'])
        self.assertEqual(result['progress'], {'task': 'inspect', 'phase': 'verify-assembly',
                                             'completed': 2, 'total': 5, 'current_instance': 'part-A',
                                             'snapshot_only': True, 'acceptance': 'not_evaluated'})
        self.assertLessEqual(len(wf._json_bytes(result)), 4096)

    def test_malformed_unknown_and_oversized_progress_preserve_known_checkpoint(self):
        path = self.running_snapshot_fixture()
        invalid = [b'{partial', b'{"schema":"unknown"}', b' ' * 8193,
                   b'{"schema":"guided-mesh-cad.progress.v1","phase":"x","completed":NaN}',
                   b'{"schema":"guided-mesh-cad.progress.v1","phase":"x","completed":true}',
                   b'{"schema":"guided-mesh-cad.progress.v1","phase":"x","completed":4,"total":2}',
                   b'{"schema":"guided-mesh-cad.progress.v1","phase":"\\ud800","completed":0}',
                   b'[' * 2000 + b']' * 2000]
        for payload in invalid:
            path.write_bytes(payload)
            result = wf.status(self.run_dir)
            self.assertEqual(result['counts'], {'running': 1}, result)
            self.assertFalse(result['execution_complete'])
            self.assertNotIn('progress', result)
            self.assertLessEqual(len(wf._json_bytes(result)), 4096)
        path.unlink()
        self.assertEqual(wf.status(self.run_dir)['counts'], {'running': 1})

    def test_progress_text_is_clipped_and_finished_tasks_are_not_live_progress(self):
        path = self.running_snapshot_fixture()
        wf._atomic_json(path, {'schema': 'guided-mesh-cad.progress.v1', 'phase': 'x' * 2000,
                              'completed': 0, 'total': None, 'current_instance': 'y' * 2000})
        result = wf.status(self.run_dir)
        self.assertEqual(len(result['progress']['phase']), 80)
        self.assertEqual(len(result['progress']['current_instance']), 120)
        self.assertLessEqual(len(wf._json_bytes(result)), 4096)
        state = self.state()
        state['tasks']['inspect']['status'] = 'succeeded'
        state['execution_complete'] = True
        wf._atomic_json(self.run_dir / 'checkpoint.json', state)
        self.assertNotIn('progress', wf.status(self.run_dir))

    def test_tampered_and_missing_reports_cannot_reuse(self):
        self.run_plan()
        first = self.state()['tasks']['inspect']
        Path(first['artifacts']['report']).write_text('{"altered":true}', encoding='utf-8')
        result = wf.resume(self.run_dir)
        self.assertTrue(result['ok'], result)
        second = self.state()['tasks']['inspect']
        self.assertNotEqual(first['attempt'], second['attempt'])
        Path(second['artifacts']['report']).unlink()
        result = wf.resume(self.run_dir)
        self.assertTrue(result['ok'], result)
        self.assertNotEqual(second['attempt'], self.state()['tasks']['inspect']['attempt'])

    def test_extra_partial_output_invalidates_complete_local_manifest(self):
        self.run_plan()
        old = self.state()['tasks']['inspect']
        Path(old['artifacts']['report']).with_name('unexpected.partial').write_bytes(b'partial evidence')
        self.assertTrue(wf.resume(self.run_dir)['ok'])
        self.assertNotEqual(old['attempt'], self.state()['tasks']['inspect']['attempt'])

    def test_external_checkpoint_evidence_manifest_is_hash_and_owner_bound(self):
        attempt = self.run_dir / 'tasks' / 'verify' / ('attempt-' + uuid.uuid4().hex)
        out = attempt / 'output'
        out.mkdir(parents=True)
        root = attempt.parent / 'verification-checkpoints' / ('a' * 64)
        root.mkdir(parents=True)
        evidence = root / 'leaf.json.gz'
        evidence.write_bytes(b'compressed evidence fixture')
        binding = {'task_fingerprint': 'a' * 64}
        request = {'operation': 'verify-assembly', 'checkpoint_root': str(root), 'checkpoint_binding': binding}
        wf._atomic_json(attempt / 'request.json', request)
        report = out / 'report.json'
        wf._atomic_json(report, {'finite_evidence_only': True})
        manifest_path = out / 'evidence_manifest.json'
        manifest = {'schema': 'guided-mesh-cad.evidence-manifest.v1', 'checkpoint_root': str(root),
                    'checkpoint_binding': binding, 'refs': [{'path': str(evidence), 'sha256': wf.file_hash(evidence),
                                                          'bytes': evidence.stat().st_size}]}
        wf._atomic_json(manifest_path, manifest)
        def record():
            artifacts = {'report': str(report), 'evidence_manifest': str(manifest_path)}
            return {'artifacts': artifacts, 'artifact_hashes': {key: wf.file_hash(path) for key, path in artifacts.items()}}
        receipt = record()
        self.assertTrue(wf._artifacts_valid(receipt, self.run_dir))
        evidence.write_bytes(b'tampered evidence')
        self.assertFalse(wf._artifacts_valid(receipt, self.run_dir))
        evidence.write_bytes(b'compressed evidence fixture')
        manifest['checkpoint_binding'] = {'task_fingerprint': 'b' * 64}
        wf._atomic_json(manifest_path, manifest)
        self.assertFalse(wf._artifacts_valid(record(), self.run_dir))
        manifest['checkpoint_binding'] = binding
        manifest['checkpoint_root'] = str(self.base)
        wf._atomic_json(manifest_path, manifest)
        self.assertFalse(wf._artifacts_valid(record(), self.run_dir))

    def test_source_changes_invalidate_dependents_but_preserve_independent_task(self):
        other = self.base / 'other.txt'
        other.write_text('independent', encoding='utf-8')
        self.run_plan([self.task('first'), self.task('second', inputs={'proof': {'task': 'first', 'artifact': 'report'}}),
                       self.task('independent', inputs={'source': 'other.txt'})])
        before = self.state()['tasks']
        self.source.write_text('modified source', encoding='utf-8')
        result = wf.resume(self.run_dir)
        after = self.state()['tasks']
        self.assertTrue(result['ok'], result)
        self.assertEqual(result['reused_tasks'], 1)
        for name in ('first', 'second'):
            self.assertNotEqual(before[name]['attempt'], after[name]['attempt'])
        self.assertEqual(before['independent']['attempt'], after['independent']['attempt'])

    def test_scope_change_invalidates_receipt(self):
        self.run_plan()
        first = self.state()['tasks']['inspect']['attempt']
        self.plan([self.task(scope='different explicitly scoped evidence')])
        self.assertTrue(wf.resume(self.run_dir)['ok'])
        self.assertNotEqual(first, self.state()['tasks']['inspect']['attempt'])

    def test_edited_failed_evidence_cannot_be_cached(self):
        self.run_plan([self.task(params={'expected_sha256': {'source': '0' * 64}})])
        old = self.state()['tasks']['inspect']
        Path(old['artifacts']['report']).write_text('changed evidence', encoding='utf-8')
        self.assertEqual(wf.resume(self.run_dir)['exit_code'], 2)
        self.assertNotEqual(old['attempt'], self.state()['tasks']['inspect']['attempt'])

    def test_deterministic_rejection_requires_retry_and_blocks_only_descendants(self):
        self.run_plan([self.task('reject', params={'expected_sha256': {'source': '0' * 64}}),
                       self.task('blocked', depends_on=['reject']), self.task('independent')])
        first = self.state()['tasks']['reject']['attempt']
        result = wf.resume(self.run_dir)
        self.assertEqual(result['exit_code'], 2)
        self.assertEqual(result['counts'], {'blocked': 1, 'failed': 1, 'succeeded': 1})
        self.assertEqual(first, self.state()['tasks']['reject']['attempt'])
        result = wf.resume(self.run_dir, retry_failed=True)
        self.assertEqual(result['exit_code'], 2)
        self.assertNotEqual(first, self.state()['tasks']['reject']['attempt'])

    def test_interrupted_attempt_is_preserved_and_automatically_retried(self):
        self.plan()
        with patch.object(wf, '_execute_task', side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt):
                wf.run(self.plan_path, self.run_dir)
        interrupted = self.state()['tasks']['inspect']
        self.assertEqual(interrupted['status'], 'running')
        Path(interrupted['attempt'], 'result.json.partial').write_text('unfinished', encoding='utf-8')
        result = wf.resume(self.run_dir)
        self.assertTrue(result['ok'], result)
        current = self.state()['tasks']['inspect']
        self.assertNotEqual(interrupted['attempt'], current['attempt'])
        self.assertTrue(Path(interrupted['attempt'], 'result.json.partial').is_file())

    def test_code_and_runtime_changes_invalidate_success(self):
        binding = wf.runtime_binding()
        with patch.object(wf, 'runtime_binding', return_value=binding):
            self.run_plan()
        first = self.state()['tasks']['inspect']['attempt']
        changed = {**binding, 'code_sha256': 'new-code', 'python': 'changed-runtime'}
        with patch.object(wf, 'runtime_binding', return_value=changed):
            result = wf.resume(self.run_dir)
        self.assertTrue(result['ok'], result)
        self.assertNotEqual(first, self.state()['tasks']['inspect']['attempt'])

    def test_input_changed_during_execution_is_rejected(self):
        execute = wf._execute_task
        def mutate(*args):
            result = execute(*args)
            self.source.write_text('concurrently changed', encoding='utf-8')
            return result
        with patch.object(wf, '_execute_task', side_effect=mutate):
            result = self.run_plan()
        self.assertEqual(result['exit_code'], 1)
        self.assertEqual(self.state()['tasks']['inspect']['error_code'], 'input_changed_during_execution')

    def test_transitive_reconstruction_sources_are_bound_without_cad_import(self):
        construction = self.base / 'construction.obj'
        construction.write_text('mesh bytes', encoding='utf-8')
        reconstruction = self.base / 'reconstruction.json'
        reconstruction.write_text(json.dumps({'mesh': 'source.txt', 'construction_mesh': 'construction.obj'}), encoding='utf-8')
        task = self.task(operation='reconstruct', inputs={'plan': 'reconstruction.json'})
        self.plan([task])
        normalized = wf.load_plan(self.plan_path)['tasks'][0]
        inputs = wf._resolved_inputs(normalized, self.base, {'tasks': {}})
        binding = wf.runtime_binding()
        first, sources = wf._fingerprint(normalized, inputs, {'tasks': {}}, binding)
        self.assertIn(str(construction.resolve()), sources)
        self.assertIn(str(self.source.resolve()), sources)
        construction.write_text('changed transitive bytes', encoding='utf-8')
        second, _ = wf._fingerprint(normalized, inputs, {'tasks': {}}, binding)
        self.assertNotEqual(first, second)

    def test_bounded_diagnostics_and_json_pointer(self):
        self.run_plan()
        task = self.state()['tasks']['inspect']
        Path(task['log']).write_text('x' * 100000 + '\n' + 'second\n', encoding='utf-8')
        result = wf.diagnose(self.run_dir, 'inspect', 'log', offset=2, limit=100)
        self.assertTrue(result['ok'], result)
        self.assertLess(len(json.dumps(result)), 4000)
        result = wf.diagnose(self.run_dir, 'inspect', 'report', pointer='/inputs/source/sha256')
        self.assertEqual(result['value'], wf.file_hash(self.source))
        self.assertEqual(wf.diagnose(self.run_dir, 'inspect', limit=101)['exit_code'], 1)
        self.assertEqual(wf.diagnose(self.run_dir, '../outside')['exit_code'], 1)

    def test_lock_is_exclusive_and_does_not_depend_on_stale_pid(self):
        self.run_dir.mkdir()
        with wf._run_lock(self.run_dir):
            with self.assertRaises(wf.WorkflowError) as raised:
                with wf._run_lock(self.run_dir):
                    pass
            self.assertEqual(raised.exception.code, 'run_locked')
        with wf._run_lock(self.run_dir):
            pass

    def test_resource_admission_rejects_without_starting_process(self):
        self.plan()
        with patch.object(wf, 'check_reserves', side_effect=ResourceError('fixture low memory')), patch.object(wf.subprocess, 'Popen') as spawn:
            result = wf.run(self.plan_path, self.run_dir)
        spawn.assert_not_called()
        self.assertEqual(result['exit_code'], 2)
        self.assertEqual(self.state()['tasks']['inspect']['error_code'], 'resource_unavailable')
        errors = wf.diagnose(self.run_dir, 'inspect', 'errors')
        self.assertTrue(errors['ok'], errors)
        self.assertEqual(errors['value']['error_code'], 'resource_unavailable')
        self.assertTrue(wf.resume(self.run_dir)['ok'])

    def test_explicit_timeout_stops_pure_worker_and_remains_resumable(self):
        result = self.run_plan(resources={'timeout_seconds': 1e-9})
        self.assertEqual(result['exit_code'], 2)
        self.assertEqual(self.state()['tasks']['inspect']['error_code'], 'timeout')
        self.plan(resources={'timeout_seconds': None})
        self.assertTrue(wf.resume(self.run_dir)['ok'])

    def test_invalid_operations_cycles_and_unsafe_names_never_spawn(self):
        cases = [[self.task(operation='shell')], [self.task('a', depends_on=['b']), self.task('b', depends_on=['a'])],
                 [self.task('../escape')], [self.task('CON')], [self.task(scope='')],
                 [self.task(operation='reconstruct', inputs={'plan': 'source.txt'}, params={'ignored': True})],
                 [self.task(operation='diagnose-native', inputs={'native': 'source.txt'}, params={'ignored': True})],
                 [self.task(operation='validate', inputs={'mesh': 'source.txt', 'step': 'source.txt'}, params={'ignored': True})],
                 [self.task(operation='audit-domains', inputs={'source': 'source.txt', 'domains': 'source.txt'}, params={'ignored': True})]]
        for tasks in cases:
            self.plan(tasks)
            with patch.object(wf.subprocess, 'Popen') as spawn:
                result = wf.run(self.plan_path, self.run_dir)
            self.assertEqual(result['exit_code'], 1, result)
            spawn.assert_not_called()


class ResourceTests(unittest.TestCase):
    def test_real_stdlib_process_cannot_exceed_memory_job(self):
        # Pure Python allocation only: no native CAD process or geometry fixture.
        limits = ResourceLimits.from_mapping({'memory_limit_mib': 256})
        check_reserves(limits)
        code = ("import sys,os\n"
                "sys.stdin.buffer.readline()\n"
                "if os.name != 'nt':\n"
                " import resource\n"
                " resource.setrlimit(resource.RLIMIT_AS,(268435456,268435456))\n"
                "try:\n"
                " value=bytearray(536870912)\n"
                " print('UNLIMITED')\n"
                "except MemoryError:\n"
                " print('LIMITED')\n")
        process = subprocess.Popen([sys.executable, '-c', code], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, start_new_session=os.name != 'nt',
                                   creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
        guard = ProcessGuard(process, limits)
        try:
            output, _ = process.communicate(b'GO\n', timeout=15)
            self.assertEqual(process.returncode, 0, output)
            self.assertEqual(output.strip(), b'LIMITED')
        finally:
            if process.poll() is None:
                guard.terminate()
            guard.close()

    def test_exact_byte_reserves_and_invalid_limits(self):
        limits = ResourceLimits.from_mapping()
        self.assertEqual(limits.memory_limit_mib, 4096)
        self.assertIsNone(limits.timeout_seconds)
        check_reserves(limits, {'available_physical': 1536 * 1048576, 'available_commit': 1536 * 1048576})
        with self.assertRaises(ResourceError):
            check_reserves(limits, {'available_physical': 1536 * 1048576 - 1, 'available_commit': 10**12})
        for value in ({'memory_limit_mib': 0}, {'timeout_seconds': -1}, {'poll_seconds': 0},
                      {'memory_limit_mib': float('nan')}, {'shell': True}, {'reserve_commit_mib': True}):
            with self.assertRaises(ValueError):
                ResourceLimits.from_mapping(value)


if __name__ == '__main__':
    unittest.main()
