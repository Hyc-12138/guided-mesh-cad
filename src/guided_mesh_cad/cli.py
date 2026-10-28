import argparse
import contextlib
import json
import os
from pathlib import Path
import sys
import traceback
import uuid
from . import __version__
from .reporting import print_report


@contextlib.contextmanager
def _operation_log(path):
    """Capture Python and native fd1/fd2 chatter; keep the CLI response bounded."""
    path.parent.mkdir(parents=True, exist_ok=True)
    saved = {}
    with path.open('x', encoding='utf-8') as log:
        try:
            for stream in (sys.stdout, sys.stderr):
                try: stream.flush()
                except Exception: pass
            for fd in (1, 2):
                try:
                    saved[fd] = os.dup(fd)
                    os.dup2(log.fileno(), fd)
                except OSError: pass
            with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                try: yield
                except BaseException:
                    traceback.print_exc(file=log)
                    raise
        finally:
            log.flush()
            for fd, previous in saved.items():
                os.dup2(previous, fd)
                os.close(previous)


def main(argv=None):
    parser = argparse.ArgumentParser(description='Source-bound guided reconstruction and resumable native workflows.')
    parser.add_argument('--version', action='version', version=__version__)
    commands = parser.add_subparsers(dest='command', required=True)
    r = commands.add_parser('reconstruct'); r.add_argument('plan'); r.add_argument('--out', required=True)
    v = commands.add_parser('validate'); v.add_argument('mesh'); v.add_argument('step'); v.add_argument('--report', required=True)
    v.add_argument('--tolerance-mm', type=float, default=.01); v.add_argument('--samples', type=int, default=3000)
    d = commands.add_parser('diagnose-step'); d.add_argument('step'); d.add_argument('--report', required=True)
    n = commands.add_parser('diagnose-native'); n.add_argument('native'); n.add_argument('--report', required=True)
    e = commands.add_parser('example'); e.add_argument('name', choices=['freeform', 'loop', 'workflow']); e.add_argument('--out', required=True)
    for sub in (r, v, d, n, e): sub.add_argument('--full', action='store_true', help='Print the full report instead of the default bounded summary.')
    commands.add_parser('doctor')
    run = commands.add_parser('run'); run.add_argument('plan'); run.add_argument('--out', required=True)
    resume = commands.add_parser('resume'); resume.add_argument('run_dir'); resume.add_argument('--retry-failed', action='store_true')
    status = commands.add_parser('status'); status.add_argument('run_dir')
    diag = commands.add_parser('diagnose'); diag.add_argument('run_dir'); diag.add_argument('--task', required=True)
    diag.add_argument('--view', choices=['errors', 'metrics', 'evidence', 'log', 'report'], default='errors')
    diag.add_argument('--offset', type=int, default=0); diag.add_argument('--limit', type=int, default=5)
    diag.add_argument('--pointer', default=None, help='JSON pointer for stored report inspection; no CAD is rerun.')
    args = parser.parse_args(argv)
    report_path = log_path = None
    try:
        if args.command in ('run', 'resume', 'status', 'diagnose'):
            from . import workflow
            if args.command == 'run': result = workflow.run(args.plan, args.out)
            elif args.command == 'resume': result = workflow.resume(args.run_dir, retry_failed=args.retry_failed)
            elif args.command == 'status': result = workflow.status(args.run_dir)
            else: result = workflow.diagnose(args.run_dir, args.task, view=args.view, offset=args.offset, limit=args.limit, pointer=args.pointer)
            print(json.dumps(result, ensure_ascii=False, allow_nan=False, separators=(',', ':')))
            return int(result.get('exit_code', 0))
        if args.command == 'doctor':
            import cadquery, OCP, vtk, numpy
            data = {'package': __version__, 'python': sys.version, 'cadquery': cadquery.__version__,
                    'ocp': OCP.__version__, 'vtk': vtk.vtkVersion.GetVTKVersion(), 'numpy': numpy.__version__}
            print(json.dumps(data, ensure_ascii=False, separators=(',', ':')))
            return 0
        requested = Path(args.out if hasattr(args, 'out') else args.report).resolve()
        log_path = requested.parent / '.gmc-logs' / (args.command + '-' + uuid.uuid4().hex + '.log')
        with _operation_log(log_path):
            if args.command == 'example':
                from importlib import import_module
                if requested.exists() and any(requested.iterdir()): raise ValueError('example output directory must be empty')
                import_module(f'guided_mesh_cad.examples.make_{args.name}_example').main(['--out', str(requested)])
                report = {'status': 'example_created', 'acceptance': 'not_checked', 'scope': 'Example input only; no reconstruction acceptance.'}
                report_path = requested / 'plan.json'; ok = True
            else:
                from .pipeline import reconstruct, validate, dump
                if args.command == 'reconstruct':
                    report = reconstruct(args.plan, str(requested))
                    report_path = requested / 'manifest.json'
                    ok = report['status'] == 'sampled_geometry_pass'
                elif args.command in ('diagnose-step', 'diagnose-native'):
                    from .cad import read_native_unchecked, shape_check_report, require_solid, PatchTopologyError
                    path = args.step if args.command == 'diagnose-step' else args.native
                    shape = read_native_unchecked(path)
                    report = shape_check_report(shape)
                    try:
                        require_solid(shape); ok = True
                    except PatchTopologyError as exc:
                        ok = False; report['formal_solid_gate_error'] = str(exc)
                    report['formal_solid_gate_passed'] = ok
                    requested.parent.mkdir(parents=True, exist_ok=True)
                    dump(requested, report); report_path = requested
                else:
                    requested.parent.mkdir(parents=True, exist_ok=True)
                    report = validate(args.mesh, args.step, str(requested), args.tolerance_mm, args.samples)
                    report_path = requested; ok = report['passed']
        print_report(report, full=args.full, command=args.command, report_path=report_path, log_path=log_path, ok=ok)
        return 0 if ok else 2
    except Exception as exc:
        report = {'status': 'failed', 'type': type(exc).__name__, 'error_code': 'INVALID_INPUT' if isinstance(exc, (ValueError, FileNotFoundError)) else 'RUNTIME_ERROR', 'error': str(exc)}
        print_report(report, full=getattr(args, 'full', False), command=args.command, report_path=report_path, log_path=log_path, ok=False)
        return 1

