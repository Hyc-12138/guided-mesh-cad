import argparse
import json
import sys
from . import __version__


def main(argv=None):
    parser = argparse.ArgumentParser(description='Guided freeform patch reconstruction; requires an explicit source-bound plan.')
    parser.add_argument('--version', action='version', version=__version__)
    commands = parser.add_subparsers(dest='command', required=True)
    r = commands.add_parser('reconstruct'); r.add_argument('plan'); r.add_argument('--out', required=True)
    v = commands.add_parser('validate'); v.add_argument('mesh'); v.add_argument('step'); v.add_argument('--report', required=True)
    v.add_argument('--tolerance-mm', type=float, default=.01); v.add_argument('--samples', type=int, default=3000)
    d = commands.add_parser('diagnose-step'); d.add_argument('step'); d.add_argument('--report', required=True)
    n = commands.add_parser('diagnose-native'); n.add_argument('native'); n.add_argument('--report', required=True)
    commands.add_parser('doctor')
    e = commands.add_parser('example'); e.add_argument('name', choices=['freeform','loop']); e.add_argument('--out', required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == 'example':
            from importlib import import_module
            from pathlib import Path
            if Path(args.out).exists() and any(Path(args.out).iterdir()):
                raise ValueError('example output directory must be empty')
            import_module(f'guided_mesh_cad.examples.make_{args.name}_example').main(['--out',args.out])
            return 0
        if args.command == 'doctor':
            import cadquery, OCP, vtk, numpy
            data = {'package':__version__, 'python':sys.version, 'cadquery':cadquery.__version__,
                    'ocp':OCP.__version__, 'vtk':vtk.vtkVersion.GetVTKVersion(), 'numpy':numpy.__version__}
            print(json.dumps(data, indent=2)); return 0
        from .pipeline import reconstruct, validate, dump
        if args.command == 'reconstruct':
            report = reconstruct(args.plan, args.out)
            ok = report['status'] == 'sampled_geometry_pass'
        elif args.command in ('diagnose-step', 'diagnose-native'):
            from .cad import read_native_unchecked, shape_check_report, require_solid, PatchTopologyError
            path = args.step if args.command == 'diagnose-step' else args.native
            shape = read_native_unchecked(path)
            report = shape_check_report(shape)
            try:
                require_solid(shape)
                ok = True
            except PatchTopologyError as exc:
                ok = False
                report['formal_solid_gate_error'] = str(exc)
            report['formal_solid_gate_passed'] = ok
            dump(args.report, report)
        else:
            report = validate(args.mesh, args.step, args.report, args.tolerance_mm, args.samples)
            ok = report['passed']
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0 if ok else 2
    except Exception as exc:
        print(json.dumps({'status':'failed','type':type(exc).__name__,'error':str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 1
