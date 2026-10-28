"""Explicit, fail-closed native positive-material union checks.

These checks run at the existing OCCT precisions. They are runtime evidence,
not a mathematical set proof, source reconstruction or contact acceptance.
"""

from collections.abc import Mapping
import io
import math

from .cad import PatchTopologyError, _require_native_integrity, shape_check_report, volume
from .material_boundary import _fingerprint


class _NativeWarningUnresolved(ValueError):
    """A native warning leaves this operation's material evidence unresolved."""


def _named_inputs(named_inputs):
    if not isinstance(named_inputs, Mapping) or not named_inputs:
        raise ValueError('named_inputs must be a nonempty ordered mapping of names to native shapes')
    items = list(named_inputs.items())
    if any(not isinstance(name, str) or not name.strip() for name, _ in items):
        raise ValueError('every positive input requires a nonempty string name')
    return items


def _witnesses(inside_witnesses, names):
    if inside_witnesses is None:
        return {}
    if not isinstance(inside_witnesses, Mapping) or set(inside_witnesses) - set(names):
        raise ValueError('inside_witnesses must map known input names to finite XYZ points')
    result = {}
    for name, points in inside_witnesses.items():
        parsed = []
        for point in points:
            if len(point) != 3 or any(isinstance(v, bool) for v in point):
                raise ValueError('each inside witness must be a finite XYZ point')
            xyz = tuple(float(v) for v in point)
            if not all(math.isfinite(v) for v in xyz):
                raise ValueError('each inside witness must be a finite XYZ point')
            parsed.append(xyz)
        if not parsed:
            raise ValueError('a supplied witness group must not be empty')
        result[name] = parsed
    return result


def _expected_count(expected_solids):
    if expected_solids is not None and (isinstance(expected_solids, bool) or
            not isinstance(expected_solids, int) or expected_solids < 1):
        raise ValueError('expected_solids must be a positive integer or None')


def _copy(shape):
    import cadquery as cq
    from OCP.BRepBuilderAPI import BRepBuilderAPI_Copy
    operation = BRepBuilderAPI_Copy(shape.wrapped, True, False)
    if not operation.IsDone() or operation.Shape().IsNull():
        raise ValueError('native independent copy failed')
    return cq.Shape.cast(operation.Shape())


def _native_boolean(kind, left, right, row):
    """Always operate on independent geometry, without repair or fuzzy growth."""
    import cadquery as cq
    from OCP.BOPAlgo import BOPAlgo_BOP, BOPAlgo_FUSE, BOPAlgo_CUT
    from OCP.TopTools import TopTools_ListOfShape
    arguments, tools = TopTools_ListOfShape(), TopTools_ListOfShape()
    arguments.Append(_copy(left).wrapped)
    tools.Append(_copy(right).wrapped)
    # Own the operation directly. OCP 7.9's BRepAlgoAPI wrapper exposes IsDone
    # but hides errors/report; no borrowed DSFiller or builder handles are used.
    operation = BOPAlgo_BOP()
    operation.SetOperation(BOPAlgo_FUSE if kind == 'fuse' else BOPAlgo_CUT)
    operation.SetArguments(arguments)
    operation.SetTools(tools)
    operation.SetRunParallel(False)
    operation.SetNonDestructive(True)
    operation.SetFuzzyValue(0.0)
    operation.Perform()
    errors, warnings = io.BytesIO(), io.BytesIO()
    operation.DumpErrors(errors)
    operation.DumpWarnings(warnings)
    row.update(native_operation=kind, native_api='owned BOPAlgo_BOP',
               perform_returned=True, is_done_api_available=False,
               has_errors=bool(operation.HasErrors()),
               has_warnings=bool(operation.HasWarnings()), fuzzy_mm=0.0,
               errors=errors.getvalue().decode('utf-8', errors='replace'),
               warnings=warnings.getvalue().decode('utf-8', errors='replace'),
               run_parallel=False, non_destructive=True)
    if row['has_errors']:
        raise ValueError(f'native {kind} did not complete without errors')
    # Empty/valid Shape() output cannot resolve a failed orientation,
    # classification or building substep. Use a generic conservative policy:
    # no native warning is eligible for positive-material certification.
    if row['has_warnings'] or row['warnings'].strip():
        row['evidence_status'] = 'unresolved_native_warning'
        detail = row['warnings'].strip() or 'native warning detail unavailable'
        raise _NativeWarningUnresolved(
            f'native {kind} warning leaves material preservation evidence unresolved: {detail}')
    native = operation.Shape()
    if native.IsNull():
        # A successfully empty Cut is an empty compound, not a null shape.
        raise ValueError(f'native {kind} returned an unresolved null shape')
    return cq.Shape.cast(native)


def _complete_material(shape, row):
    row['check'] = check = shape_check_report(shape)
    if not check['valid'] or not check['solids'] or not check['shells'] or \
            check['closed_shells'] != check['shells']:
        raise ValueError('positive material must contain complete valid closed native solids')
    _require_native_integrity(check)
    row['solid_checks'] = solid_checks = [shape_check_report(solid) for solid in shape.Solids()]
    if any(not value['valid'] or value['solids'] != 1 or not value['shells'] or
           value['closed_shells'] != value['shells'] for value in solid_checks):
        raise ValueError('each positive component must be one valid closed native solid')
    row['solid_volumes_mm3'] = values = [volume(solid)[0] for solid in shape.Solids()]
    if any(not math.isfinite(v) or v <= 0.0 for v in values):
        raise ValueError('every material solid must have finite positive oriented volume')


def _preserved(source, result, row):
    outside = _native_boolean('cut', source, result, row)
    row['outside_topology'] = counts = {
        name: len(getattr(outside, method)()) for name, method in
        [('solids', 'Solids'), ('shells', 'Shells'), ('faces', 'Faces'),
         ('wires', 'Wires'), ('edges', 'Edges'), ('vertices', 'Vertices')]}
    if any(counts.values()):
        raise ValueError('positive input has residual topology outside union material')
    # An empty, invalid difference is not usable containment evidence.
    row['empty_difference_valid'] = bool(outside.isValid())
    if not row['empty_difference_valid']:
        raise ValueError('native empty material difference is invalid')


def _inside(shape, point):
    from OCP.BRepClass3d import BRepClass3d_SolidClassifier
    from OCP.TopAbs import TopAbs_IN
    from OCP.gp import gp_Pnt
    states = [classifier.State() for classifier in
              (BRepClass3d_SolidClassifier(solid.wrapped, gp_Pnt(*point), 1e-9)
               for solid in shape.Solids())]
    return any(state == TopAbs_IN for state in states), [str(state) for state in states]


def _check_witnesses(result, items, witnesses, report):
    rows = report.setdefault('inside_witnesses', [])
    for name, source in items:
        for point in witnesses.get(name, ()):
            row = {'input_name': name, 'point_mm': list(point)}
            rows.append(row)
            source_in, row['input_states'] = _inside(source, point)
            result_in, row['result_states'] = _inside(result, point)
            row.update(input_strict_IN=source_in, result_strict_IN=result_in)
            if not source_in or not result_in:
                raise ValueError('inside witness must be strictly IN both original input and union')


def _run(named_inputs, result, expected_solids, inside_witnesses, construct):
    items = _named_inputs(named_inputs)
    _expected_count(expected_solids)
    witnesses = _witnesses(inside_witnesses, [name for name, _ in items])
    tracked = items + ([] if construct else [('__candidate_result__', result)])
    hashes = [_fingerprint(shape) for _, shape in tracked]
    report = {'operation': 'serial_positive_union' if construct else 'positive_union_preservation_check',
              'status': 'not_checked', 'expected_solids': expected_solids,
              'native_material_preservation_checked': False,
              'native_warning_policy': 'any_native_warning_is_unresolved',
              'inputs': [], 'steps': [], 'original_input_preservation': [],
              'scope': 'Native runtime material checks at existing precisions; not a mathematical proof, source conformity or contact acceptance.',
              'source_conformity_accepted': False, 'contact_accepted': False,
              'native_self_interference_checked': False,
              'full_BOP_geometry_acceptance': False,
              'clean_used': False, 'largest_solid_selected': False,
              'fuzzy_tolerance_mm': 0.0, 'topology_tolerance_adjustment_requested': False}
    try:
        for (name, shape), fingerprint in zip(items, hashes):
            row = {'name': name, 'brep_sha256': fingerprint}
            report['inputs'].append(row)
            _complete_material(shape, row)
        if construct:
            result = _copy(items[0][1])
            for name, operand in items[1:]:
                row = {'operand_name': name, 'fuse': {}, 'previous_material': {}, 'operand_material': {}}
                report['steps'].append(row)
                previous = result
                result = _native_boolean('fuse', previous, operand, row['fuse'])
                _complete_material(result, row)
                _preserved(previous, result, row['previous_material'])
                _preserved(operand, result, row['operand_material'])
                _check_witnesses(result, items[:len(report['steps']) + 1], witnesses, row)
        _complete_material(result, report.setdefault('output', {}))
        if expected_solids is not None and report['output']['check']['solids'] != expected_solids:
            raise ValueError('union solid count disagrees with expected_solids')
        for name, source in items:
            row = {'input_name': name}
            report['original_input_preservation'].append(row)
            _preserved(source, result, row)
        _check_witnesses(result, items, witnesses, report)
        report['output_brep_sha256'] = _fingerprint(result)
    except Exception as exc:
        report.update(status='unresolved_native_warning' if isinstance(exc, _NativeWarningUnresolved)
                      else 'rejected', failure_type=type(exc).__name__, failure=str(exc))
        report['inputs_unchanged'] = all(_fingerprint(shape) == before
            for (_, shape), before in zip(tracked, hashes))
        raise PatchTopologyError(str(exc), report) from exc
    report['inputs_unchanged'] = all(_fingerprint(shape) == before
        for (_, shape), before in zip(tracked, hashes))
    if not report['inputs_unchanged']:
        report.update(status='rejected', failure='an original input changed during union verification')
        raise PatchTopologyError(report['failure'], report)
    report['status'] = 'native_material_preservation_checked'
    report['native_material_preservation_checked'] = True
    return result, report


def validate_positive_union(result, named_inputs, *, expected_solids=None, inside_witnesses=None):
    """Check a candidate contains every named complete positive native input.

    Return a JSON-compatible report; any native warning/error, invalid/open material,
    residual topology, wrong optional Solid count or failed optional strict-IN
    witness raises PatchTopologyError with the incomplete diagnostic report.
    Inputs may each contain multiple closed solids. No extra-material exclusion,
    source geometry, intended contact or reconstruction quality is certified.
    Each supplied witness must be strictly IN its original named input and result.
    A native warning raises with status unresolved_native_warning; it does not
    prove the geometry is wrong, and finite witnesses cannot resolve that warning.
    """
    return _run(named_inputs, result, expected_solids, inside_witnesses, False)[1]


def union_positive_solids(named_inputs, *, expected_solids=None, inside_witnesses=None):
    """Return ``(shape, report)`` after serial native union with preservation checks.

    Mapping order determines union order. Each step checks previous material and
    the new operand by full native subtraction, then the final result is checked
    against every unchanged original input. Disconnected solids are retained;
    expected_solids is checked only on the final complete result. No clean,
    healing, largest-solid selection or fuzzy/tolerance increase is performed.
    Runtime native subtraction is not an exact-real mathematical containment proof.
    Any native warning leaves evidence unresolved and raises; it is not retried,
    waived or interpreted as proof that the underlying geometry is wrong.
    """
    return _run(named_inputs, None, expected_solids, inside_witnesses, True)
