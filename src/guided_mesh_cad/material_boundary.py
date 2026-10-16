"""Explicit extraction of an oriented solid's material boundary.

This is not shape repair, feature deletion, or an automatic validator fallback.
Only whole open shells made exclusively of INTERNAL faces may be omitted.
All closed, oriented boundary shells and their actual located faces are reused.
Source-shape comparison remains a separate required caller responsibility.
"""

import hashlib
import io
import math

from .cad import (PatchTopologyError, _computed_shell_closure,
                  _require_native_integrity, require_solid,
                  shape_check_report, volume)


def _fingerprint(shape):
    from OCP.BRepTools import BRepTools
    stream = io.BytesIO()
    BRepTools.Write_s(shape.wrapped, stream)
    return hashlib.sha256(stream.getvalue()).hexdigest()


def extract_material_boundary(shape, *, evidence):
    """Return ``(solid, report)`` after explicit INTERNAL-sheet extraction.

    The input must already be one valid, positive-volume native solid with no
    unowned topology and no native tolerance above 0.01 mm. A closed shell may
    contain only FORWARD/REVERSED faces. An open shell may contain only INTERNAL
    faces; mixed or externally oriented open shells are rejected. Multiple
    solids, free sheets, invalid inputs and excessive tolerances are rejected.

    Each omitted face must also pass a full native face-minus-material test on
    independent copies, with its query orientation set FORWARD and no added
    fuzzy tolerance. Labels and corner samples alone are never containment
    evidence. Kernel failure or any residual topology is rejected. Containment
    is evaluated at the original native precisions, not as an exact-real set
    theorem. This operation preserves all oriented boundary face and shell
    occurrences by IsEqual identity, including location and orientation. It
    does not certify feature completeness or source distances. Keep input/report
    and perform the original complete source/native checks on the output.
    Default readers and ``require_solid`` remain strict and never call this.
    """
    import cadquery as cq
    from OCP.BRep import BRep_Builder
    from OCP.BRepBuilderAPI import BRepBuilderAPI_Copy
    from OCP.BRepAlgoAPI import BRepAlgoAPI_Cut
    from OCP.TopoDS import TopoDS_Solid, TopoDS_Iterator
    from OCP.TopTools import TopTools_ListOfShape
    from OCP.TopAbs import (TopAbs_SHELL, TopAbs_FACE, TopAbs_FORWARD, TopAbs_REVERSED,
                           TopAbs_INTERNAL)

    if not isinstance(evidence, str) or not evidence.strip():
        raise ValueError('explicit non-empty material-boundary evidence is required')
    before_hash = _fingerprint(shape)
    report = {'operation': 'explicit_material_boundary_extraction',
              'evidence': evidence, 'input_brep_sha256': before_hash,
              'default_validation_policy_changed': False,
              'source_conformity_accepted': False}

    def reject(message):
        report['input_unchanged'] = _fingerprint(shape) == before_hash
        raise PatchTopologyError(message, report)

    check = shape_check_report(shape)
    report['input_check'] = check
    if not check['valid'] or check['solids'] != 1:
        reject('material extraction requires exactly one already-valid native solid')
    try:
        _require_native_integrity(check)
    except PatchTopologyError as exc:
        reject(str(exc))
    if max(check['native_tolerance_max_mm'].values()) > 0.01:
        reject('material extraction cannot reduce excessive native tolerances')
    original = shape.Solids()[0]
    original_volume, original_error = volume(original)
    if not math.isfinite(original_volume) or original_volume <= 1e-12:
        reject('material extraction requires a positive-volume oriented input')

    def face_occurrences(shell):
        faces = []
        iterator = TopoDS_Iterator(shell.wrapped, True, True)
        while iterator.More():
            child = iterator.Value()
            if child.ShapeType() != TopAbs_FACE:
                reject('shell contains a direct non-face child')
            faces.append(cq.Shape.cast(child))
            iterator.Next()
        return faces

    def same_occurrences(expected, actual):
        if len(expected) != len(actual):
            return False
        unmatched = list(actual)
        for item in expected:
            match = next((i for i, other in enumerate(unmatched)
                          if item.wrapped.IsEqual(other.wrapped)), None)
            if match is None:
                return False
            unmatched.pop(match)
        return not unmatched

    kept, removed, boundary_faces, internal_faces = [], [], [], []
    children = TopoDS_Iterator(original.wrapped, True, True)
    shell_index = 0
    while children.More():
        child = children.Value()
        if child.ShapeType() != TopAbs_SHELL:
            reject('solid contains a direct non-shell child')
        shell = cq.Shape.cast(child)
        faces = face_occurrences(shell)
        if not faces:
            reject('empty shell has no verified material-boundary role')
        closure = _computed_shell_closure(shell)
        orientations = [face.wrapped.Orientation() for face in faces]
        record = {'input_shell_index': shell_index, 'closure': closure,
                  'face_count': len(faces),
                  'face_orientations': [str(value) for value in orientations]}
        if closure['closed']:
            if any(value not in (TopAbs_FORWARD, TopAbs_REVERSED)
                   for value in orientations):
                reject('closed boundary shell contains a non-boundary face orientation')
            kept.append(shell)
            boundary_faces.extend(faces)
        else:
            if any(value != TopAbs_INTERNAL for value in orientations):
                reject('open shell is not exclusively INTERNAL; physical faces cannot be removed')
            record['face_areas_mm2'] = [float(face.Area()) for face in faces]
            removed.append(record)
            internal_faces.extend(faces)
        shell_index += 1
        children.Next()
    report['retained_closed_shell_count'] = len(kept)
    report['removed_internal_open_shells'] = removed
    if not kept:
        reject('no complete oriented material boundary is available')
    if not removed:
        require_solid(shape)
        report.update({'status': 'unchanged_already_material_boundary',
                       'input_unchanged': _fingerprint(shape) == before_hash,
                       'output_brep_sha256': before_hash,
                       'output_check': check})
        if not report['input_unchanged']:
            reject('input changed during material-boundary inspection')
        return shape, report

    # A new top-level solid only; no sewing, shape healing, tolerance changes
    # or largest-solid selection. Later Booleans are read-only containment
    # checks on copies, never geometric repairs. Shells already carry the
    # cumulative original orientation/location from the explicit iterator.
    native = TopoDS_Solid()
    builder = BRep_Builder()
    builder.MakeSolid(native)
    for shell in kept:
        builder.Add(native, shell.wrapped)
    result = cq.Solid(native)
    output_shells = []
    iterator = TopoDS_Iterator(result.wrapped, True, True)
    while iterator.More():
        output_shells.append(cq.Shape.cast(iterator.Value()))
        iterator.Next()
    output_faces = [face for shell in output_shells for face in face_occurrences(shell)]
    if not same_occurrences(kept, output_shells):
        reject('complete located oriented closed-shell occurrences were not preserved')
    if not same_occurrences(boundary_faces, output_faces):
        reject('complete located oriented boundary face identity was not preserved')
    require_solid(result)
    result_hash_before_proofs = _fingerprint(result)
    containment = []
    report['internal_face_containment'] = containment
    for face_index, face in enumerate(internal_faces):
        query = BRepBuilderAPI_Copy(face.wrapped, True, False).Shape()
        query.Orientation(TopAbs_FORWARD)
        material_copy = BRepBuilderAPI_Copy(result.wrapped, True, False).Shape()
        arguments, tools = TopTools_ListOfShape(), TopTools_ListOfShape()
        arguments.Append(query)
        tools.Append(material_copy)
        operation = BRepAlgoAPI_Cut()
        operation.SetArguments(arguments)
        operation.SetTools(tools)
        operation.SetRunParallel(False)
        operation.SetNonDestructive(True)
        operation.SetFuzzyValue(0.0)
        operation.Build()
        row = {'internal_face_occurrence': face_index,
               'query_orientation_on_copy': 'FORWARD',
               'native_full_face_minus_material': True,
               'fuzzy_tolerance_mm': 0.0,
               'native_operation_done': bool(operation.IsDone())}
        containment.append(row)
        if not operation.IsDone():
            reject('native whole-face material containment is unresolved')
        outside_native = operation.Shape()
        if outside_native.IsNull():
            counts = {'faces': 0, 'edges': 0, 'vertices': 0, 'solids': 0}
        else:
            outside = cq.Shape.cast(outside_native)
            counts = {'faces': len(outside.Faces()), 'edges': len(outside.Edges()),
                      'vertices': len(outside.Vertices()), 'solids': len(outside.Solids())}
        row['outside_topology'] = counts
        if any(counts.values()):
            reject('INTERNAL label is not containment proof: face has residual topology outside material')
    if _fingerprint(result) != result_hash_before_proofs:
        reject('retained boundary changed during containment proofs')
    output_volume, output_error = volume(result)
    if not all(math.isfinite(v) for v in (original_error, output_error, output_volume)):
        reject('native volume diagnostic is non-finite')
    # The exact material-preservation argument is unchanged oriented faces.
    # This additional integration comparison uses the kernel's own reported
    # numerical error estimates, not a new physical or source tolerance.
    numerical_budget = (abs(original_volume * original_error) +
                        abs(output_volume * output_error) +
                        128 * math.ulp(max(abs(original_volume), abs(output_volume), 1.0)))
    if abs(output_volume - original_volume) > numerical_budget:
        reject('native volume diagnostics disagree after oriented boundary extraction')
    if _fingerprint(shape) != before_hash:
        reject('input changed during material-boundary extraction')
    report.update({'status': 'material_boundary_extracted', 'input_unchanged': True,
                   'retained_boundary_face_count': len(boundary_faces),
                   'all_closed_shell_occurrences_IsEqual_preserved': True,
                   'all_oriented_faces_IsEqual_preserved': True,
                   'input_volume_mm3': original_volume,
                   'output_volume_mm3': output_volume,
                   'volume_numerical_error_budget_mm3': numerical_budget,
                   'volume_check_scope': 'Diagnostic; unchanged oriented native faces are the preservation evidence.',
                   'output_check': shape_check_report(result),
                   'output_brep_sha256': _fingerprint(result)})
    return result, report
