"""Explicit native-only meshing retry for valid narrow faces.

Only triangulation caches are updated. Geometry, topology, tolerances, and
source files are never repaired or simplified. Unsupported failures reject.
"""
import hashlib
import io
import math


def _geometry_digest(shape):
    from OCP.BRepTools import BRepTools
    from OCP.BRepBuilderAPI import BRepBuilderAPI_Copy
    from OCP.TopTools import TopTools_FormatVersion_VERSION_3
    # UpdateFace correctly invalidates the original face's Checked cache.
    # Normalize that cache only on an independent topology snapshot; never
    # turn source validation flags on or mask changed geometric fields.
    snapshot=BRepBuilderAPI_Copy(shape,False,False).Shape()
    snapshot.Checked(False)
    stream=io.BytesIO()
    BRepTools.Write_s(snapshot,stream,False,False,TopTools_FormatVersion_VERSION_3)
    return hashlib.sha256(stream.getvalue()).hexdigest()


def recover_narrow_face_meshes(shape,linear,angular):
    import cadquery as cq
    from OCP.BRep import BRep_Tool,BRep_Builder
    from OCP.BRepTools import BRepTools
    from OCP.BRepMesh import BRepMesh_IncrementalMesh
    from OCP.BRepBuilderAPI import BRepBuilderAPI_Copy
    from OCP.TopLoc import TopLoc_Location
    rows=[]
    for face_id,face in enumerate(shape.Faces()):
        old=BRep_Tool.Triangulation_s(face.wrapped,TopLoc_Location())
        if old is not None and old.NbTriangles()>0 and old.HasUVNodes():
            continue
        if not face.isValid():
            raise ValueError(f'face {face_id}: invalid native face cannot use narrow-face mesh recovery')
        area=float(face.Area());perimeter=sum(float(e.Length()) for e in face.Edges())
        if not math.isfinite(area) or not math.isfinite(perimeter) or area<=0 or perimeter<=0:
            raise ValueError(f'face {face_id}: nonpositive/nonfinite face size cannot use mesh recovery')
        width=2*area/perimeter
        finer=width/4
        if not math.isfinite(finer) or not 0<finer<linear:
            raise ValueError(f'face {face_id}: unmeshed face is not narrow enough for this explicit recovery')
        before=_geometry_digest(face.wrapped)
        copy=cq.Shape.cast(BRepBuilderAPI_Copy(face.wrapped,False,False).Shape())
        BRepTools.Clean_s(copy.wrapped)
        mesher=BRepMesh_IncrementalMesh(copy.wrapped,finer,False,angular,False)
        location=TopLoc_Location();mesh=BRep_Tool.Triangulation_s(copy.wrapped,location)
        if not mesher.IsDone() or mesher.GetStatusFlags()!=0 or mesh is None or mesh.NbTriangles()==0 or not mesh.HasUVNodes():
            raise ValueError(f'face {face_id}: single geometry-sized native meshing retry failed; face retained, verification rejected')
        source_location=face.wrapped.Location().Transformation();copy_location=location.Transformation()
        if any(source_location.Value(i,j)!=copy_location.Value(i,j) for i in range(1,4) for j in range(1,5)):
            raise ValueError(f'face {face_id}: mesh location does not match original native face')
        if tuple(BRepTools.UVBounds_s(copy.wrapped))!=tuple(BRepTools.UVBounds_s(face.wrapped)):
            raise ValueError(f'face {face_id}: copied trim bounds changed during native retry')
        # Only auxiliary triangulation is transferred. The original face still
        # owns its unchanged surface, trim edges, pcurves, vertices and tolerances.
        BRep_Builder().UpdateFace(face.wrapped,mesh,True)
        after=_geometry_digest(face.wrapped)
        if after!=before:
            raise ValueError(f'face {face_id}: native geometry/topology changed; recovery rejected')
        rows.append({'face_index':face_id,'area_mm2':area,'perimeter_mm':perimeter,
                     'width_proxy_2A_over_P_mm':width,'nominal_deflection_mm':linear,
                     'retry_deflection_mm':finer,'angle_radians':angular,'parallel':False,
                     'native_status':mesher.GetStatusFlags(),'nodes':mesh.NbNodes(),
                     'triangles':mesh.NbTriangles(),'geometry_without_mesh_sha256':before,
                     'geometry_topology_tolerances_unchanged':True})
    return rows
