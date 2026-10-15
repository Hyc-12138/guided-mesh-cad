"""Finite comparison against actual trimmed native boundaries.

Inspection triangles select face hints and UV sample locations. Their 3-D
coordinates never replace the CAD boundary in this metric. Legacy mesh-based
validation remains separate and unchanged.
"""
from pathlib import Path
import time
import numpy as np
from .mesh import load_mesh, polydata, sample_triangles, mesh_distances, sha256
from .pipeline import dump, positive


def _native_mesh_uv(shape, linear=.001, angular=.05):
    from OCP.BRepTools import BRepTools
    from OCP.BRepMesh import BRepMesh_IncrementalMesh
    from OCP.BRep import BRep_Tool
    from OCP.TopLoc import TopLoc_Location
    from OCP.TopAbs import TopAbs_REVERSED
    BRepTools.Clean_s(shape.wrapped)
    mesher = BRepMesh_IncrementalMesh(shape.wrapped, linear, False, angular, True)
    if not mesher.IsDone():
        raise ValueError('native absolute-deflection mesher failed')
    vertices, triangles, uv, node_faces, triangle_faces = [], [], [], [], []
    offset = 0
    for face_id, face in enumerate(shape.Faces()):
        location = TopLoc_Location()
        mesh = BRep_Tool.Triangulation_s(face.wrapped, location)
        if mesh is None or mesh.NbTriangles() == 0 or not mesh.HasUVNodes():
            raise ValueError('a complete native face has no UV triangulation; it cannot be omitted')
        v = np.empty((mesh.NbNodes(), 3)); u = np.empty((len(v), 2))
        f = np.empty((mesh.NbTriangles(), 3), dtype=np.int64)
        for i in range(1, len(v)+1):
            point = mesh.Node(i).Transformed(location.Transformation())
            parameter = mesh.UVNode(i)
            v[i-1] = [point.X(), point.Y(), point.Z()]
            u[i-1] = [parameter.X(), parameter.Y()]
        for i in range(1, len(f)+1):
            a, b, c = mesh.Triangle(i).Get()
            f[i-1] = [a-1, b-1, c-1]
        if face.wrapped.Orientation() == TopAbs_REVERSED:
            f = f[:, [0, 2, 1]]
        vertices.append(v); triangles.append(f+offset); uv.append(u)
        node_faces.append(np.full(len(v), face_id, dtype=np.int64))
        triangle_faces.append(np.full(len(f), face_id, dtype=np.int64))
        offset += len(v)
    return (np.vstack(vertices), np.vstack(triangles), np.vstack(uv),
            np.concatenate(node_faces), np.concatenate(triangle_faces))


def _owned_solid(shape):
    from .cad import require_solid
    require_solid(shape)


def _stats(values, limit):
    return {'count':len(values), 'max_mm':float(values.max()),
            'p95_mm':float(np.percentile(values,95)),
            'rms_mm':float(np.sqrt(np.mean(values*values))),
            'over_tolerance':int(np.count_nonzero(values>limit))}


def _worst(points, distances, projected=None):
    count = min(8, len(distances))
    ids = np.argpartition(distances, len(distances)-count)[-count:]
    ids = ids[np.argsort(distances[ids])[::-1]]
    rows = []
    for i in ids:
        row = {'sample_index':int(i),'point_mm':points[i].tolist(),
               'distance_mm':float(distances[i])}
        if projected is not None:
            row['actual_native_boundary_point_mm'] = projected[i].tolist()
        rows.append(row)
    return rows


def _native_uv_triangle_hint(vertices,triangles,uv_nodes,cell_index,closest):
    """A parameter seed only. It is never accepted as a 3-D CAD point."""
    triangle=triangles[cell_index];a,b,c=vertices[triangle]
    e0,e1,q=b-a,c-a,np.asarray(closest,dtype=float)-a
    d00,d01,d11=float(e0@e0),float(e0@e1),float(e1@e1)
    denominator=d00*d11-d01*d01
    if not np.isfinite(denominator) or denominator<=0:return None
    s=(d11*float(q@e0)-d01*float(q@e1))/denominator
    t=(d00*float(q@e1)-d01*float(q@e0))/denominator
    weights=np.array([1-s-t,s,t])
    if not np.isfinite(weights).all() or weights.min()<-1e-5 or weights.max()>1+1e-5:return None
    uv=weights@uv_nodes[triangle]
    return uv if np.isfinite(uv).all() else None


def validate_native_boundary(source_path, step_path, report_path, *,
                             tolerance_mm=.01, samples=100000, seed=73191,
                             diagnostics_dir=None, progress=None,_uv_witness=False,
                             _low_memory=False,_mesh_parallel=True,_resource_registry=None):
    """Compare complete source with actual trimmed native geometry.

    Both inputs must use mm. Every source vertex/triangle centre and all native
    face mesh-UV nodes/triangle centres participate, plus random samples in each
    direction and native edge/vertex witnesses. Surface points are classified
    against the actual face trims; outside UV samples are projected onto that
    same trimmed face, counted and recorded, never silently dropped. Source
    projections to valid face points are conservative upper bounds; unresolved
    or rejecting hints use the complete native boundary.

    This remains a finite-sample test, not a global Hausdorff certificate.
    It does not rewrite legacy mesh-based results or relax their metrics.
    """
    import cadquery as cq
    import vtk
    from OCP.BRep import BRep_Tool
    from OCP.BRepTools import BRepTools
    from OCP.GeomAPI import GeomAPI_ProjectPointOnSurf
    from OCP.BRepClass import BRepClass_FaceClassifier
    from OCP.BRepExtrema import BRepExtrema_DistShapeShape
    from OCP.TopAbs import TopAbs_IN, TopAbs_ON
    from OCP.gp import gp_Pnt, gp_Pnt2d
    from .cad import read_step_unchecked, shape_check_report
    from . import __version__
    if not isinstance(_uv_witness,bool):raise ValueError('internal UV witness strategy must be boolean')
    if not isinstance(_low_memory,bool) or not isinstance(_mesh_parallel,bool):
        raise ValueError('memory and mesh strategies must be boolean')
    if _low_memory and not _uv_witness:
        raise ValueError('mapped memory strategy requires its explicit UV witness metric')
    if _low_memory and diagnostics_dir is None:
        raise ValueError('low-memory validation requires a fresh diagnostics directory')
    limit = positive(tolerance_mm, 'verification tolerance')
    if not isinstance(samples,int) or isinstance(samples,bool) or not 100<=samples<=1000000:
        raise ValueError('samples must be an integer in [100,1000000]')
    report_path = Path(report_path)
    if report_path.exists():
        raise ValueError('report path already exists; preserve frozen prior evidence')
    diagnostic_out = Path(diagnostics_dir) if diagnostics_dir is not None else None
    if diagnostic_out is not None:
        if diagnostic_out.exists() and (not diagnostic_out.is_dir() or any(diagnostic_out.iterdir())):
            raise ValueError('diagnostics directory must be empty')
        diagnostic_out.mkdir(parents=True,exist_ok=True)
    timings = {}; start = time.perf_counter()
    store=None;support_reconstruction_count=0;support_coordinate_residual_max=0.
    if _low_memory:
        from .memory_validation import NativeArrayStore,native_mesh_uv_mapped,mapped_native_sampling,mapped_mesh_hints,mesh_distances_into,owned_extrema_point
        store=NativeArrayStore(diagnostic_out/'native_arrays')
        if _resource_registry is not None:_resource_registry.append(store)
    def actual_extrema_point(extrema,owner):
        nonlocal support_reconstruction_count,support_coordinate_residual_max
        point=owned_extrema_point(extrema,owner)
        cached=extrema.PointOnShape1(1)
        support_coordinate_residual_max=max(support_coordinate_residual_max,float(point.Distance(cached)))
        support_reconstruction_count+=1
        return point
    sv,sf = load_mesh(source_path); shape = read_step_unchecked(step_path)
    _owned_solid(shape); faces = shape.Faces()
    face_edges = [cq.Compound.makeCompound(face.Edges()) for face in faces]
    boundary = cq.Compound.makeCompound(faces)
    timings['read_native_checks'] = time.perf_counter()-start
    stage = time.perf_counter()
    if _low_memory:
        mv,mf,mesh_uv,node_faces,triangle_faces,face_mesh_counts=native_mesh_uv_mapped(
            shape,store,limit/10,.05,_mesh_parallel,progress)
    else:
        mv,mf,mesh_uv,node_faces,triangle_faces = _native_mesh_uv(shape, limit/10, .05)
    timings['native_uv_mesh'] = time.perf_counter()-stage
    surfaces = [BRep_Tool.Surface_s(face.wrapped) for face in faces]
    projectors = []
    for face,surface in zip(faces,surfaces):
        projector = GeomAPI_ProjectPointOnSurf()
        projector.Init(surface,*BRepTools.UVBounds_s(face.wrapped),1e-9)
        projectors.append(projector)
    classifier = BRepClass_FaceClassifier()
    locator=None
    if not _low_memory:
        locator = vtk.vtkStaticCellLocator()
        locator.SetDataSet(polydata(mv,mf)); locator.BuildLocator()
    closest=[0.,0.,0.]; cell=vtk.reference(0); sub=vtk.reference(0); distance=vtk.reference(0.)
    source_points = np.vstack([sv,sv[sf].mean(axis=1),sample_triangles(sv,sf,samples,seed)])
    if _low_memory:
        hint_closest,hint_cells,hint_squared_distances,hint_face_queries=mapped_mesh_hints(
            source_points,mv,mf,face_mesh_counts,progress)
    forward = np.empty(len(source_points)); projected = np.empty_like(source_points)
    source_mesh_hint = np.empty(len(source_points)); fallback = 0
    witness_used=np.zeros(len(source_points),dtype=bool) if _uv_witness else None
    witness_uv=np.full((len(source_points),2),np.nan) if _uv_witness else None
    witness_faces=np.full(len(source_points),-1,dtype=np.int64) if _uv_witness else None
    witness_rejections={'unusable_uv_hint':0,'outside_or_on_trim':0,'actual_point_over_limit':0,'native_evaluation_error':0}
    stage = time.perf_counter()
    for i,xyz in enumerate(source_points):
        if _low_memory:
            closest=hint_closest[i];cell=int(hint_cells[i]);distance=float(hint_squared_distances[i])
        else:
            locator.FindClosestPoint(xyz,closest,cell,sub,distance)
        face_id = int(triangle_faces[int(cell)])
        accepted = False
        if _uv_witness:
            hint_failed=False
            try:hint=_native_uv_triangle_hint(mv,mf,mesh_uv,int(cell),closest)
            except Exception:
                hint=None;hint_failed=True;witness_rejections['native_evaluation_error']+=1
            if hint is None:
                if not hint_failed:witness_rejections['unusable_uv_hint']+=1
            else:
                try:
                    u,v=map(float,hint)
                    classifier.Perform(faces[face_id].wrapped,gp_Pnt2d(u,v),1e-9,True,1e-7)
                    # ON points still use the original actual trim-edge path.
                    if classifier.State()!=TopAbs_IN:witness_rejections['outside_or_on_trim']+=1
                    else:
                        candidate=surfaces[face_id].Value(u,v)
                        candidate_xyz=np.array([candidate.X(),candidate.Y(),candidate.Z()])
                        measured=float(np.linalg.norm(candidate_xyz-xyz))
                        if np.isfinite(candidate_xyz).all() and np.isfinite(measured) and measured<=limit:
                            point=candidate;accepted=True;witness_used[i]=True
                            witness_uv[i]=hint;witness_faces[i]=face_id
                        else:witness_rejections['actual_point_over_limit']+=1
                except Exception:
                    # A failed optional seed cannot accept or discard a point.
                    # The original complete projection path remains mandatory.
                    witness_rejections['native_evaluation_error']+=1
                    accepted=False;witness_used[i]=False;witness_uv[i]=np.nan;witness_faces[i]=-1
        if not accepted:
            projector = projectors[face_id]
            projector.Perform(gp_Pnt(*map(float,xyz)))
            if projector.IsDone() and projector.NbPoints()>0:
                u,v = projector.LowerDistanceParameters()
                classifier.Perform(faces[face_id].wrapped,gp_Pnt2d(u,v),1e-9,True,1e-7)
                if classifier.State() == TopAbs_IN:
                    if _low_memory:
                        point=surfaces[face_id].Value(float(u),float(v))
                        measured=float(np.linalg.norm(np.array([point.X(),point.Y(),point.Z()])-xyz))
                    else:
                        point = projector.NearestPoint(); measured = float(projector.LowerDistance())
                    accepted = True
                elif classifier.State() == TopAbs_ON:
                    extrema = BRepExtrema_DistShapeShape(
                        face_edges[face_id].wrapped,cq.Vertex.makeVertex(*map(float,xyz)).wrapped)
                    if extrema.IsDone() and extrema.NbSolution()>0:
                        if _low_memory:
                            point=actual_extrema_point(extrema,face_edges[face_id])
                            measured=float(np.linalg.norm(np.array([point.X(),point.Y(),point.Z()])-xyz))
                        else:
                            point=extrema.PointOnShape1(1); measured=float(extrema.Value())
                        accepted=True
        if not accepted or measured>limit:
            extrema = BRepExtrema_DistShapeShape(
                boundary.wrapped,cq.Vertex.makeVertex(*map(float,xyz)).wrapped)
            if not extrema.IsDone() or extrema.NbSolution()<1:
                raise ValueError('complete trimmed native boundary projection failed')
            if _low_memory:
                point=actual_extrema_point(extrema,boundary)
                measured=float(np.linalg.norm(np.array([point.X(),point.Y(),point.Z()])-xyz))
            else:
                point = extrema.PointOnShape1(1); measured = float(extrema.Value())
            fallback+=1
        forward[i] = measured
        projected[i] = [point.X(),point.Y(),point.Z()]
        source_mesh_hint[i] = float(distance)**.5
        if progress is not None and (i+1)%20000==0:
            progress({'stage':'source_actual_boundary','done':i+1,'total':len(source_points),
                      'elapsed_seconds':time.perf_counter()-start})
    timings['source_actual_boundary'] = time.perf_counter()-stage
    if _low_memory:
        # The complete locator was used for every source hint. Release its
        # copied arrays before constructing the dense native sample storage.
        import gc
        del hint_closest,hint_cells,hint_squared_distances;gc.collect()
    # Use every native mesh-UV node and every triangle centre, not just a UV
    # bounding rectangle. Triangles additionally provide finite sampling density
    # and area weights; all actual 3-D samples are evaluated on the CAD surface.
    if _low_memory:
        uv,random_faces,choices,bary=mapped_native_sampling(mv,mf,mesh_uv,triangle_faces,samples,seed,store)
        edge_vertex_count=len(shape.Edges())*33+len(shape.Vertices())
        native_points=store.allocate('native_actual_points',(len(uv)+edge_vertex_count,3))
        node_offsets=np.r_[0,np.cumsum([n for n,t in face_mesh_counts])]
        triangle_offsets=np.r_[0,np.cumsum([t for n,t in face_mesh_counts])]
    else:
        rng = np.random.default_rng(seed+1)
        areas = np.linalg.norm(np.cross(mv[mf[:,1]]-mv[mf[:,0]],mv[mf[:,2]]-mv[mf[:,0]]),axis=1)
        if not np.isfinite(areas).all() or areas.sum()<=0:
            raise ValueError('native locator has no finite sampling area')
        choices = rng.choice(len(mf),samples,p=areas/areas.sum())
        bary = rng.random((samples,2)); flip=bary.sum(axis=1)>1; bary[flip]=1-bary[flip]
        random_uv = ((1-bary.sum(axis=1))[:,None]*mesh_uv[mf[choices,0]]+
                     bary[:,:1]*mesh_uv[mf[choices,1]]+bary[:,1:]*mesh_uv[mf[choices,2]])
        uv = np.vstack([mesh_uv,mesh_uv[mf].mean(axis=1),random_uv])
        uv_faces = np.r_[node_faces,triangle_faces,triangle_faces[choices]]
        native_points = np.empty((len(uv),3))
    face_reports=[]; trim_projections=0; trim_projection_max=0.; outside_count=0; on_count=0
    stage = time.perf_counter()
    for face_id,(face,surface) in enumerate(zip(faces,surfaces)):
        if _low_memory:
            ids=np.r_[np.arange(node_offsets[face_id],node_offsets[face_id+1]),
                      len(mesh_uv)+np.arange(triangle_offsets[face_id],triangle_offsets[face_id+1]),
                      len(mesh_uv)+len(mf)+np.flatnonzero(random_faces==face_id)]
        else:
            ids = np.flatnonzero(uv_faces==face_id)
        if not len(ids):
            raise ValueError('a complete native face has no actual-boundary samples')
        corrected=0; outside=0; on_boundary=0
        for i in ids:
            u,v=map(float,uv[i]); point=surface.Value(u,v)
            classifier.Perform(face.wrapped,gp_Pnt2d(u,v),1e-9,True,1e-7)
            state=classifier.State()
            if state != TopAbs_IN:
                # A UV point outside a trimmed domain lies on its supporting
                # surface, not on material. Map it to a real native trim edge;
                # ON points use the actual edge geometry as well. No samples
                # are dropped and no tolerance band is treated as material.
                extrema=BRepExtrema_DistShapeShape(
                    face_edges[face_id].wrapped,cq.Vertex.makeVertex(point.X(),point.Y(),point.Z()).wrapped)
                if not extrema.IsDone() or extrema.NbSolution()<1:
                    raise ValueError('outside UV sample could not be projected to its actual trimmed face')
                point=actual_extrema_point(extrema,face_edges[face_id]) if _low_memory else extrema.PointOnShape1(1)
                corrected+=1
                trim_projection_max=max(trim_projection_max,float(extrema.Value()))
                if state==TopAbs_ON:on_boundary+=1
                else:outside+=1
            native_points[i]=[point.X(),point.Y(),point.Z()]
        trim_projections+=corrected; outside_count+=outside; on_count+=on_boundary
        face_reports.append({'face_index':face_id,'surface_type':face.geomType(),
                             'mesh_uv_nodes':int(face_mesh_counts[face_id][0]) if _low_memory else int(np.count_nonzero(node_faces==face_id)),
                             'mesh_triangles':int(face_mesh_counts[face_id][1]) if _low_memory else int(np.count_nonzero(triangle_faces==face_id)),
                             'actual_surface_samples':len(ids),'trim_boundary_projections':corrected,
                             'outside_uv_projections':outside,'on_boundary_uv_projections':on_boundary})
        if progress is not None:
            progress({'stage':'native_actual_boundary','face':face_id,'faces':len(faces),
                      'samples':len(ids),'elapsed_seconds':time.perf_counter()-start})
    edge_points=[]
    for edge in shape.Edges():
        for value in np.linspace(0,1,33):
            point=edge.positionAt(float(value));edge_points.append([point.x,point.y,point.z])
    for vertex in shape.Vertices():
        point=vertex.Center();edge_points.append([point.x,point.y,point.z])
    if _low_memory:
        if len(edge_points)!=edge_vertex_count:raise ValueError('native edge/vertex coverage changed during verification')
        native_points[len(uv):]=np.asarray(edge_points,dtype=float).reshape(-1,3);native_points.flush()
    else:
        native_points=np.vstack([native_points,np.asarray(edge_points,dtype=float).reshape(-1,3)])
    timings['native_actual_surface_evaluation']=time.perf_counter()-stage
    stage=time.perf_counter()
    if _low_memory:
        reverse=store.allocate('native_to_source_distances',(len(native_points),))
        mesh_distances_into(native_points,sv,sf,reverse,progress)
    else:
        reverse=mesh_distances(native_points,sv,sf)
    timings['native_actual_to_source']=time.perf_counter()-stage
    diagnostics=None
    if _low_memory:
        evidence=diagnostic_out/'source_actual_boundary_samples.npz'
        np.savez_compressed(evidence,source_sample_points=source_points,
                            source_projection_points=projected,source_to_native_distances=forward,
                            source_uv_witness_used=witness_used,source_uv_witness_parameters=witness_uv,
                            source_uv_witness_face_ids=witness_faces)
        diagnostics={'format':'native_memory_mapped_arrays.v1',
                     'native_uv_tessellation_arrays':store.manifest(['vertices','faces','uv_nodes','node_face_ids','triangle_face_ids']),
                     'native_actual_sample_arrays':store.manifest(['all_native_sample_uv','native_actual_points','native_to_source_distances']),
                     'source_actual_boundary_samples':str(evidence.resolve()),
                     'source_actual_boundary_samples_sha256':sha256(evidence)}
    elif diagnostic_out is not None:
        cache=diagnostic_out/'native_uv_tessellation.npz'
        np.savez_compressed(cache,vertices=mv,faces=mf,uv_nodes=mesh_uv,
                            node_face_ids=node_faces,triangle_face_ids=triangle_faces)
        evidence=diagnostic_out/'actual_boundary_samples.npz'
        witness_evidence=({'source_uv_witness_used':witness_used,
                           'source_uv_witness_parameters':witness_uv,
                           'source_uv_witness_face_ids':witness_faces} if _uv_witness else {})
        np.savez_compressed(evidence,source_sample_points=source_points,
                            source_projection_points=projected,source_to_native_distances=forward,
                            native_actual_points=native_points,native_to_source_distances=reverse,**witness_evidence)
        diagnostics={'native_uv_tessellation':str(cache.resolve()),'native_uv_tessellation_sha256':sha256(cache),
                     'actual_boundary_samples':str(evidence.resolve()),'actual_boundary_samples_sha256':sha256(evidence)}
    maximum=max(float(forward.max()),float(reverse.max()))
    result={'verification_method':('actual_trimmed_native_boundary_uv_witness_low_memory' if _low_memory else 'actual_trimmed_native_boundary_uv_witness') if _uv_witness else 'actual_trimmed_native_boundary','package_version':__version__,
            'passed':bool(maximum<=limit),'sampled_max_mm':maximum,'tolerance_mm':limit,
            'source_to_actual_native_boundary_upper_bounds':_stats(forward,limit),
            'actual_native_boundary_to_source':_stats(reverse,limit),
            'worst_source_samples':_worst(source_points,forward,projected),
            'worst_actual_native_samples':_worst(native_points,reverse),
            'source_complete_global_boundary_fallbacks':fallback,
            'source_vertices':len(sv),'source_triangle_centres':len(sf),
            'native_face_coverage':face_reports,'native_edges_sampled':len(shape.Edges()),
            'native_vertices_sampled':len(shape.Vertices()),'native_edge_vertex_points':len(edge_points),
            'trim_boundary_samples_projected_not_omitted':trim_projections,
            'outside_uv_samples_projected_not_omitted':outside_count,
            'on_boundary_uv_samples_projected_to_edges':on_count,
            'outside_uv_projection_max_mm':trim_projection_max,
            'diagnostic_source_to_locator_mesh_max_mm':float(source_mesh_hint.max()),
            'native_tessellation_requested_deflection_mm':limit/10,
            'native_tessellation_is_relative':False,'native_tessellation_requested_angle_radians':.05,
            'seed':seed,'random_samples_each_direction':samples,'source_sha256':sha256(source_path),
            'step_sha256':sha256(step_path),'native':shape_check_report(shape),
            'diagnostics':diagnostics,'timings_seconds':timings,'elapsed_seconds':time.perf_counter()-start,
            'source_completeness_and_manufacturing_intent_verified':False,
            'scope':'Complete supplied source vertices/triangle centres plus random samples compared with valid actual trimmed CAD face or edge points; all native faces sampled at every mesh-UV node/triangle centre plus random UV and native edges/vertices. Finite sampling only, not a global Hausdorff certificate. Input mesh may be partial or noisy; this metric does not establish source completeness or manufacturing intent. Inspection mesh coordinates are diagnostic and are never substituted for CAD boundary positions; prior mesh-based failures remain separate.'}
    if _uv_witness:
        result['source_uv_witness']={'accepted_actual_in_trim_points':int(witness_used.sum()),
            'original_projection_points':int((~witness_used).sum()),'hint_rejections':witness_rejections,
            'scope':'UV interpolation is only a seed. Accepted witnesses are actual native Surface.Value points strictly classified IN the real trimmed face and at measured distance <= the unchanged limit. ON/outside/over-limit/unusable hints run the original global surface and complete-boundary fallback. Source values are conservative actual-point distance upper bounds, not claimed globally nearest distances; prior metrics/reports are not rewritten.'}
    if _low_memory:
        result['execution_strategy']={'storage':'npy_memory_maps','mesh_parallel':_mesh_parallel,
            'whole_native_mesh':True,'sampling_array_arithmetic_chunk_size':65536,
            'source_hint_locator_used_complete_native_mesh':True,
            'source_hint_search':'all native mesh faces; per-face complete triangle locator with mesh-box lower-bound pruning',
            'source_hint_face_query_coverage':hint_face_queries,
            'all_native_UV_nodes_triangle_centres_random_edges_vertices_retained':True,
            'owned_support_coordinate_reconstructions':support_reconstruction_count,
            'cached_CPP_to_evaluated_owned_point_max_mm':support_coordinate_residual_max,
            'scope':'Storage/execution change with the same requested mesh precision and complete generated-mesh sample coverage. Returned CPP points are re-evaluated on owned native supports and all source distances measured to those actual points. Working arrays remain on disk as diagnostic evidence; external process memory guarding is still required.'}
    dump(report_path,result)
    if store is not None:store.close()
    return result


def validate_native_boundary_uv_witness(source_path,step_path,report_path,**kwargs):
    """Distinct conservative native-point witness metric; old API stays default."""
    if '_uv_witness' in kwargs:raise ValueError('UV witness strategy is fixed by this explicit entry point')
    return validate_native_boundary(source_path,step_path,report_path,_uv_witness=True,**kwargs)


def validate_native_boundary_low_memory(source_path,step_path,report_path,*,mesh_parallel=False,**kwargs):
    """Explicit complete UV-witness metric with mapped storage and serial mesh."""
    if any(k in kwargs for k in ('_uv_witness','_low_memory','_mesh_parallel','_resource_registry')):
        raise ValueError('strategies are fixed by this explicit entry point')
    resources=[]
    try:
        return validate_native_boundary(source_path,step_path,report_path,
                                        _uv_witness=True,_low_memory=True,
                                        _mesh_parallel=mesh_parallel,
                                        _resource_registry=resources,**kwargs)
    finally:
        for store in resources:store.close()
