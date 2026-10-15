"""Low-peak storage for the explicit native-boundary verification strategy.

This module changes storage/execution, not native precision or sample coverage.
The original in-memory verifier remains a separate unchanged default strategy.
"""
from pathlib import Path
import gc
import numpy as np


class NativeArrayStore:
    def __init__(self, path):
        self.path=Path(path)
        if self.path.exists():
            raise ValueError('native array workspace must be new')
        self.path.mkdir(parents=True)
        self.arrays={}

    def allocate(self,name,shape,dtype=float):
        if name in self.arrays or not name.replace('_','').isalnum():
            raise ValueError('array name must be new and alphanumeric')
        target=self.path/(name+'.npy')
        array=np.lib.format.open_memmap(target,mode='w+',dtype=dtype,shape=shape)
        self.arrays[name]=array
        return array

    def manifest(self,names):
        from .mesh import sha256
        rows={}
        for name in names:
            array=self.arrays[name];array.flush();path=self.path/(name+'.npy')
            rows[name]={'path':str(path.resolve()),'sha256':sha256(path),
                        'shape':list(array.shape),'dtype':array.dtype.str}
        return rows

    def close(self):
        """Release Windows mappings even if diagnostic callbacks retain views."""
        for array in self.arrays.values():
            mapping=getattr(array,'_mmap',None)
            if mapping is not None and not mapping.closed:
                try:array.flush()
                finally:mapping.close()

    def __del__(self):
        try:self.close()
        except Exception:pass


def native_mesh_uv_mapped(shape,store,linear=.001,angular=.05,parallel=False,progress=None):
    """Whole native mesher, serial by default; retain every face/node/triangle."""
    from OCP.BRepTools import BRepTools
    from OCP.BRepMesh import BRepMesh_IncrementalMesh
    from OCP.BRep import BRep_Tool
    from OCP.TopLoc import TopLoc_Location
    from OCP.TopAbs import TopAbs_REVERSED
    if not isinstance(parallel,bool):raise ValueError('mesh_parallel must be boolean')
    BRepTools.Clean_s(shape.wrapped)
    mesher=BRepMesh_IncrementalMesh(shape.wrapped,linear,False,angular,parallel)
    if not mesher.IsDone():raise ValueError('native absolute-deflection mesher failed')
    faces=shape.Faces();counts=[]
    for face in faces:
        mesh=BRep_Tool.Triangulation_s(face.wrapped,TopLoc_Location())
        if mesh is None or mesh.NbTriangles()==0 or not mesh.HasUVNodes():
            raise ValueError('a complete native face has no UV triangulation; it cannot be omitted')
        counts.append((mesh.NbNodes(),mesh.NbTriangles()))
    nodes=sum(n for n,t in counts);triangles=sum(t for n,t in counts)
    vertices=store.allocate('vertices',(nodes,3));uv=store.allocate('uv_nodes',(nodes,2))
    facets=store.allocate('faces',(triangles,3),np.int64)
    node_faces=store.allocate('node_face_ids',(nodes,),np.int64)
    triangle_faces=store.allocate('triangle_face_ids',(triangles,),np.int64)
    no=to=0
    for face_id,(face,(nv,nt)) in enumerate(zip(faces,counts)):
        location=TopLoc_Location();mesh=BRep_Tool.Triangulation_s(face.wrapped,location)
        trsf=location.Transformation();reversed_face=face.wrapped.Orientation()==TopAbs_REVERSED
        for i in range(1,nv+1):
            point=mesh.Node(i).Transformed(trsf);parameter=mesh.UVNode(i)
            vertices[no+i-1]=[point.X(),point.Y(),point.Z()]
            uv[no+i-1]=[parameter.X(),parameter.Y()]
        for i in range(1,nt+1):
            a,b,c=mesh.Triangle(i).Get()
            facets[to+i-1]=[no+a-1,no+c-1,no+b-1] if reversed_face else [no+a-1,no+b-1,no+c-1]
        node_faces[no:no+nv]=face_id;triangle_faces[to:to+nt]=face_id
        no+=nv;to+=nt
        if progress is not None:progress({'stage':'native_mesh_to_mapped_arrays','face':face_id,'faces':len(faces),'nodes':nv,'triangles':nt})
    for a in [vertices,uv,facets,node_faces,triangle_faces]:a.flush()
    # Retain triangulation while face and edge references are in use.
    # Geometry and cached mesh share a lifetime until validation exits.
    # Mapped arrays reduce Python memory use but do not bound native kernel
    # allocation; callers still need external resource monitoring.
    del mesh,mesher
    gc.collect()
    return vertices,facets,uv,node_faces,triangle_faces,counts


def mapped_native_sampling(mv,mf,mesh_uv,triangle_faces,samples,seed,store,chunk_size=65536):
    areas=store.allocate('triangle_double_areas',(len(mf),))
    for start in range(0,len(mf),chunk_size):
        ids=np.asarray(mf[start:start+chunk_size]);q=mv[ids]
        areas[start:start+len(ids)]=np.linalg.norm(np.cross(q[:,1]-q[:,0],q[:,2]-q[:,0]),axis=1)
    if not np.isfinite(areas).all() or areas.sum()<=0:
        raise ValueError('native locator has no finite sampling area')
    rng=np.random.default_rng(seed+1)
    choices=rng.choice(len(mf),samples,p=areas/areas.sum())
    bary=rng.random((samples,2));flip=bary.sum(axis=1)>1;bary[flip]=1-bary[flip]
    random_uv=((1-bary.sum(axis=1))[:,None]*mesh_uv[mf[choices,0]]+
               bary[:,:1]*mesh_uv[mf[choices,1]]+bary[:,1:]*mesh_uv[mf[choices,2]])
    uv=store.allocate('all_native_sample_uv',(len(mesh_uv)+len(mf)+samples,2))
    uv[:len(mesh_uv)]=mesh_uv
    for start in range(0,len(mf),chunk_size):
        ids=np.asarray(mf[start:start+chunk_size])
        uv[len(mesh_uv)+start:len(mesh_uv)+start+len(ids)]=mesh_uv[ids].mean(axis=1)
    uv[len(mesh_uv)+len(mf):]=random_uv
    return uv,np.asarray(triangle_faces[choices]),choices,bary


def mapped_mesh_hints(points,vertices,triangles,face_counts,progress=None):
    """Search all inspection faces with exact mesh-box lower-bound pruning.

    Only hints are computed here. Every accepted point is subsequently evaluated
    on the real CAD support and trim, or uses the complete-boundary fallback.
    """
    import vtk
    from .mesh import polydata
    best=np.full(len(points),np.inf);nearest=np.empty_like(points)
    cells=np.full(len(points),-1,dtype=np.int64);records=[];no=to=0
    for face_id,(nv,nt) in enumerate(face_counts):
        vv=vertices[no:no+nv];ff=np.asarray(triangles[to:to+nt])-no
        if not np.isfinite(vv).all():raise ValueError('nonfinite native mesh hint coordinates')
        lo=vv.min(axis=0);hi=vv.max(axis=0)
        delta=np.maximum(np.maximum(lo-points,points-hi),0.)
        lower=np.einsum('ij,ij->i',delta,delta)
        # The lower bound applies to this face's complete inspection triangles,
        # not to an assumed CAD surface box or a reduced sampling surrogate.
        ids=np.flatnonzero(lower<=best)
        if len(ids):
            locator=vtk.vtkStaticCellLocator();locator.SetDataSet(polydata(vv,ff));locator.BuildLocator()
            closest=[0.,0.,0.];cell=vtk.reference(0);sub=vtk.reference(0);distance=vtk.reference(0.)
            for i in ids:
                locator.FindClosestPoint(points[i],closest,cell,sub,distance)
                value=float(distance)
                if not np.isfinite(value) or value<0:raise ValueError('invalid native inspection distance')
                if value<best[i]:best[i]=value;nearest[i]=closest;cells[i]=to+int(cell)
            locator.FreeSearchStructure();locator.SetDataSet(None);del locator
        records.append({'face_index':face_id,'complete_triangles':nt,'source_hint_queries':len(ids)})
        no+=nv;to+=nt
        if progress is not None:progress({'stage':'native_face_hint_search','face':face_id,'faces':len(face_counts),'queries':len(ids)})
    if no!=len(vertices) or to!=len(triangles) or np.any(cells<0) or not np.isfinite(best).all():
        raise ValueError('complete native inspection hint coverage failed')
    return nearest,cells,best,records


def mesh_distances_into(points,vertices,faces,out,progress=None):
    """One original-source locator, no Python list of all native distances."""
    import vtk
    from .mesh import polydata
    locator=vtk.vtkStaticCellLocator();locator.SetDataSet(polydata(vertices,faces));locator.BuildLocator()
    closest=[0.,0.,0.];cell=vtk.reference(0);sub=vtk.reference(0);distance=vtk.reference(0.)
    for i,point in enumerate(points):
        locator.FindClosestPoint(point,closest,cell,sub,distance);out[i]=float(distance)**.5
        if progress is not None and (i+1)%250000==0:
            progress({'stage':'mapped_native_to_source','done':i+1,'total':len(points)})
    if hasattr(out,'flush'):out.flush()
    return out


def owned_extrema_point(extrema,owner,index=1):
    """Evaluate the returned owned native support, not a rounded CPP coordinate.

    This is an actual-point witness. It is not a certified global nearest point.
    """
    import math
    from OCP.BRep import BRep_Tool
    from OCP.BRepAdaptor import BRepAdaptor_Curve,BRepAdaptor_Surface
    from OCP.BRepClass import BRepClass_FaceClassifier
    from OCP.TopAbs import TopAbs_VERTEX,TopAbs_EDGE,TopAbs_FACE,TopAbs_IN
    from OCP.TopExp import TopExp
    from OCP.TopTools import TopTools_IndexedMapOfShape
    from OCP.TopoDS import TopoDS
    from OCP.gp import gp_Pnt2d
    support=extrema.SupportOnShape1(index);kind=support.ShapeType()
    if kind not in (TopAbs_VERTEX,TopAbs_EDGE,TopAbs_FACE):
        raise ValueError('projection has no evaluable native support')
    identities=TopTools_IndexedMapOfShape();TopExp.MapShapes_s(owner.wrapped,kind,identities)
    if not identities.Contains(support):raise ValueError('projection support is not owned by requested native boundary')
    if kind==TopAbs_VERTEX:return BRep_Tool.Pnt_s(TopoDS.Vertex_s(support))
    if kind==TopAbs_EDGE:
        edge=TopoDS.Edge_s(support)
        if BRep_Tool.Degenerated_s(edge):
            vertices=TopTools_IndexedMapOfShape();TopExp.MapShapes_s(edge,TopAbs_VERTEX,vertices)
            if vertices.Extent()!=1:raise ValueError('degenerate support does not identify one native vertex')
            return BRep_Tool.Pnt_s(TopoDS.Vertex_s(vertices.FindKey(1)))
        (parameter,)=extrema.ParOnEdgeS1(index);curve=BRepAdaptor_Curve(edge)
        if not math.isfinite(parameter) or parameter<curve.FirstParameter() or parameter>curve.LastParameter():
            raise ValueError('actual edge projection parameter outside native trimmed interval')
        return curve.Value(parameter)
    face=TopoDS.Face_s(support);u,v=extrema.ParOnFaceS1(index)
    if not math.isfinite(u) or not math.isfinite(v):raise ValueError('nonfinite native face projection')
    classifier=BRepClass_FaceClassifier();classifier.Perform(face,gp_Pnt2d(u,v),1e-9,True,1e-7)
    if classifier.State()!=TopAbs_IN:
        # A kernel face support can lie within its numerical ON band. Return
        # an actual owned trim edge/vertex witness instead of accepting a band
        # of the untrimmed support surface as physical material.
        import cadquery as cq
        from OCP.BRepExtrema import BRepExtrema_DistShapeShape
        point=BRepAdaptor_Surface(face).Value(u,v)
        edges=cq.Compound.makeCompound(cq.Face(face).Edges())
        trim=BRepExtrema_DistShapeShape(edges.wrapped,cq.Vertex.makeVertex(point.X(),point.Y(),point.Z()).wrapped)
        if not trim.IsDone() or trim.NbSolution()<1:raise ValueError('native face support cannot be resolved to its actual trim')
        return owned_extrema_point(trim,edges)
    return BRepAdaptor_Surface(face).Value(u,v)
