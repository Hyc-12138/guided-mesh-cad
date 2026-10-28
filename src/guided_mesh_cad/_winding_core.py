"""Spatial hierarchy for the ORIGINAL signed solid-angle winding definition.
No vertex motion, hole fill, sampling or triangle omission. Classification uses abs(w)>.5.
The flux kernel F(x)=(x-p)/|x-p|^3 has Jacobian norm <=2/r^3.
A node's triangles lie in ball(c,R), so integral error <=2*area_upper*R/(D-R)^3.
Signed vector area gives the node estimate; area/vector, ball and floating pads enlarge bounds.
The analytic far-field bound and numerical floating-point padding are distinct.
The implementation is numerically conservative within tested ranges, not a formal
interval-arithmetic proof for arbitrary scales or ill-conditioned inputs.
Undecidable leaves use the original atan2 expression; open soups are not closed solids.
"""
import heapq
import numpy as np
try:
 import numba as nb
 ACCELERATION_BACKEND = 'numba'
except ImportError:
 ACCELERATION_BACKEND = 'python'
 class _PythonBackend:
  @staticmethod
  def njit(*args, **kwargs):
   return lambda function: function
 nb = _PythonBackend()

@nb.njit(cache=False)
def _estimate(p,c,r,avec,aup,aerr):
 dx=c[0]-p[0];dy=c[1]-p[1];dz=c[2]-p[2];d=np.sqrt(dx*dx+dy*dy+dz*dz);scale=1.+d+r+max(abs(c[0]),abs(c[1]),abs(c[2]))+max(abs(p[0]),abs(p[1]),abs(p[2]));margin=2e-12*scale;dl=d-margin;rr=r+margin
 if dl<=rr:return 0.,1e100,True
 q=(avec[0]*dx+avec[1]*dy+avec[2]*dz)/(d*d*d)
 e=2.*aup*rr/((dl-rr)**3)+aerr/(dl*dl)+1e-12*(1.+abs(q))
 return q,np.nextafter(e,np.inf),False
@nb.njit(cache=False)
def _leaf(p,T,start,end):
 acc=0.;absolute=0.
 for j in range(start,end):
  ax=T[j,0,0]-p[0];ay=T[j,0,1]-p[1];az=T[j,0,2]-p[2];bx=T[j,1,0]-p[0];by=T[j,1,1]-p[1];bz=T[j,1,2]-p[2];cx=T[j,2,0]-p[0];cy=T[j,2,1]-p[1];cz=T[j,2,2]-p[2]
  la=np.sqrt(ax*ax+ay*ay+az*az);lb=np.sqrt(bx*bx+by*by+bz*bz);lc=np.sqrt(cx*cx+cy*cy+cz*cz);det=ax*(by*cz-bz*cy)-ay*(bx*cz-bz*cx)+az*(bx*cy-by*cx)
  den=la*lb*lc+(ax*bx+ay*by+az*bz)*lc+(bx*cx+by*cy+bz*cz)*la+(cx*ax+cy*ay+cz*az)*lb
  v=2.*np.arctan2(det,den);acc+=v;absolute+=abs(v)
 return acc,absolute
@nb.njit(cache=False)
def _query(P,T,C,R,A,AU,AE,L,RR,ST,EN):
 n=len(P);states=np.zeros(n,np.int8);lo=np.empty(n);hi=np.empty(n);leaves=np.zeros(n,np.int64);nodes=np.zeros(n,np.int64);reached=np.zeros(n,np.bool_);pad=4.*np.pi*(1e-7+len(T)*5e-15)
 for pi in range(n):
  p=P[pi]
  if not np.all(np.isfinite(p)):lo[pi]=np.nan;hi[pi]=np.nan;continue
  val,err,inf=_estimate(p,C[0],R[0],A[0],AU[0],AE[0]);total=val;absolute_accumulation=abs(val);error=0. if inf else err;unknown=1 if inf else 0;heap=[(-err,0)]
  while len(heap):
   if unknown==0:
    numeric_pad=pad+1e-9*absolute_accumulation
    lower=(total-error-numeric_pad)/(4.*np.pi);upper=(total+error+numeric_pad)/(4.*np.pi)
    if lower>.5 or upper<-.5:states[pi]=1;break
    if lower>-.5 and upper<.5:states[pi]=-1;break
   neg,k=heapq.heappop(heap);nodes[pi]+=1;v,e,ii=_estimate(p,C[k],R[k],A[k],AU[k],AE[k]);total-=v
   if ii:unknown-=1
   else:error=np.nextafter(max(0.,error-e),np.inf)
   if L[k]<0:
    v,av=_leaf(p,T,ST[k],EN[k]);total+=v;absolute_accumulation+=av;leaves[pi]+=EN[k]-ST[k]
   else:
    for child in (L[k],RR[k]):
     v,e,ii=_estimate(p,C[child],R[child],A[child],AU[child],AE[child]);total+=v;absolute_accumulation+=abs(v)
     if ii:unknown+=1
     else:error=np.nextafter(error+e,np.inf)
     heapq.heappush(heap,(-e,child))
  if len(heap)==0:error=0.
  numeric_pad=pad+1e-9*absolute_accumulation
  lower=(total-error-numeric_pad)/(4.*np.pi);upper=(total+error+numeric_pad)/(4.*np.pi);lo[pi]=lower;hi[pi]=upper
  if len(heap)==0:
   reached[pi]=True
   if lower>.5 or upper<-.5:states[pi]=1
   elif lower>-.5 and upper<.5:states[pi]=-1
  if not np.isfinite(lower) or not np.isfinite(upper):states[pi]=0
 return states,lo,hi,leaves,nodes,reached
class _BoundedWindingCore:
 def __init__(self,V,F,leaf_size=24):
  V=np.asarray(V,float);F=np.asarray(F,np.int64);raw=V[F];cross=.5*np.cross(raw[:,1]-raw[:,0],raw[:,2]-raw[:,0]);ar=np.linalg.norm(cross,axis=1);prod=.5*np.linalg.norm(raw[:,1]-raw[:,0],axis=1)*np.linalg.norm(raw[:,2]-raw[:,0],axis=1);cen=raw.mean(1);order=np.arange(len(F));C=[];R=[];A=[];AU=[];AE=[];L=[];RS=[];ST=[];EN=[]
  def build(start,end):
   ids=order[start:end];tt=raw[ids];lo=tt.min((0,1));hi=tt.max((0,1));c=(lo+hi)/2.;radius=np.nextafter(np.linalg.norm(tt-c,axis=2).max(),np.inf);k=len(C);C.append(c);R.append(radius);A.append(cross[ids].sum(0));area_error=1e-10*ar[ids].sum()+1e-12*prod[ids].sum();AU.append(ar[ids].sum()+area_error);AE.append(area_error);L.append(-1);RS.append(-1);ST.append(start);EN.append(end)
   if end-start>leaf_size:
    ax=int(np.argmax(hi-lo));m=(end-start)//2;order[start:end]=ids[np.argpartition(cen[ids,ax],m)];mid=start+m;L[k]=build(start,mid);RS[k]=build(mid,end)
   return k
  build(0,len(F));self.T=np.ascontiguousarray(raw[order]);self.arrays=[np.array(x) for x in [C,R,A,AU,AE,L,RS,ST,EN]];self.triangles=len(F);self.nodes=len(C);self.vertex_coordinate_change=False;self.original_face_order=order.copy()
 def query(self,P,chunk=2000):
  P=np.ascontiguousarray(P,float);parts=[]
  for st in range(0,len(P),chunk):parts.append(_query(P[st:st+chunk],self.T,*self.arrays))
  if not parts:return dict(state=np.empty(0,np.int8),lower=np.empty(0),upper=np.empty(0),exact_triangles=np.empty(0,np.int64),nodes=np.empty(0,np.int64),full_atan2=np.empty(0,bool))
  vals=[np.concatenate([part[i] for part in parts]) for i in range(6)]
  return dict(zip(['state','lower','upper','exact_triangles','nodes','full_atan2'],vals))



# Import availability alone is not proof that JIT is enabled.
if not hasattr(_query, "signatures"):
 ACCELERATION_BACKEND = "python"
