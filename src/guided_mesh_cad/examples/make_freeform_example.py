"""Synthetic uneven body: six freeform faces, off-centre depression, skew bore.

No customer data. The reference uses a denser independently sampled analytic
surface than the reconstruction grid. It is an algorithm benchmark, not proof
of real-world segmentation or unseen feature recovery.
"""
from pathlib import Path
import argparse
import json
import hashlib
import numpy as np


def deform(q):
    x,y,z = np.moveaxis(np.asarray(q), -1, 0)
    return np.stack([8*x+1.15*np.sin(1.7*y+.3*z)+.45*y*z,
                     6*y+.8*np.sin(1.5*z+.5*x)+.4*x*z,
                     5*z+1.1*np.sin(1.4*x+.6*y)+.6*x*y-
                     1.8*(z+1)/2*np.exp(-((x-.2)**2/.18+(y+.3)**2/.2))], axis=-1)


def cube_patches(n):
    grids=[]; vertices=[]; faces=[]; patches=[]
    uv=np.linspace(-1,1,n)
    for axis in range(3):
        for side in [-1,1]:
            coordinates=np.empty((n,n,3)); coordinates[...,axis]=side
            remaining=[i for i in range(3) if i!=axis]
            a,b=np.meshgrid(uv,uv,indexing='ij')
            coordinates[...,remaining[0]]=a;coordinates[...,remaining[1]]=b
            grid=deform(coordinates); grids.append(grid)
            start=len(vertices); vertices.extend(grid.reshape(-1,3))
            ids=np.arange(start,start+n*n).reshape(n,n)
            patches.append({'id':f'face_{axis}_{side}', 'kind':'indexed_grid','indices':ids.tolist()})
            for i in range(n-1):
                for j in range(n-1):
                    p=ids[i,j];faces.extend([[p,p+n,p+1],[p+1,p+n,p+n+1]])
    return np.asarray(vertices),np.asarray(faces),patches,grids


def main(argv=None):
    parser=argparse.ArgumentParser();parser.add_argument('--out',required=True);args=parser.parse_args(argv)
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    v,f,patches,_=cube_patches(25)
    np.savez_compressed(out/'construction.npz',vertices=v,faces=f)
    # Denser analytic reference. Interpolation/trimming is a kernel ground truth
    # used solely for this test; the target package receives only its mesh.
    from guided_mesh_cad.cad import interpolate_face,sew_solid,apply_operations,triangulate
    _,_,_,dense=cube_patches(49)
    reference=sew_solid([interpolate_face(g,1e-7) for g in dense],1e-6)
    operation={'kind':'cylinder_cut','origin':[-14,-2,-2],'direction':[1,.12,.21],
               'radius':1.15,'length':30,'evidence':'Synthetic fixture: exact declared skew through-bore.'}
    reference=apply_operations(reference,[operation])
    rv,rf=triangulate(reference,.0005)
    np.savez_compressed(out/'reference.npz',vertices=rv,faces=rf)
    digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    plan={'schema_version':1,'units':'mm', 'scope':'Synthetic complete six-patch asymmetric body with concave depression and skew through bore.',
          'mesh':'reference.npz','mesh_sha256':digest(out/'reference.npz'),
          'construction_mesh':'construction.npz','construction_mesh_sha256':digest(out/'construction.npz'),
          'construction_evidence':'Synthetic analytic deformation sampled at 25x25 per face before bore; a deliberately provided construction aid, not automatically inferred from the final input mesh.',
          'patches':patches,'operations':[operation],
          'verification':{'tolerance_mm':.01,'samples':4000}}
    (out/'plan.json').write_text(json.dumps(plan,indent=2),encoding='utf-8')
    print(json.dumps({'vertices':len(rv),'triangles':len(rf),'plan':str(out/'plan.json')}))


if __name__=='__main__':main()
