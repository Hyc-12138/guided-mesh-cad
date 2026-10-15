"""Complete non-planar, uneven loop: sixteen charts, genus-one topology."""
from pathlib import Path
import argparse,json,hashlib
import numpy as np


def surface(u,v):
    r=1.65+.28*np.sin(3*u+2*v)+.12*np.cos(5*u-v)
    return np.stack([(8+.8*np.sin(3*u)+r*np.cos(v))*np.cos(u)+.2*np.sin(v+u)*np.sin(u),
                     (7+.5*np.cos(2*u)+r*np.cos(v))*np.sin(u)-.2*np.sin(v+u)*np.cos(u),
                     1.6*np.sin(2*u)+.5*np.cos(3*u)+r*np.sin(v)],axis=-1)


def main(argv=None):
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);args=p.parse_args(argv)
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    vertices=[];faces=[];patches=[];n=65
    for a in range(4):
        for b in range(4):
            u,v=np.meshgrid(np.linspace(a*np.pi/2,(a+1)*np.pi/2,n),np.linspace(b*np.pi/2,(b+1)*np.pi/2,n),indexing='ij')
            grid=surface(u,v);start=len(vertices);vertices.extend(grid.reshape(-1,3))
            ids=np.arange(start,start+n*n).reshape(n,n)
            patches.append({'id':f'patch_{a}_{b}','kind':'indexed_grid','indices':ids[::4,::4].tolist()})
            for i in range(n-1):
                for j in range(n-1):
                    t=ids[i,j];faces.extend([[t,t+n,t+1],[t+1,t+n,t+n+1]])
    path=out/'source.npz';np.savez_compressed(path,vertices=np.array(vertices),faces=np.array(faces))
    plan={'schema_version':1,'units':'mm','scope':'Complete non-planar uneven closed loop; 16 freeform patches, handle topology and overhangs.',
          'mesh':'source.npz','mesh_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
          'patches':patches,'verification':{'tolerance_mm':.01,'samples':4000}}
    (out/'plan.json').write_text(json.dumps(plan,indent=2),encoding='utf-8')
    print(json.dumps({'vertices':len(vertices),'triangles':len(faces),'patches':len(patches)}))


if __name__=='__main__':main()
