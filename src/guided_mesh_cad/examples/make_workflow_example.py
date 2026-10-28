"""Small generic curved assembly exercising run, resume and stored diagnostics."""
import argparse
import hashlib
import json
from pathlib import Path

def main(argv=None):
    parser=argparse.ArgumentParser()
    parser.add_argument('--out',required=True)
    args=parser.parse_args(argv)
    out=Path(args.out).resolve()
    if out.exists() and any(out.iterdir()):raise ValueError('example directory must be empty')
    out.mkdir(parents=True,exist_ok=True)
    import cadquery as cq
    body=(cq.Workplane('XY').rect(2.5,1.5).workplane(offset=1.5)
          .transformed(rotate=(0,0,25)).rect(1.5,2.8).workplane(offset=2)
          .center(.3,0).rect(1.9,1.3).loft(combine=False).val())
    pin=cq.Solid.makeCylinder(.3,1.2)
    body.exportBrep(str(out/'freeform.brep'));pin.exportBrep(str(out/'pin.brep'))
    def write(name,data):
        (out/name).write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
    scope='Generic software fixture only; no product design, material or motion acceptance.'
    resources=[{'id':name,'path':name+'.brep','sha256':hashlib.sha256((out/(name+'.brep')).read_bytes()).hexdigest(),'scope':scope} for name in ('freeform','pin')]
    write('assembly.json',{'schema':'guided-mesh-cad.assembly.v1','units':'mm','resources':resources,'instances':[
        {'id':'body','resource_id':'freeform','matrix':[[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]},
        {'id':'pin','resource_id':'pin','matrix':[[0,-1,0,5],[1,0,0,2],[0,0,1,.5],[0,0,0,1]]}], 'excluded':[]})
    write('plan.json',{'schema':'guided-mesh-cad.workflow.v1','resources':{'memory_limit_mib':4096,'reserve_physical_mib':1536,'reserve_commit_mib':1536},'tasks':[
        {'id':'fingerprints','operation':'inspect-input','inputs':{'freeform':'freeform.brep','pin':'pin.brep'},'params':{},'depends_on':[],'scope':scope},
        {'id':'build','operation':'assemble','inputs':{'plan':'assembly.json'},'params':{'tolerance_mm':.01},'depends_on':['fingerprints'],'scope':scope},
        {'id':'verify','operation':'verify-assembly','inputs':{'plan':'assembly.json','build_manifest':{'task':'build','artifact':'manifest'}},'params':{'tolerance_mm':.01},'depends_on':[],'scope':scope}]})
    print(json.dumps({'plan':str(out/'plan.json'),'assembly_plan':str(out/'assembly.json'),'geometry_acceptance':'not_checked'}))

if __name__=='__main__':main()
