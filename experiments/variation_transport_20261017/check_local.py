"""Independent local Python/nodal checks after recovery; no scoring changes."""
from metric import *
from core import checked_nodal
def main():
    export=json.loads((OUT/'NATIVE_CURVES.json').read_text());cache={};replays=[];models=[];physical=[]
    tasks=json.loads((OUT/'TASKS.json').read_text())['tasks'];lookup={t[0]:t for t in tasks}
    for item in export['curves']:
        tid=item['target'];p=np.array(item['parameters']);f=np.array(item['frequency']);native=np.array(item['z_real'])+1j*np.array(item['z_imag'])
        if tid not in cache:cache[tid]=Variation(tid)
        m=cache[tid];python=m.c.response(f,p)[0];err=float(np.max(abs(native-python)/np.maximum(abs(python),1e-30)))
        assert np.all(abs(native-python)<=1e-8+1e-7*abs(python)),item['id']
        ix=np.unique(np.linspace(0,len(f)-1,min(21,len(f))).astype(int));nodal=checked_nodal(m.c,f[ix],p)
        task=lookup[item['id']];label=task[3][0];rec=json.loads((OUT/'scores'/f"{item['id']}.json").read_text())
        finest=rec['windows'][label]['polygon_levels'][-1]['n'];v=m.polygon(p,label,finest);r=rec['windows'][label]['scores'];assert agree(v,r)
        replays.append(dict(id=item['id'],scores=v,server_scores=r,passed=True));models.append(dict(id=item['id'],native_vs_python_max_relative=err,nodal=nodal))
    c=control_target()['circuit']
    for task in tasks[:684]:
        p=np.array(task[2]);f=np.unique(np.r_[np.geomspace(100,1e6,101),*[fc for fc,w in c.modes(p)]]);z=c.response(f,p)[0];other=np.zeros(len(f),complex)
        for r,l,cc in p.reshape(-1,3):
            f0=1/(2*np.pi*np.sqrt(l*cc));q=r*np.sqrt(cc/l);other+=r/(1+1j*q*(f/f0-f0/f))
        assert np.all(abs(z-other)<=1e-8+1e-7*abs(other))
        physical.append(dict(id=task[0],max_relative=float(np.max(abs(z-other)/np.maximum(abs(other),1e-30)))))
    dump(OUT/'LOCAL_CHECKS.json',dict(passed=True,replays=replays,native_models=models,controlled_independent=physical,analytic=analytic_checks(),local_numpy=np.__version__))
    print('Local checks passed:',len(replays),'replays;',len(models),'nodal curves;',len(physical),'physical settings',flush=True)
if __name__=='__main__':main()
