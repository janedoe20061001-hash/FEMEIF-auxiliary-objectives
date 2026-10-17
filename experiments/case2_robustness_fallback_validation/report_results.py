"""Audit and analyze independent fallback validation."""
from pathlib import Path
from collections import defaultdict
import csv,hashlib,json
import numpy as np
import experiment as e
def bootstrap(candidate,baseline,components,seed):
    d=candidate-baseline;rng=np.random.default_rng(seed);b=d[rng.integers(0,len(d),(20000,len(d)))].mean(1)
    return d,np.quantile(b,[.025,.975],axis=0),np.quantile(b,[.025/components,1-.025/components],axis=0)
def main():
    root=e.HERE/'runs';mf=json.loads((root/'manifest.json').read_text());cfg=mf['config']
    for n,p in e.dependency_files().items():
        if hashlib.sha256(p.read_bytes()).hexdigest()!=mf['code_sha256'][n]:raise ValueError(f'code changed {n}')
    cfg2,full_f,full_z=e.config(e.HERE/'config.json')
    if cfg2!=cfg:raise ValueError('config changed')
    seeds=list(range(cfg['seeds']['start'],cfg['seeds']['start']+cfg['seeds']['count']));expected={(c['name'],e.group_name(g),s) for c in cfg['conditions'] for g in e.GROUPS for s in seeds}
    seen=set();records=[];initial=defaultdict(set)
    for p in root.glob('*/*/seed_*/summary.json'):
      r=json.loads(p.read_text());key=(r['condition'],r['group'],r['seed'])
      if key in seen or r['fingerprint']!=mf['fingerprint']:raise ValueError(f'invalid {p}')
      seen.add(key);x0=np.load(p.parent/'initial_log_parameters.npy');digest=hashlib.sha256(x0.tobytes()).hexdigest();initial[(r['condition'],r['seed'])].add(digest)
      if digest!=r['initial_population_sha256']:raise ValueError('initial mismatch')
      c=next(x for x in cfg['conditions'] if x['name']==r['condition']);f,z,meta=e.BASE.observation(full_f,full_z,c,r['seed']);g=next(x for x in e.GROUPS if e.group_name(x)==r['group'])
      with np.load(p.parent/'final_population.npz') as pop:
       x,F=pop['log_parameters'],pop['objectives'];verified=e.BASE.Objectives(f,cfg['features'],g,target_z=z).evaluate_spectra(e.BASE.impedance(f,10**x));np.testing.assert_allclose(verified,F,rtol=1e-11,atol=1e-12)
       sel=e.BASE.select_wsmse(F,cfg['final_selection']['phase_weight']);assert sel['wsmse_selected_index']==int(pop['selected_index'])
       clean=e.BASE.Objectives(full_f,cfg['features'],e.BASE.DIAGNOSTICS,target_z=full_z).evaluate_spectra(e.BASE.impedance(full_f,pop['selected_parameters']))[0];np.testing.assert_allclose(clean,pop['selected_clean_metrics'],rtol=1e-11,atol=1e-12)
      if meta['observed_target_sha256']!=r['observed_target_sha256']:raise ValueError('observation mismatch')
      records.append(r)
    if seen!=expected or any(len(v)!=1 for v in initial.values()):raise ValueError('missing/unpaired')
    data={(r['condition'],r['group'],r['seed']):r for r in records};alpha=cfg['final_selection']['phase_weight'];rows=[];components=2*(len(e.GROUPS)-1)*len(cfg['conditions'])
    for ci,c in enumerate(cfg['conditions']):
      b=np.array([[data[c['name'],'baseline',s]['clean_magnitude_mse'],data[c['name'],'baseline',s]['clean_phase_mse']] for s in seeds])
      for gi,g in enumerate(e.GROUPS[1:]):
       name=e.group_name(g);a=np.array([[data[c['name'],name,s]['clean_magnitude_mse'],data[c['name'],name,s]['clean_phase_mse']] for s in seeds]);d,ordinary,adj=bootstrap(a,b,components,20260921+ci*10+gi);change=100*d.mean(0)/b.mean(0);w=100*((a[:,0]+alpha*a[:,1]).mean()/(b[:,0]+alpha*b[:,1]).mean()-1)
       rows.append({'condition':c['name'],'group':name,'magnitude_change_pct':float(change[0]),'phase_change_pct':float(change[1]),'wsmse_change_pct':float(w),'both_lower_seeds':int(np.all(d<0,axis=1).sum()),'ordinary_both':bool(np.all(ordinary[1]<0)),'adjusted_both':bool(np.all(adj[1]<0)),'magnitude_ci_high':float(ordinary[1,0]),'phase_ci_high':float(ordinary[1,1]),'magnitude_adjusted_high':float(adj[1,0]),'phase_adjusted_high':float(adj[1,1])})
    with (root/'fallback_summary.csv').open('w',encoding='utf-8-sig',newline='') as h:w=csv.DictWriter(h,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    text=['# 条件型备选组合独立验证','',f"新 seeds {seeds[0]}–{seeds[-1]}；共 {len(records)} 次优化。所有误差在原始201点目标上计算。",'', '|条件|组合|幅值变化%|相位变化%|WSMSE变化%|两项同降seed|普通双通过|严格双通过|','|---|---|---:|---:|---:|---:|---|---|']
    for r in rows:text.append(f"|{r['condition']}|{r['group']}|{r['magnitude_change_pct']:.3f}|{r['phase_change_pct']:.3f}|{r['wsmse_change_pct']:.3f}|{r['both_lower_seeds']}/{len(seeds)}|{'是' if r['ordinary_both'] else '否'}|{'是' if r['adjusted_both'] else '否'}|")
    (root/'RESULTS.md').write_text('\n'.join(text)+'\n',encoding='utf-8');audit={'status':'passed','runs':len(records),'expected_runs':len(expected),'same_condition_seed_initial_populations_match':True,'observations_reconstructed':True,'final_populations_recomputed':True,'clean_metrics_recomputed':True,'selection_recomputed':True,'fingerprint':mf['fingerprint']};(root/'completion_audit.json').write_text(json.dumps(audit,indent=2)+'\n');print(json.dumps({'status':'complete','runs':len(records)}))
if __name__=='__main__':main()

