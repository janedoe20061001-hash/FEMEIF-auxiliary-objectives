"""Matched-primary-error ranking of a signed-phase candidate on saved populations."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'): os.environ[key]='1'
from pathlib import Path
import csv
import hashlib
import importlib.util
import io
import json
import platform
import sys
import tarfile
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT/'tmp/curve_refinement_deps'))
from scipy.spatial import cKDTree
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

SOURCE=ROOT/'experiments/crosscase_consolidation_20261005'
OUT=ROOT/'output/indicator_design_probe_20261006'
ARCHIVE=SOURCE/'server_run/formal_results.tar.gz'
REVIEW=SOURCE/'server_run/review_results'
COLS=['magnitude_mse','phase_mse','I03','II02','D_phi','D_LP','worst_band_phase_rmse','same_sign_phase_mse','wrong_sign_phase_mse']
GROUPS=['baseline','reactance_crossing_direction_consistency+characteristic_frequency_shift_error_decades',
        'reactance_crossing_direction_consistency+transition_speed_relative_error','cumulative_length_logpolar',
        'all_interval_mean_residual_logpolar','bidirectional_residual_logpolar']
if hasattr(sys.stdout,'reconfigure'): sys.stdout.reconfigure(encoding='utf-8')


def dump(path,value): path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def csvout(path,rows):
    if not rows: return
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path); m=importlib.util.module_from_spec(spec)
    sys.modules[name]=m; spec.loader.exec_module(m); return m


class Evaluator:
    def __init__(self,case):
        self.case=case
        self.legacy=module('probe_legacy_'+case,SOURCE/case/'legacy_objectives.py')
        self.config=json.loads((SOURCE/case/'config.json').read_text(encoding='utf-8'))
        self.saved=np.load(REVIEW/case/'analysis/selected_curves.npz')
        self.f=self.saved['frequency']; self.x=np.log10(self.f); self.h=np.diff(self.x)
        if case=='case1':
            self.impedance=self.legacy.impedance; self.target=self.impedance(self.f,self.legacy.TARGET)[0]
            self.lower,self.upper=self.legacy.LOWER,self.legacy.UPPER
        else:
            self.network=module('probe_network_case2',SOURCE/case/'case2.py')
            self.impedance=self.network.impedance; f,self.target=self.network.load_target(self.config['target_data'])
            np.testing.assert_array_equal(f,self.f); self.lower,self.upper=self.network.LOWER,self.network.UPPER
        self.phase=np.angle(self.target); self.amp=20*np.log10(abs(self.target))
        self.slope=np.diff(self.phase)/self.h; self.energy=np.sum(self.h*self.slope**2)
        assert self.energy>0
        self.scale=np.array([np.ptp(self.amp),np.ptp(self.phase)])
        self.target_lp=np.stack([self.amp,self.phase],axis=-1)/self.scale
        self.lp_slope=np.diff(self.target_lp,axis=0)/self.h[:,None]
        self.lp_energy=np.sum(self.h*np.sum(self.lp_slope**2,axis=1))
        u=(self.x-self.x[0])/(self.x[-1]-self.x[0]); self.bins=np.minimum((u*8).astype(int),7)
        assert all(np.any(self.bins==j) for j in range(8))
        self.old=self.legacy.Objectives(self.f,aux=('reactance_crossing_direction_consistency','transition_speed_relative_error'),target_z=self.target)

    def score(self,z):
        z=np.atleast_2d(z); phase=np.angle(z); amp=20*np.log10(abs(z)); err=(phase-self.phase)**2
        old=self.old.evaluate_spectra(z)
        dphi=np.sqrt(np.sum(self.h*(np.diff(phase,axis=1)/self.h-self.slope)**2,axis=1)/self.energy)
        lp=np.stack([amp,phase],axis=-1)/self.scale
        dlp=np.sqrt(np.sum(self.h*np.sum((np.diff(lp,axis=1)/self.h[:,None]-self.lp_slope)**2,axis=2),axis=1)/self.lp_energy)
        worst=np.sqrt(np.max(np.stack([err[:,self.bins==j].mean(axis=1) for j in range(8)],axis=1),axis=1))
        same=np.signbit(z.imag)==np.signbit(self.target.imag)
        result=np.column_stack([old,dphi,dlp,worst,(err*same).mean(axis=1),(err*(~same)).mean(axis=1)])
        assert np.isfinite(result).all() and np.all(result>=0)
        np.testing.assert_allclose(result[:,0],((amp-self.amp)**2).mean(axis=1),rtol=1e-10,atol=1e-10)
        np.testing.assert_allclose(result[:,1],err.mean(axis=1),rtol=1e-10,atol=1e-12)
        np.testing.assert_allclose(result[:,7]+result[:,8],result[:,1],atol=1e-14)
        return result

    def spectrum(self,log_parameters): return self.impedance(self.f,10**np.atleast_2d(log_parameters))


def match_pairs(scores):
    coords=np.log(np.maximum(scores[:,:2],1e-30))
    pairs=cKDTree(coords).query_pairs(np.log(1.10),p=np.inf,output_type='ndarray')
    if not len(pairs): return np.empty((0,2),int)
    q=scores[:,6]; a,b=pairs.T
    valid=np.maximum(q[a],q[b])>=1.20*np.maximum(np.minimum(q[a],q[b]),1e-15)
    pairs=pairs[valid]
    if not len(pairs): return np.empty((0,2),int)
    ratio=np.maximum(q[pairs[:,0]],q[pairs[:,1]])/np.maximum(np.minimum(q[pairs[:,0]],q[pairs[:,1]]),1e-15)
    order=np.lexsort((pairs[:,1],pairs[:,0],-ratio)); used=set(); kept=[]
    for a,b in pairs[order]:
        if a in used or b in used: continue
        used.update((int(a),int(b)))
        kept.append((int(a),int(b)) if q[a]<q[b] else (int(b),int(a)))
    return np.asarray(kept,dtype=int).reshape(-1,2)


def concordance(values,pairs):
    good,bad=values[pairs[:,0]],values[pairs[:,1]]
    tie=np.abs(good-bad)<=1e-12*np.maximum(1,np.maximum(abs(good),abs(bad)))
    return np.where(tie,.5,(good<bad).astype(float))


def main():
    start=time.perf_counter(); OUT.mkdir(exist_ok=True,parents=True)
    for folder in ('pools','figures','data'): (OUT/folder).mkdir(exist_ok=True)
    archive_record=next(r for r in json.loads((SOURCE/'server_run/ARCHIVES.json').read_text(encoding='utf-8')) if r['file']=='formal_results.tar.gz')
    assert sha(ARCHIVE)==archive_record['sha256']
    inputs=[ARCHIVE,*[SOURCE/c/'config.json' for c in ('case1','case2')],*[REVIEW/c/'analysis/selected_curves.npz' for c in ('case1','case2')],SOURCE/'case2/data/targetImpedance_case2.csv']
    dump(OUT/'data/PROVENANCE.json',dict(protocol_sha256=sha(Path(__file__).with_name('PROTOCOL.md')),
        source_sha256={str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__),*[SOURCE/c/'legacy_objectives.py' for c in ('case1','case2')],SOURCE/'case2/case2.py']},
        input_sha256={str(p.relative_to(ROOT)):sha(p) for p in inputs},new_optimizations=0,
        runtime=dict(python=platform.python_version(),numpy=np.__version__),saved_selected_parameters_encoding='log10; the old NPZ field is named parameters'))
    ev={c:Evaluator(c) for c in ('case1','case2')}
    checks=[]
    for c,e in ev.items():
        s=e.score(e.target); assert np.max(s)<1e-12
        scale=e.score(1.3*e.target); assert scale[0,4]<1e-12 and abs(scale[0,1])<1e-20
        # Curve-space invariance; not asserted to be realizable by the case circuit.
        residual=np.diff(e.phase+.01)/e.h-e.slope
        assert np.max(abs(residual))<1e-10
        checks.append(dict(case=c,target_zero=True,positive_scale_phase_invariance=True,constant_phase_offset_blind=True,
                           max_target_phase_step_rad=float(np.max(abs(np.diff(e.phase))))))
    pool={}; archive_count=0; archive_score_checks=0; archive_aux_checks=0
    with tarfile.open(ARCHIVE,mode='r|gz') as tar:
        for member in tar:
            if not member.name.endswith('/final_population.npz'): continue
            case,_,group,seed_dir,_=member.name.split('/'); seed=int(seed_dir.split('_')[1])
            assert case in ev and group in GROUPS and 2000<=seed<2100
            with np.load(io.BytesIO(tar.extractfile(member).read())) as saved:
                pars=saved['log_parameters'].copy(); original=saved['objectives'].copy()
            pool.setdefault((case,seed),[]).append((GROUPS.index(group),pars,original)); archive_count+=1
    assert archive_count==1200
    print('Verified archive; loaded 1,200 final populations.',flush=True)
    pair_rows=[]; seed_rows=[]; population_rows=[]; example_candidates=[]; total_individuals=0
    for case,e in ev.items():
        for seed in range(2000,2100):
            entries=sorted(pool[(case,seed)],key=lambda r:r[0]); assert [r[0] for r in entries]==list(range(6))
            final=np.concatenate([r[1] for r in entries]); original=np.concatenate([r[2][:,:2] for r in entries])
            group_ix=np.concatenate([np.full(len(r[1]),r[0]) for r in entries])
            initial=np.random.default_rng(seed).uniform(e.lower,e.upper,(e.config['population'],len(e.lower)))
            for context,pars,group_ids in [('initial',initial,np.full(len(initial),-1)),('final',final,group_ix)]:
                spectra=e.spectrum(pars); scores=e.score(spectra); total_individuals+=len(pars)
                if context=='final':
                    np.testing.assert_allclose(scores[:,:2],original,atol=1e-9,rtol=1e-9); archive_score_checks+=len(pars)
                    offset=0
                    for group,parameters,saved_scores in entries:
                        if group in (1,2):
                            np.testing.assert_allclose(scores[offset:offset+len(parameters),2],saved_scores[:,2],atol=1e-10,rtol=1e-9)
                            archive_aux_checks+=len(parameters)
                        if group==2:
                            np.testing.assert_allclose(scores[offset:offset+len(parameters),3],saved_scores[:,3],atol=1e-9,rtol=1e-9)
                            archive_aux_checks+=len(parameters)
                        offset+=len(parameters)
                pairs=match_pairs(scores)
                np.savez_compressed(OUT/'pools'/f'{case}_{context}_{seed}.npz',log_parameters=pars,group_index=group_ids,scores=scores,score_names=np.asarray(COLS),pairs=pairs)
                row=dict(case=case,context=context,seed=seed,individuals=len(pars),pairs=len(pairs),eligible=len(pairs)>=5)
                for metric,col in [('II02',3),('D_phi',4),('D_LP',5),('phase_MSE',1)]:
                    row[metric+'_correct_fraction']=float(concordance(scores[:,col],pairs).mean()) if len(pairs) else None
                seed_rows.append(row)
                for good,bad in pairs:
                    pair_rows.append(dict(case=case,context=context,seed=seed,good_index=int(good),bad_index=int(bad),good_group=int(group_ids[good]),bad_group=int(group_ids[bad]),
                        **{f'{side}_{name}':float(scores[ix,col]) for side,ix in [('good',good),('bad',bad)] for col,name in enumerate(COLS)}))
                if context=='final' and len(pairs):
                    # Representative plots selected by independent local-quality contrast, before auxiliary preference.
                    good,bad=pairs[0]; ratio=scores[bad,6]/scores[good,6]
                    example_candidates.append(dict(case=case,seed=seed,ratio=float(ratio),good=int(good),bad=int(bad),scores=scores[[good,bad]].copy(),z=spectra[[good,bad]].copy()))
            if (seed-1999)%25==0: print(f'{case}: {seed-1999}/100 seeds evaluated.',flush=True)
    csvout(OUT/'data/pairs.csv',pair_rows); csvout(OUT/'data/seed_ranking.csv',seed_rows)
    rng=np.random.default_rng(20261006); ranking=[]
    for case in ev:
        for context in ('initial','final'):
            rr=[r for r in seed_rows if r['case']==case and r['context']==context and r['eligible']]
            d=np.array([r['D_phi_correct_fraction']-r['II02_correct_fraction'] for r in rr])
            if len(d):
                boot=d[rng.integers(0,len(d),(10000,len(d)))].mean(axis=1); interval=np.quantile(boot,[.025,.975]).tolist()
            else: interval=[None,None]
            ranking.append(dict(case=case,context=context,eligible_seeds=len(rr),matched_pairs=sum(r['pairs'] for r in rr),
                **{metric+'_seed_mean':float(np.mean([r[metric+'_correct_fraction'] for r in rr])) if rr else None for metric in ('II02','D_phi','D_LP','phase_MSE')},
                candidate_minus_speed=float(d.mean()) if len(d) else None,paired_seed_bootstrap95=interval))
    failure_rows=[]; failure_summary=[]; selected_plots=[]
    for case,e in ev.items():
        zz=e.saved['curves']; old=e.score(zz.reshape(-1,len(e.f))).reshape(100,6,-1)
        np.testing.assert_allclose(e.spectrum(e.saved['parameters'].reshape(-1,len(e.lower))).reshape(zz.shape),zz,atol=1e-9,rtol=1e-9)
        mask=(old[:,2,3]<old[:,0,3])&(old[:,2,1]>old[:,0,1])
        for i,seed in enumerate(e.saved['seeds']):
            bands=np.array([((np.angle(zz[i,2,e.bins==j])-e.phase[e.bins==j])**2).mean()-((np.angle(zz[i,0,e.bins==j])-e.phase[e.bins==j])**2).mean() for j in range(8)])
            j=int(bands.argmax()); ff=e.f[e.bins==j]
            failure_rows.append(dict(case=case,seed=int(seed),speed_improves_phase_worsens=bool(mask[i]),baseline_phase_mse=float(old[i,0,1]),old_pair_phase_mse=float(old[i,2,1]),
                baseline_speed=float(old[i,0,3]),old_pair_speed=float(old[i,2,3]),baseline_D_phi=float(old[i,0,4]),old_pair_D_phi=float(old[i,2,4]),
                same_sign_delta=float(old[i,2,7]-old[i,0,7]),wrong_sign_delta=float(old[i,2,8]-old[i,0,8]),
                largest_deterioration_band=j+1,band_min_hz=float(ff[0]),band_max_hz=float(ff[-1])))
        failure_summary.append(dict(case=case,seeds=100,speed_improves_phase_worsens=int(mask.sum()),
            phase_mean_baseline=float(old[:,0,1].mean()),phase_mean_old_pair=float(old[:,2,1].mean()),
            same_sign_mean_change=float(np.mean(old[:,2,7]-old[:,0,7])),wrong_sign_mean_change=float(np.mean(old[:,2,8]-old[:,0,8]))))
        indices=np.flatnonzero(mask); indices=indices[np.argsort(-(old[indices,2,1]-old[indices,0,1]))][:3]
        for i in indices: selected_plots.append((case,int(e.saved['seeds'][i]),zz[i,[0,2]]))
    csvout(OUT/'data/real_failure_seeds.csv',failure_rows)
    main_rank=next(r for r in ranking if r['case']=='case1' and r['context']=='final')
    reasons=[]
    if any(r['eligible_seeds']<50 for r in ranking): reasons.append('至少一个Case/阶段的可配对种子少于50，覆盖不足。')
    if main_rank['candidate_minus_speed'] is None or main_rank['candidate_minus_speed']<.05: reasons.append('Case1最终阶段相对速度项的排序率增量未达到5个百分点。')
    for r in ranking:
        if r is main_rank: continue
        if r['paired_seed_bootstrap95'][0] is None or r['paired_seed_bootstrap95'][0]<-.05: reasons.append(f"{r['case']} / {r['context']} 的差值区间下界低于-5个百分点，存在排序退步。")
    gate=dict(passed=not reasons,reasons=reasons,primary_candidate='D_phi',candidate_definition_unchanged=True,new_optimizations=0)
    font_manager.fontManager.addfont('C:/Windows/Fonts/msyh.ttc')
    plt.rcParams.update({'font.family':'Microsoft YaHei','font.size':10,'axes.unicode_minus':False,'axes.spines.top':False,'axes.spines.right':False})
    plot_results(ev,ranking,selected_plots,example_candidates)
    summary=dict(saved_final_populations=archive_count,evaluated_individuals=total_individuals,saved_primary_score_checks=archive_score_checks,saved_auxiliary_score_checks=archive_aux_checks,
        matched_pairs=len(pair_rows),ranking=ranking,real_failure_summary=failure_summary,definition_checks=checks,gate=gate,
        elapsed_seconds=time.perf_counter()-start,local_only=True,new_optimizations=0)
    dump(OUT/'SUMMARY.json',summary); make_report(summary)
    files={str(p.relative_to(OUT)):sha(p) for p in OUT.rglob('*') if p.is_file() and p.name!='INTEGRITY.json'}
    dump(OUT/'INTEGRITY.json',dict(completed=True,source_sha256=sha(Path(__file__)),artifacts_sha256=files))
    print(json.dumps(summary,ensure_ascii=False),flush=True)


def plot_results(evaluators,ranking,failures,examples):
    fig,ax=plt.subplots(figsize=(10,4.3)); xx=np.arange(4); width=.24
    for k,(name,label,color) in enumerate([('II02','旧 II-02','#8d9caf'),('D_phi','候选 Dφ','#1769ab'),('phase_MSE','原相位 MSE','#168778')]):
        ax.bar(xx+(k-1)*width,[r[name+'_seed_mean'] or 0 for r in ranking],width,label=label,color=color)
    ax.set_xticks(xx,[r['case']+' / '+('初始' if r['context']=='initial' else '最终') for r in ranking]); ax.set_ylim(0,1); ax.set_ylabel('种子平均局部质量排序正确率'); ax.legend(ncol=3)
    ax.set_title('幅值与相位MSE均接近时，谁更能识别较差的局部相位？'); fig.tight_layout(); fig.savefig(OUT/'figures/ranking.png',dpi=150); plt.close(fig)
    for case,seed,z in failures:
        e=evaluators[case]; fig,axes=plt.subplots(2,1,figsize=(11,6),sharex=True)
        axes[0].semilogx(e.f,np.rad2deg(e.phase),color='#25394e',label='目标')
        for row,label,color in zip(z,('双主目标基线','原 I-03 + II-02'),('#168778','#bd553f')):
            phase=np.angle(row); axes[0].semilogx(e.f,np.rad2deg(phase),label=label,color=color)
            axes[1].semilogx(e.f,np.rad2deg(phase-e.phase),color=color)
        axes[0].set_ylabel('相位 / °'); axes[1].set_ylabel('相位残差 / °'); axes[1].set_xlabel('频率 / Hz'); axes[0].legend()
        for ax in axes:
            for boundary in np.linspace(e.x[0],e.x[-1],9)[1:-1]: ax.axvline(10**boundary,color='#dce3ea',lw=.7)
            ax.grid(alpha=.15)
        fig.suptitle(f'{case} / seed {seed}：速度项下降，但相位MSE上升（旧实验选中解）'); fig.tight_layout(); fig.savefig(OUT/'figures'/f'{case}_failure_{seed}.png',dpi=150); plt.close(fig)
    for case in evaluators:
        chosen=sorted([r for r in examples if r['case']==case],key=lambda r:(-r['ratio'],r['seed']))[:2]
        for j,r in enumerate(chosen):
            e=evaluators[case]; fig,axes=plt.subplots(1,2,figsize=(12,4.3))
            for i,(label,color) in enumerate([('局部误差较好','#168778'),('局部误差较差','#bd553f')]):
                axes[0].semilogx(e.f,np.rad2deg(np.angle(r['z'][i])-e.phase),label=label,color=color)
            axes[0].legend(); axes[0].set_xlabel('频率 / Hz'); axes[0].set_ylabel('相位残差 / °'); axes[0].grid(alpha=.15)
            axes[1].axis('off'); text=['同种子最终种群，主误差各自相差≤10%','',*[
                f'{name}：好 {r["scores"][0,k]:.5g} / 差 {r["scores"][1,k]:.5g}' for k,name in [(0,'幅值MSE'),(1,'相位MSE'),(3,'II-02'),(4,'Dφ'),(6,'最差频段相位RMSE')]]]
            axes[1].text(.02,.95,'\n'.join(text),va='top',fontsize=12)
            fig.suptitle(f'{case} / seed {r["seed"]}：主误差接近的对照（按局部质量差异选图）'); fig.tight_layout(); fig.savefig(OUT/'figures'/f'{case}_matched_{j+1}.png',dpi=150); plt.close(fig)


def make_report(summary):
    lines=['# 带方向相位指标的预筛结果','',
        f"2026-10-06。本地重建{summary['saved_final_populations']:,}个已保存最终种群，共评价{summary['evaluated_individuals']:,}个初始/最终个体；新增优化0次。",'',
        '## 检验什么','',
        '主候选Dφ沿用前轮带方向相位割线定义。主幅值和相位MSE分别接近（比值≤1.10）的候选中，比较它和II-02能否优先识别最差频段角度误差较小的曲线。局部质量判据为8个等log频段中的最大相位RMSE；它不含相位导数，仍是预先选定的一种局部质量标准，不是所有形状质量的真值。', '',
        '## 排序结果','',
        '| Case / 阶段 | 种子数 | 配对数 | II-02正确率 | Dφ正确率 | 原相位MSE正确率 | Dφ−II-02（95%描述区间） |',
        '| --- | --- | --- | --- | --- | --- | --- |']
    for r in summary['ranking']:
        fm=lambda v:'—' if v is None else f'{100*v:.1f}%'
        lo,hi=r['paired_seed_bootstrap95']; diff=f"{fm(r['candidate_minus_speed'])} [{fm(lo)}, {fm(hi)}]"
        lines.append(f"| {r['case']} / {r['context']} | {r['eligible_seeds']} | {r['matched_pairs']:,} | {fm(r['II02_seed_mean'])} | {fm(r['D_phi_seed_mean'])} | {fm(r['phase_MSE_seed_mean'])} | {diff} |")
    lines+=['','配对个体不重复使用，统计与bootstrap以种子为单位；每种子至少5对。初始/最终四个区间只作描述，不能直接写成四项校正后的显著性结果。','',
            '![排序对照](figures/ranking.png)','', '## 真实失败片段','',
            '| Case | II-02下降但相位MSE上升的种子 | 基线相位均值 rad² | 原组合相位均值 rad² | 同号误差均值变化 | 错号误差均值变化 |',
            '| --- | --- | --- | --- | --- | --- |']
    for r in summary['real_failure_summary']:
        lines.append(f"| {r['case']} | {r['speed_improves_phase_worsens']}/100 | {r['phase_mean_baseline']:.6g} | {r['phase_mean_old_pair']:.6g} | {r['same_sign_mean_change']:.6g} | {r['wrong_sign_mean_change']:.6g} |")
    lines+=['','这些来自旧实验固定WSMSE选中解，不是新优化结果。辅助项下降与相位变差同时出现是诊断证据，不证明进化失败的因果路径。']
    for p in sorted((OUT/'figures').glob('*failure*.png')): lines+=['',f'![真实失败片段](figures/{p.name})']
    lines+=['','## 主误差接近的曲线对照','', '图按局部质量差异确定性挑选，不按哪个指标胜出挑选。']
    for p in sorted((OUT/'figures').glob('*matched*.png')): lines+=['',f'![主误差接近的对照](figures/{p.name})']
    lines+=['','## 是否进入新优化','', '预筛门槛：**'+('通过' if summary['gate']['passed'] else '未通过')+'**。']
    lines += ['- '+r for r in summary['gate']['reasons']]
    lines+=['','通过仅表示值得用新种子检验，不等于获得优化收益；未通过也不证明所有带方向指标无效。本轮不更改候选定义或阈值以追求通过。','',
        'Dφ对恒定相位偏移不敏感，由原相位MSE锚定。正数整体阻抗缩放不改变相位，Dφ保持零响应。这些是定义性质，不要求每个辅助项包办所有误差。',
        '保留Case2原相位主值和原采样点，未展开分支、未删点、未平滑。割线可能对噪声和采样分辨率敏感。局部质量分窗、主误差10%容差与20%局部差异都是本轮预设筛选条件，结论不能外推到其他条件。',
        '',f"旧种群主误差重算核对{summary['saved_primary_score_checks']:,}个个体；所有候选评价均为有限非负值。原始包SHA256、代码、协议和输入版本见data/PROVENANCE.json。",'',
        '数据：data/seed_ranking.csv（种子统计）、data/pairs.csv（全部配对）、data/real_failure_seeds.csv（真实失败及频段）、pools/（完整评分、参数和配对索引）。','']
    (OUT/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')


if __name__=='__main__': main()
