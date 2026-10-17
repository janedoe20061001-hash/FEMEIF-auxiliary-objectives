from investigate import *
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
font=Path('C:/Windows/Fonts/msyh.ttc')
if font.exists():
    font_manager.fontManager.addfont(str(font));plt.rcParams['font.family']=font_manager.FontProperties(fname=str(font)).get_name()
plt.rcParams['axes.unicode_minus']=False
FIG=OUT/'figures';FIG.mkdir(exist_ok=True)
fig=plt.figure(figsize=(15,9))
for row,(ident,new,seed,band) in enumerate([
    ('mixed_resonators/base','D',45002,(45000,58000)),
    ('mixed_resonators/lc_reciprocal_8p0','A',45003,(38500,45000))]):
    c=target(ident)['circuit'];x=np.linspace(*np.log10(band),4097);f=10**x
    t,_=c.response(f);p,_,_=load(ident,'MAG_PHA',seed);z,_=c.response(f,p);pn,_,_=load(ident,new,seed);zn,_=c.response(f,pn)
    tm=descriptor(c,c.p);cm=descriptor(c,p)
    tg=max((g for g in groups(tm) if g['osc']),key=lambda q:q['freq']);cg=max((g for g in groups(cm) if g['osc']),key=lambda q:q['freq'])
    uu=1j*2*np.pi*f[:,None]/WREF
    ht=np.sum(tm['res'][None,tg['ix']]/(uu-tm['poles'][None,tg['ix']]),axis=1)
    hc=np.sum(cm['res'][None,cg['ix']]/(uu-cm['poles'][None,cg['ix']]),axis=1)
    hybrid=z-hc+ht
    axes=[fig.add_subplot(2,3,3*row+j+1,projection='3d' if j==2 else None) for j in range(3)]
    series=[('目标',t,'#181818','-',2.2),('基础解',z,'#dd7040','-',1.7),(new+'解',zn,'#2879b8','-',1.7),('仅换目标高频成分（诊断）',hybrid,'#8d58aa','--',1.4)]
    for label,q,color,style,lw in series:
        axes[0].plot(f/1000,q.real,color=color,ls=style,lw=lw,label=label)
        axes[1].plot(f/1000,q.imag,color=color,ls=style,lw=lw)
        axes[2].plot(f/1000,q.real,q.imag,color=color,ls=style,lw=lw)
    axes[0].set_title(('原始LC' if row==0 else 'L3×8、C3÷8')+f'，seed {seed}：实部')
    axes[1].set_title('虚部');axes[2].set_title('带频率的联合轨迹')
    for ax in axes[:2]:ax.set_xlabel('频率 / kHz');ax.set_ylabel('阻抗 / Ω');ax.grid(alpha=.2)
    axes[0].legend(fontsize=8)
    axes[2].set_xlabel('频率 / kHz');axes[2].set_ylabel('ReZ / Ω');axes[2].set_zlabel('ImZ / Ω')
fig.suptitle('局部弯折与端口高频成分：紫色线是诊断替换，未进行参数优化',fontsize=14)
fig.tight_layout(rect=(0,0,1,.95));fig.savefig(FIG/'modal_cause.png',dpi=160);plt.close(fig)

fig,axes=plt.subplots(1,2,figsize=(13,5))
for ax,m,seed,window in zip(axes,['A','D_H'],[45000,45001],['P6','gap_14']):
    d=target('case1');c=d['circuit'];ff=Features(d);p,_,_=load('case1',m,seed)
    w=next(w for w in ff.windows if w['block']['label']==window)
    pp=p.reshape(6,3);fc=1/(2*np.pi*np.sqrt(pp[:,1]*pp[:,2]));Q=pp[:,0]*np.sqrt(pp[:,2]/pp[:,1])
    good=(np.log10(fc)>=w['x'][0])&(np.log10(fc)<=w['x'][-1]);ii=np.flatnonzero(good)
    zc,_=c.response(fc[ii],p);j=ii[np.argmax(abs(zc))]
    xx=np.unique(np.r_[np.linspace(w['x'][0],w['x'][-1],4097),np.log10(fc[j])+np.linspace(-10,10,20001)/(Q[j]*np.log(10))])
    xx=xx[(xx>=w['x'][0])&(xx<=w['x'][-1])];zz,_=c.response(10**xx,p);tt,_=c.response(10**xx)
    coarse,_=c.response(10**w['x'],p)
    ax.plot(10**xx/1000,20*np.log10(abs(tt)),color='#181818',label='目标')
    ax.plot(10**xx/1000,20*np.log10(abs(zz)),color='#2879b8',label='候选：含峰中心的加密查询')
    ax.plot(10**w['x']/1000,20*np.log10(abs(coarse)),'o',ms=3,color='#dd7040',label='辅助评分原129节点')
    ax.set_title(f'Case1 {m}，seed {seed}，{window}')
    ax.set_xlabel('频率 / kHz');ax.set_ylabel('幅值 / dBΩ');ax.grid(alpha=.2);ax.legend(fontsize=8)
fig.suptitle('候选尖峰可以藏在辅助网格之间：这是模型查询，不是新增实测',fontsize=13)
fig.tight_layout(rect=(0,0,1,.93));fig.savefig(FIG/'case1_grid.png',dpi=160);plt.close(fig)

rr=list(csv.DictReader((OUT/'A_COMPONENT_RANKS.csv').open(encoding='utf-8-sig')))
fig,axes=plt.subplots(1,2,figsize=(13,4.7));names=['J0_32','D_32','H_32','A_full','AtR','AtX','ARX'];labels=['复数点误差','D','H','完整A','频率–Re','频率–Im','Re–Im']
for ax,ident,title in zip(axes,['mixed_resonators/base','mixed_resonators/lc_reciprocal_8p0'],['混合原始LC','混合L3×8、C3÷8']):
    vv=[float(next(r['rho_pair_failed'] for r in rr if r['target']==ident and r['score']==n)) for n in names]
    ax.bar(range(len(names)),vv,color=['#8a9ba8','#8a9ba8','#8a9ba8','#dc9b5f','#5e94bc','#5e94bc','#378c77'])
    ax.set_xticks(range(len(names)),labels,rotation=25,ha='right');ax.axhline(0,color='k',lw=.5)
    ax.set_ylim(-.4,1.05);ax.set_ylabel('分数与局部失败数的排序相关');ax.set_title(title+'，80个已有候选')
    for j,v in enumerate(vv):ax.text(j,v+(.035 if v>=0 else -.09),f'{v:.2f}',ha='center',fontsize=9)
fig.suptitle('固定候选的评分诊断：相关性不是优化成功率，也不是独立推广',fontsize=13)
fig.tight_layout(rect=(0,0,1,.92));fig.savefig(FIG/'area_components.png',dpi=160);plt.close(fig)

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
manifest=read(DATA.parent/'MANIFEST.json');inputhash={}
for name,h in manifest['hashes'].items():
    p=CODE.parents[1]/name
    assert p.exists() and sha(p)==h,('source input hash',name)
    inputhash[name]=h
jobs=[]
for ident,methods in [('mixed_resonators/base',CFG['methods']),('mixed_resonators/lc_reciprocal_8p0',CFG['methods']),('case1',['D','H','A','D_H','D_A','H_A','D_H_A'])]:
    for m in methods:
        for s in CFG['seeds']:
            dd=DATA/ident/m/f'seed_{s}';su=read(dd/'summary.json');au=read(dd/'audit.json')
            gp=dd/'generation_0300.npz';assert sha(gp)==su['files_sha256'][gp.name]
            assert sha(dd/'audited_curve.npz')==au['curve_sha256']
            jobs.append(dict(target=ident,method=m,seed=s,population_sha256=sha(gp),audit_sha256=sha(dd/'audit.json')))
provenance=dict(scope='Postprocessing only; no optimizer, no server connection',jobs_verified=len(jobs),
    source_input_hashes=inputhash,jobs=jobs,analysis_hashes={p.name:sha(p) for p in OUT.glob('*.py')},
    numerical_checks='34 descriptor decompositions; independent node response, rational pole/zero roots, reconstruction; original 1% feature tolerance retained',
    local_compute_note='First all-56 dense-anchor plan was intentionally shortened after the first eight D seeds; all56 coarse/branch checks retained, four stratified deep cases plus eight initial D deep cases.')
(OUT/'PROVENANCE.json').write_text(json.dumps(provenance,ensure_ascii=False,indent=2),encoding='utf8')
print('Saved three scientific figures; verified source inputs and',len(jobs),'saved populations/curves',flush=True)
