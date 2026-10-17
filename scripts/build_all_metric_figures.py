"""Controlled worked examples for the retained objectives; original illustrations.

Outputs are teaching examples, not optimization observations. Most scores call
the existing objectives. II-08/09 use explicit event-layer examples and the same
region/matching helpers, with a separate label in the manifest.
"""
from pathlib import Path
import importlib.util
import json
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'output/figures/metric_atlas'
OUT.mkdir(parents=True,exist_ok=True)
font_manager.fontManager.addfont('C:/Windows/Fonts/msyh.ttc')
plt.rcParams.update({'font.family':'Microsoft YaHei','font.size':10,
    'axes.unicode_minus':False,'axes.spines.top':False,'axes.spines.right':False,
    'text.color':'#173247','axes.labelcolor':'#173247','axes.titleweight':'bold',
    'svg.fonttype':'none','savefig.facecolor':'white'})
C,G,O,V='#203b50','#168c8b','#df7d2e','#8b60a9'

def load(name,rel):
    spec=importlib.util.spec_from_file_location(name,ROOT/rel)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
pdf=load('atlas_pdf','experiments/case2_dual_pdf18/objectives.py')
paper=load('atlas_paper','experiments/case2_paper5_combinations/objectives.py')
names={row[0]:row[1] for row in pdf.CATALOG}
names.update({row[0]:row[1] for row in paper.CATALOG})
manifest={}

def amp(z):return 20*np.log10(np.maximum(abs(z),1e-300))
def polar(a,phase=0):return 10**(np.asarray(a)/20)*np.exp(1j*np.asarray(phase))
def response(x):
    w=2*np.pi*10**np.asarray(x)
    return 1/(1/120+1j*(w/(4*np.pi**2*1e8*.001)-1/(w*.001)))
def style(ax):ax.grid(alpha=.14);ax.tick_params(labelsize=9)
def panels(n=2,height=4.2):return plt.subplots(1,n,figsize=(11.5,height),layout='constrained')
def score(id,x,z,zc):
    model=(paper if id.startswith('P-') else pdf).Objectives(10**x,aux=(names[id],),target_z=z)
    return model,model.evaluate_spectra(zc)[:,2]
def store(id,fig,caption,talk,limit,values=None):
    for ax in fig.axes:style(ax)
    stem=id.lower().replace('-','_')
    fig.savefig(OUT/(stem+'.png'),dpi=200,bbox_inches='tight')
    fig.savefig(OUT/(stem+'.svg'),bbox_inches='tight')
    plt.close(fig)
    manifest[id]={'images':[f'figures/metric_atlas/{stem}.png'],
        'caption':caption,'talk':talk,'limit':limit,'values':values or {},
        'example_kind':'controlled_synthetic_spectrum'}
def curves(ax,x,z,cs,labels):
    ax.plot(x,amp(z),c=C,lw=2.2,label='目标')
    for zz,col,label in zip(cs,[G,O,V],labels):ax.plot(x,amp(zz),c=col,lw=1.7,label=label)
    ax.set(xlabel='对数频率 x',ylabel='幅值 / dB');ax.legend(fontsize=8.5)
def bars(ax,vals,labels,ylabel,title):
    ax.bar(np.arange(len(vals)),vals,color=[G,O,V][:len(vals)],width=.6)
    for i,v in enumerate(vals):ax.annotate(f'{v:.4g}',(i,v),xytext=(0,5),textcoords='offset points',ha='center',fontsize=9)
    ax.set(xticks=np.arange(len(vals)),xticklabels=labels,ylabel=ylabel,title=title)
    ax.margins(y=.2)

x=np.linspace(2,6,61)
A=np.interp(x,[2,2.8,3.5,4.3,5,5.8,6],[10,30,12,25,10,22,12])
z=polar(A)
e1=np.full(len(x),2.);e2=np.zeros(len(x));e2[30]=2*np.sqrt(len(x))
zc=np.array([polar(A+e1),polar(A+e2)])
mse=np.mean((amp(zc)-A)**2,axis=1)
np.testing.assert_allclose(mse,[4,4],atol=1e-12)
fig,ax=panels()
curves(ax[0],x,z,zc,['A：每点都差 2 dB','B：一个点误差大'])
ax[0].set_title('(a) 同样的平均平方误差，分布不同')
ax[1].plot(x,e1**2,c=G,label='A 的逐点平方误差')
ax[1].plot(x,e2**2,c=O,label='B 的逐点平方误差')
ax[1].set(xlabel='对数频率 x',ylabel='平方误差 / dB²',title='(b) 将这些高度求平均：两者都是 4')
ax[1].legend(fontsize=8.5)
store('M-01',fig,'均匀偏差和单点尖峰可以得到相同的幅值 MSE；右图显示实际被平均的平方误差。',
    '先在每个频点求 dB 幅值差，平方后平均。大误差惩罚更重，但最后只有一个平均数，相同数值不代表误差分布相同。',
    '主目标描述总体幅值误差，不能单靠它分辨最差点或峰谷错位。',{'mse_db2':mse.tolist()})

phase=.25*np.sin(np.linspace(0,2*np.pi,len(x)))
fig,ax=panels()
ax[0].plot(x,phase,c=C,label='目标相位');ax[0].plot(x,phase+.2,c=G,label='候选相位')
ax[0].set(xlabel='对数频率 x',ylabel='相位 / rad',title='(a) 每点相差 0.2 rad：MSE = 0.04 rad²');ax[0].legend()
q=np.pi-.05;ph=np.array([q,-q]);theta=np.linspace(-np.pi,np.pi,300)
ax[1].plot(np.cos(theta),np.sin(theta),c='#cbd5d9')
ax[1].scatter(np.cos(ph),np.sin(ph),c=[C,O],s=50)
ax[1].annotate('目标：π − 0.05',(-1,.05),xytext=(-.7,.45),arrowprops={'arrowstyle':'->','color':C},fontsize=9)
ax[1].annotate('候选：−π + 0.05',(-1,-.05),xytext=(-.7,-.45),arrowprops={'arrowstyle':'->','color':O},fontsize=9)
ax[1].set(xlim=(-1.2,1.2),ylim=(-1.1,1.1),aspect='equal',xlabel='单位复数实部',ylabel='单位复数虚部',
    title='(b) ±π 分支附近：几何很近，直接相减却很大')
store('M-02',fig,'左图说明逐点相位差的平方平均；右图展示 angle 的 ±π 分支边界。',
    '这项是弧度相位直接相减，再平方平均。右图两个点其实只隔 0.1 rad，但直接相减约为 6.18 rad；当前实现没有对相位差做环绕修正。',
    '跨 ±π 时需检查相位分支，不能把直接相位差与最短圆周相位差当作同一种计算。',{'uniform_mse_rad2':.04,'boundary_direct_mse_rad2':float((2*q)**2)})

u=np.linspace(0,1,101);xx=2+4*u
zt=30+1j*20*np.tanh((u-.5)/.08)
cs=np.array([30+1j*20*np.tanh((u-.54)/.08),30-1j*20*np.tanh((u-.5)/.08)])
ob,v=score('I-03',xx,zt,cs)
fig,ax=panels()
ax[0].plot(u,zt.imag,c=C,label='目标：负到正')
for zz,col,lab in zip(cs,[G,O],['A：同方向、略错位','B：反方向']):ax[0].plot(u,zz.imag,c=col,label=lab)
ax[0].axhline(0,c='#95aab4',lw=1);ax[0].set(xlabel='归一化频率 u',ylabel='电抗 Im Z / Ω',title='(a) 过零方向是否与目标一致');ax[0].legend(fontsize=9)
bars(ax[1],v,['A 同方向','B 反方向'],'未匹配比例','(b) 目标事件来自数据，没有预设次数')
store('I-03',fig,'目标曲线有一次负到正的过零；候选 A 区域和方向都匹配，B 的方向相反。这个“一次”来自本例目标，不是规定数量。',
    '用目标过零划区域，再查每个区域内是否存在同方向候选过零。没匹配到的比例就是损失。',
    '区域内的精确零点偏差和额外过零不直接受罚；目标事件数由数据决定。',{'loss':v.tolist(),'target_crossings':len(ob.cross_events)})

shifted=np.interp(x-2*(x[1]-x[0]),x,A)
missing=np.interp(x,[2,2.8,3.5,5.8,6],[10,30,12,22,12])
peaks_cs=np.array([polar(shifted),polar(missing)])
for id in ['I-06','P-02']:
    ob,v=score(id,x,z,peaks_cs)
    fig,ax=panels()
    curves(ax[0],x,z,peaks_cs,['A：峰谷错位','B：缺少局部峰谷'])
    events=ob.extrema if id=='I-06' else ob.events
    edges=ob.extrema_edges if id=='I-06' else ob.edges
    for j,(lo,hi) in enumerate(zip(edges[:-1],edges[1:])):
        ax[0].axvspan(lo,hi,color=G if j%2 else O,alpha=.07)
    ax[0].scatter(events,np.interp(events,x,A),c=C,s=25,zorder=5)
    ax[0].set_title('(a) 目标峰谷划区域，查同类型候选事件')
    bars(ax[1],v,['A 错位','B 缺失'],'峰谷位置损失','(b) 缺失给固定代价；匹配按位置差计罚')
    nearest=id=='I-06'
    store(id,fig,'黑点为目标峰谷；浅色区域按相邻目标事件的中点划分。比较位置错位与结构缺失。',
        ('I-06 在每个区域只取最近的同类峰谷，计算指数位置代价，最后对目标事件取平均。' if nearest else
         'P-02 对每个区域内全部同类峰谷的指数位置代价求和；缺失也给惩罚，最终不除以目标事件数。'),
        ('0.2 dB 邻点突出量和 0.41 惩罚系数为当前设置，未证明最优；不比较峰高或谷深。' if nearest else
         '不比较峰高、谷深；可能忽略没有同类匹配的反类型事件。0.41 为当前设置。'),{'loss':v.tolist()})

xr=np.linspace(2,6,81);zr=response(xr);compressed=response(4+1.3*(xr-4))
ob,v=score('II-02',xr,zr,[compressed])
fig,ax=panels()
curves(ax[0],xr,zr,[compressed],['候选：转变区更窄']);ax[0].set(xlim=(3.25,4.75),title='(a) 检查转变是否过快、过慢')
ax[1].plot(xr,abs(ob.g),c=C,label='目标速度')
ax[1].plot(xr,abs(pdf.derivative(compressed,xr)),c=O,label='候选速度')
ax[1].set(xlim=(3.25,4.75),xlabel='对数频率 x',ylabel='复导数的模 / (Ω / decade)',title=f'(b) 相对速度误差 = {v[0]:.4f}');ax[1].legend()
store('II-02',fig,'幅值图帮助看到过渡宽度；右图画实际使用的复导数模，候选压窄后速度发生变化。',
    '对完整复阻抗求 log 频率导数，取模得到变化速度；两者速度差除以目标速度加稳定项，再平均。',
    '只比较速度大小，不比较方向；0.05 稳定项比例需要独立检验。',{'loss':v.tolist()})

def route(cost,window=None):
    n,m=cost.shape;D=np.full((n+1,m+1),np.inf);D[0,0]=0
    prev={}
    for i in range(1,n+1):
        for j in range(1,m+1):
            if window is not None and abs(i-j)>window:continue
            opts=[(i-1,j-1),(i-1,j),(i,j-1)]
            q=opts[int(np.argmin([D[a,b] for a,b in opts]))]
            D[i,j]=cost[i-1,j-1]+D[q];prev[i,j]=q
    p=[];cur=(n,m)
    while cur!=(0,0):p.append((cur[0]-1,cur[1]-1));cur=prev[cur]
    return D[n,m],np.array(p[::-1])

xd=np.linspace(2,6,41);zd=response(xd);zs=response(xd-.14)
ob,v=score('II-03',xd,zd,[zs]);pp=ob.p;qq=ob.path(zs[None])[0]
cost=np.linalg.norm(pp[:,None,:]-qq[None,:,:],axis=-1)
total,path=route(cost,4);np.testing.assert_allclose(total/len(path),v[0],rtol=1e-12)
fig,ax=panels()
curves(ax[0],xd,zd,[zs],['候选：局部错位'])
for i,j in path[::5]:ax[0].plot([xd[i],xd[j]],[amp(zd)[i],amp(zs)[j]],c='#91a6ae',lw=.8)
ax[0].set_title('(a) 连线展示最优路径选中的频点匹配')
masked=np.where(abs(np.arange(len(xd))[:,None]-np.arange(len(xd))[None,:])<=4,cost,np.nan)
im=ax[1].imshow(masked,origin='lower',cmap='YlGnBu');ax[1].plot(path[:,1],path[:,0],c=O,lw=2)
ax[1].set(xlabel='候选采样索引 j',ylabel='目标采样索引 i',title=f'(b) 窗口内累计最小，再除步数：{v[0]:.4f}')
fig.colorbar(im,ax=ax[1],label='三维局部距离')
store('II-03',fig,'左图连线是三维距离最优路径在幅值图上的投影；右图灰白带外区域不允许匹配，橙线为所选路径。',
    '允许频点在窗口内错位匹配，以累计三维距离最小来选路径，再按路径步数归一化。',
    '窗口 4 是采样索引差，随网格改变含义；图不能证明该窗口最优。',{'loss':v.tolist(),'path_steps':len(path)})

ob,v=score('II-04',xd,zd,[zs]);p=ob.p[None];q=ob.path(zs[None])
sxy=pdf.band_soft_dtw(p,q)[0]/len(xd)
cor=.5*(pdf.band_soft_dtw(p,p)[0]+pdf.band_soft_dtw(q,q)[0])/len(xd)
div=max(sxy-cor,0);np.testing.assert_allclose(np.sqrt(div),v[0])
aa=np.linspace(-.2,1,180);b,c=.35,.9;pred=np.stack([aa,np.full_like(aa,b),np.full_like(aa,c)],1)
mm=pred.min(1);soft=mm-.1*np.log(np.exp(-(pred-mm[:,None])/.1).sum(1))
fig,ax=panels()
ax[0].plot(aa,mm,c=C,label='普通 min');ax[0].plot(aa,soft,c=G,label='平滑 min，γ=0.10')
ax[0].set(xlabel='第一条前驱代价 a',ylabel='选择后的前驱代价',title='(a) 最小值附近，多条前驱都产生影响');ax[0].legend()
bars(ax[1],[sxy,cor,div],['交叉代价/N','自相似校正','校正后散度'],'归一化累计代价',f'(b) 校正后再开方：II-04 = {v[0]:.4f}')
store('II-04',fig,'左图固定另外两个前驱代价，展示普通 min 与平滑 min 的区别；右图展示交叉代价减去自相似项。',
    'Soft-DTW 平滑地综合接近最优的路径。先算目标与候选的代价，再减去双方自匹配代价的一半，按点数归一化并开方。',
    '软代价本身可为负，因此不直接当距离。γ=0.10 和窗口 10 都是当前设置。',{'loss':v.tolist(),'cross_per_point':float(sxy),'self_correction':float(cor)})

spike=zr.copy();spike[40]+=90+70j
ob,v=score('II-05',xr,zr,[spike]);pq=ob.path(spike[None])[0]
dist=np.linalg.norm(ob.p[:,None,:]-pq[None,:,:],axis=-1)
d1=dist.min(1);d2=dist.min(0)
fig=plt.figure(figsize=(11.5,4.6),layout='constrained');ax0=fig.add_subplot(121,projection='3d');ax1=fig.add_subplot(122)
ax0.plot(*ob.p.T,c=C,label='目标');ax0.scatter(*pq.T,c=O,s=9,label='候选（一个离群点）')
for i in range(0,len(xr),12):
    j=int(dist[i].argmin());ax0.plot(*np.stack([ob.p[i],pq[j]]).T,c=G,lw=1)
ax0.set(xlabel='u',ylabel='Re Z / sR',zlabel='Im Z / sI',title='(a) 真实三维最近邻连线');ax0.legend(fontsize=8)
ax1.plot(np.sort(d1),c=C,label='目标 → 候选');ax1.plot(np.sort(d2),c=O,label='候选 → 目标')
ax1.axvline(.95*(len(xr)-1),c=G,ls='--',label='95% 分位位置')
ax1.set(xlabel='从小到大排序的距离编号',ylabel='最近邻距离',title=f'(b) II-05 = {v[0]:.3g}；最坏点仍可很远');ax1.legend(fontsize=8.5)
store('II-05',fig,'一个离群点占比不足 5%，95% 分位距离在本例为零，但最坏点距离很大。左图距离包含 u、实部和虚部三个坐标。',
    '两个方向都找最近邻，各取 95% 分位数，再取较大的一个。它主要看大部分路径点有没有近邻。',
    '忽略最差尾部是设计选择，不代表异常点已修复；不按序匹配。95% 尚需敏感性验证。',{'loss':v.tolist(),'maximum_nearest_distance':float(max(d1.max(),d2.max()))})

uu=(xr-xr[0])/(xr[-1]-xr[0]);cs=np.array([zr+2*np.exp(2j*np.pi*k*uu) for k in [2,15]])
ob,v=score('II-06',xr,zr,cs);paths=ob.path(cs)
e0=np.mean(np.sum((paths-ob.p)**2,axis=2),axis=1)
e1=np.mean(np.sum((pdf.derivative(paths,ob.u,axis=1)-ob.du)**2,axis=2),axis=1)/ob.du_energy
np.testing.assert_allclose(v,np.sqrt(e0+e1));np.testing.assert_allclose(e0[0],e0[1],rtol=1e-12)
fig,ax=panels()
for zz,col,lab in zip(cs,[G,O],['A：低频率波动残差','B：高频率波动残差']):ax[0].plot(xr,(zz-zr).real,c=col,label=lab)
ax[0].set(xlabel='对数频率 x',ylabel='实部残差 / Ω',title='(a) 残差大小相同，变化速度不同');ax[0].legend(fontsize=9)
ax[1].bar([0,1],e0,color=C,label='位置项 E0');ax[1].bar([0,1],e1,bottom=e0,color=O,label='导数项 E1')
ax[1].set(xticks=[0,1],xticklabels=['A 平缓','B 快速'],ylabel='E0 + E1',title='(b) 快速波动被导数项识别');ax[1].legend()
store('II-06',fig,'两个候选的归一化位置平方误差相同，快速波动者导数误差更大；最终对两项之和开方。',
    '先要求点的位置接近，再要求一阶导数接近。即使点误差相同，快速抖动也会得到更高损失。',
    '导数项容易受噪声和采样影响；与主 MSE、速度误差可能重叠。',{'loss':v.tolist(),'E0':e0.tolist(),'E1':e1.tolist()})

cs=np.array([zr+18+9j,zr+18+9j+.15*zr])
ob,v=score('II-07',xr,zr,cs)
fig,ax=panels()
for a,title,correct in zip(ax,['(a) 平移前：两条候选都偏离目标','(b) 去掉中位数平移，仍保留形变'],[False,True]):
    a.plot(zr.real/ob.tr_scales[0],zr.imag/ob.tr_scales[1],c=C,label='目标')
    for zz,col,lab in zip(cs,[G,O],['A：只平移','B：平移 + 形变']):
        vec=np.stack([zz.real/ob.tr_scales[0],zz.imag/ob.tr_scales[1]],1)
        base=np.stack([zr.real/ob.tr_scales[0],zr.imag/ob.tr_scales[1]],1)
        if correct:vec-=np.median(vec-base,axis=0)
        a.plot(*vec.T,c=col,ls='--',label=lab)
    a.set(xlabel='归一化实部',ylabel='归一化虚部',title=title);a.legend(fontsize=8.5)
store('II-07',fig,'左图含整体平移，右图按实部、虚部残差中位数去掉平移。A 与目标重合；B 仍有形状差异。',
    '把整条候选相对目标的恒定偏移先扣除，计算剩余残差长度的平均。',
    '纯平移可得到零损失，即使绝对阻抗差很大；要结合主目标使用。',{'loss':v.tolist()})

events=np.array([.2,.5,.8]);edges=pdf.region_edges(events)
candidate=np.array([.23,.75]);masks=pdf.region_matches(candidate,edges)
local=[.1*min(min(abs(candidate[mask]-tar))/(hi-lo),1) if mask.any() else 1
       for tar,lo,hi,mask in zip(events,edges[:-1],edges[1:],masks)]
fig,ax=panels()
for j,(lo,hi) in enumerate(zip(edges[:-1],edges[1:])):ax[0].axvspan(lo,hi,color=[G,O,V][j],alpha=.10)
ax[0].scatter(events,np.ones(3),c=C,label='目标转折');ax[0].scatter(candidate,np.zeros(2),c=G,label='候选转折')
for tar,mask in zip(events,masks):
    if mask.any():pos=candidate[mask][np.argmin(abs(candidate[mask]-tar))];ax[0].plot([tar,pos],[1,0],c=G)
ax[0].text(.5,.43,'缺失',ha='center',color=O)
ax[0].set(xlim=(0,1),ylim=(-.3,1.3),yticks=[0,1],yticklabels=['候选','目标'],xlabel='归一化频率 u',title='(a) 分区后匹配最近候选转折');ax[0].legend(fontsize=8)
bars(ax[1],local,['区域 1','区域 2','区域 3'],'局部代价',f'(b) 三个区域取平均：{np.mean(local):.4f}')
store('II-08',fig,'事件层示例：目标位置构成三个负责区域；中间区域缺失，另两个区域只按最近位置偏差轻罚。',
    '不要求事先达到某个数量。每个实际目标区域都检查：没有事件罚 1，有事件再按相对距离乘 0.10。',
    '0.10 表示存在性比位置更重要，是当前权衡系数；不检查转折强弱和类型。',{'local_cost':local,'loss':float(np.mean(local))})
manifest['II-08']['example_kind']='explicit_event_layer_example'

sets=[np.array([.2,.82]),np.array([.2,.25,.52,.82])]
miss=[];excess=[]
for ee in sets:
    covered=sum(mask.any() for mask in pdf.region_matches(ee,edges))
    miss.append((len(events)-covered)/len(events));excess.append(.1*max(len(ee)-len(events),0)/len(events))
fig,ax=panels()
for j,(lo,hi) in enumerate(zip(edges[:-1],edges[1:])):ax[0].axvspan(lo,hi,color=[G,O,V][j],alpha=.10)
ax[0].scatter(events,np.full(3,2),c=C,label='目标')
for k,(ee,col,lab) in enumerate(zip(sets,[G,O],['A：漏掉一区域','B：全覆盖但有多余事件'])):ax[0].scatter(ee,np.full(len(ee),1-k),c=col,label=lab)
ax[0].set(xlim=(0,1),ylim=(-.3,2.4),yticks=[0,1,2],yticklabels=['B','A','目标'],xlabel='归一化频率 u',title='(a) 以实际目标事件为参照')
ax[1].bar([0,1],miss,color=C,label='缺失比例');ax[1].bar([0,1],excess,bottom=miss,color=O,label='多余数量轻罚')
ax[1].set(xticks=[0,1],xticklabels=['A 漏掉区域','B 多余事件'],ylabel='两项损失之和',title='(b) 缺失重罚，超过实际目标数才轻罚');ax[1].legend(fontsize=9)
store('II-09',fig,'事件层示例：目标有三个事件是本例数据事实，不是门槛。A 漏一区域，B 全覆盖但总数多一个。',
    '缺失比例看未覆盖区域，轻罚项看候选总数是否超过目标实际总数；两项相加。',
    '总数比较不能保证一一配对。0.10 是当前轻罚权重，尚需检验。',{'missing':miss,'excess':excess,'loss':(np.array(miss)+excess).tolist()})
manifest['II-09']['example_kind']='explicit_event_layer_example'

ob=pdf.Objectives(10**xr,aux=(names['II-10'],),target_z=zr)
cs=np.array([zr+.035*ob.g,zr+1j*.035*ob.g]);v=ob.evaluate_spectra(cs)[:,2]
rr=cs-zr;shift=np.real(rr*np.conj(ob.g))/ob.den;rem=rr-ob.g*shift
fig,ax=panels()
ax[0].plot(xr,abs(rr[0])/ob.sz,c=C,ls=':',label='原始残差（两者相同）')
ax[0].plot(xr,abs(rem[0])/ob.sz,c=G,label='A：扣掉沿切线部分')
ax[0].plot(xr,abs(rem[1])/ob.sz,c=O,label='B：垂直部分仍保留')
ax[0].set(xlabel='对数频率 x',ylabel='残差模 / 目标典型阻抗',title='(a) II-01 解释的频移分量被扣除');ax[0].legend(fontsize=8.5)
bars(ax[1],v,['A 沿切线','B 垂直切线'],'II-10 相对形变','(b) 对剩余残差模取平均')
store('II-10',fig,'沿切线与垂直切线的残差大小相同，扣除频移分量后，垂直形变仍被保留。',
    '先按 II-01 算局部频移，再从复残差扣掉 gδx；剩余长度按目标典型阻抗归一化。',
    '稳定项使切向残差不能被完全去掉；曲线相差大时不能严格解释为物理法向误差。',{'loss':v.tolist()})

cs=np.array([polar(A+4),polar(40-A)])
ob,v=score('P-01',x,z,cs)
fig,ax=plt.subplots(2,2,figsize=(11.5,7),layout='constrained')
curves(ax[0,0],x,z,cs,['A：整体偏高 4 dB','B：趋势反转']);ax[0,0].set_title('(a) 幅值差与趋势差是两个判断')
pat=paper.patterns(amp(cs),x,paper.DEFAULTS)
ax[0,1].plot(x[1:-1],ob.shape,c=C,label='目标编码');ax[0,1].plot(x[1:-1],pat[1],c=O,label='B 编码')
ax[0,1].set(xlabel='对数频率 x',ylabel='趋势编码 s',title='(b) 正负表示方向，大小表示趋势变化');ax[0,1].legend(fontsize=8)
ks=np.linspace(-20,20,101);old,new=np.meshgrid(ks,ks,indexing='ij')
tr=np.stack([np.zeros_like(old),old,old+new],axis=-1)
heat=paper.patterns(tr,[0,1,2],paper.DEFAULTS)[...,0]
im=ax[1,0].imshow(heat,origin='lower',extent=[-20,20,-20,20],cmap='RdYlBu',vmin=-1.6,vmax=1.6,aspect='auto')
for threshold in [-8,8]:ax[1,0].axvline(threshold,c='#738c95',ls='--');ax[1,0].axhline(threshold,c='#738c95',ls='--')
ax[1,0].set(xlabel='后一段斜率',ylabel='前一段斜率',title='(c) 当前分段编码规则');fig.colorbar(im,ax=ax[1,0],label='s')
bars(ax[1,1],v,['A 偏高','B 反转'],'SS 损失','(d) 相同趋势即使偏高，也可得到零损失')
store('P-01',fig,'按论文 SS 的曲线对比与斜率色彩图画法，展示幅值偏移、趋势编码、编码规则与最终评分。',
    '先求相邻 dB 斜率，再由连续两段斜率编码趋势；比较目标和候选编码的绝对差并平均。',
    '幅值整体偏移不改变斜率，可得到零 SS；T=8、β=0.4、μ=0.5 为当前设置。',{'loss':v.tolist()})

ob,v=score('P-03',x,z,[polar(shifted)])
aa=ob.ad;bb=np.interp(ob.dtw_x,x,shifted);cost=abs(bb[:,None]-aa[None,:])
total,path=route(cost);np.testing.assert_allclose(total,v[0],rtol=1e-12)
fig,ax=panels()
ax[0].plot(ob.dtw_x,aa,c=C,label='目标 60 点');ax[0].plot(ob.dtw_x,bb,c=G,label='候选 60 点')
for i,j in path[::7]:ax[0].plot([ob.dtw_x[i],ob.dtw_x[j]],[bb[i],aa[j]],c='#92a7ae',lw=.8)
ax[0].set(xlabel='对数频率 x',ylabel='幅值 / dB',title='(a) 先插值到 60 点，再允许错位匹配');ax[0].legend(fontsize=8)
im=ax[1].imshow(cost,origin='lower',cmap='YlGnBu');ax[1].plot(path[:,1],path[:,0],c=O,lw=2)
ax[1].set(xlabel='目标索引 j',ylabel='候选索引 i',title=f'(b) 无窗口，返回累计代价：{v[0]:.3f}');fig.colorbar(im,ax=ax[1],label='局部绝对幅值差 / dB')
store('P-03',fig,'沿用论文 DTW 的对齐连线与矩阵路径画法；距离只使用 dB 幅值。',
    '目标和候选都在 log 频率上插值成 60 点，逐格计算幅值差，再找累计代价最小的路径。',
    '无窗口，不比较相位，也不除路径长度；60 点影响计算成本和细节保留，需敏感性验证。',{'loss':v.tolist(),'path_steps':len(path)})

ob=paper.Objectives(10**x,aux=('LFP',),target_z=z)
groups=np.split(np.arange(len(x)),ob.splits);n=len(groups)
errA=np.zeros(len(x));errB=np.zeros(len(x))
for j,ids in enumerate(groups):errA[ids]=.5 if j<min(3,n-1) else 2.
for j,ids in enumerate(groups):errB[ids]=2. if j<min(2,n) else .5
# Equalize global MSE without changing the ordering of high vs low-error zones.
errB*=np.sqrt(np.mean(errA**2)/np.mean(errB**2))
cs=np.array([polar(A+errA),polar(A+errB)]);v=ob.evaluate_spectra(cs)[:,2]
zone=np.array([[np.mean(e[ids]**2) for ids in groups] for e in [errA,errB]])
np.testing.assert_allclose(np.mean(errA**2),np.mean(errB**2))
fig,ax=panels()
curves(ax[0],x,z,cs,['A：低频先拟合好','B：高频先拟合好'])
for edge in ob.edges[1:-1]:ax[0].axvline(edge,c='#afbdc3',ls=':',lw=1)
ax[0].set_title('(a) 总体幅值 MSE 相同，误差频段不同')
j=np.arange(n);ax[1].bar(j-.17,zone[0],width=.34,color=G,label='A');ax[1].bar(j+.17,zone[1],width=.34,color=O,label='B')
ax[1].axhline(1.6,c=C,ls='--',label='阈值 1.6 dB²')
ax[1].set(xticks=j,xticklabels=[str(k+1) for k in j],xlabel='由低频到高频的目标区域',ylabel='本段 MSE / dB²',title=f'(b) 首次失败后停止计数：损失 {v[0]:.2g} / {v[1]:.2g}');ax[1].legend(fontsize=8)
store('P-04',fig,'两个候选总体幅值 MSE 相同；A 从低频连续达标，B 低频先失败，高频好段不能补回。',
    '按实际目标峰谷分段，只累计从低频开始连续低于误差阈值的段数，再除以总段数转成损失。',
    '没有规定必须达标几段，但 1.6 dB² 仍是经验阈值，需要验证；分段也影响结果。',{'loss':v.tolist(),'zone_mse_db2':zone.tolist(),'global_mse_db2':float(np.mean(errA**2))})

ob,v=score('P-05',x,z,zc)
fig,ax=panels()
curves(ax[0],x,z,zc,['A：每点都差 2 dB','B：只有一处大尖峰'])
ax[0].set_title('(a) 幅值 MSE 都是 4，最差点不同')
bars(ax[1],v,['A 均匀偏差','B 单点尖峰'],'最大幅值误差 / dB','(b) 只取逐点绝对误差的最大值')
store('P-05',fig,'复用 M-01 的等 MSE 示例，最大误差分别为 2 和约 15.62 dB，能看出单点严重失配。',
    '不把所有点平均，只看最差那个频点有多大幅值偏差。',
    '单点噪声或毛刺也能决定指标，因此不能说这个公式本身抗噪。',{'loss':v.tolist(),'mse_db2':mse.tolist()})

manifest['II-01']={
    'images':['figures/ii01_explanation/'+s+'.png' for s in
              ['01_frequency_shift','02_complex_projection','03_weighted_accumulation','04_direction_counterexample']],
    'caption':'特征左右错位 → 复平面切线投影 → 加权平均 → 垂直形变反例。',
    'talk':'顺着目标切线的残差可以近似解释为局部频移；对各点频移绝对值加权平均。',
    'limit':'右移对应负的 δx；一阶局部解释不能代替精确峰位检测，垂直形变可能不计入。',
    'values':json.loads((ROOT/'output/figures/ii01_explanation/example_values.json').read_text(encoding='utf-8')),
    'example_kind':'controlled_synthetic_spectrum'}
selection=json.loads((ROOT/'scripts/metric_catalog_selection.json').read_text(encoding='utf-8'))
assert set(manifest)==set(selection['retained'])
(OUT/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
sys.stdout.reconfigure(encoding='utf-8')
print(json.dumps({id:m['values'] for id,m in manifest.items()},ensure_ascii=False,indent=2))
