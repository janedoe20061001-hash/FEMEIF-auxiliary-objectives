"""Local analytic close-up of two persistent Case1 structures at generation 300."""
from analyze import np,plt,OUT,Model,ind,improved_dense_grid,link
from matplotlib.ticker import FixedLocator,FuncFormatter,NullLocator
def main():
    d=np.load(OUT/'selected/case1_generation_300.npz');m=Model('case1');p=10**d['log_parameters'][:,0];f=improved_dense_grid(m.record,p,2);target=ind.independent_spectra('case1',f,m.record['target_parameters'])[0];z=ind.independent_spectra('case1',f,p)
    fig,axes=plt.subplots(2,2,figsize=(12,6.8),layout='constrained');colors=['#2674ad','#cf741c','#309661']
    for col,(lo,hi,title) in enumerate([(24000,26000,'Narrow target structure: 24-26 kHz'),(1e5,10**5.5,'Residual phase band: 100-316 kHz')]):
        mask=(f>=lo)&(f<=hi)
        for si,s in enumerate(d['seeds']):axes[0,col].semilogx(f[mask],20*np.log10(abs(z[si,mask])),color=colors[si],label=f'seed {s}',lw=1.2);axes[1,col].semilogx(f[mask],np.rad2deg(np.angle(z[si,mask])),color=colors[si],lw=1.2)
        axes[0,col].semilogx(f[mask],20*np.log10(abs(target[mask])),'k--',lw=1.5,label='Target');axes[1,col].semilogx(f[mask],np.rad2deg(np.angle(target[mask])),'k--',lw=1.5);axes[0,col].set_title(title)
        for ax in axes[:,col]:ax.grid(True,alpha=.2);ax.set_xlim(lo,hi)
        axes[1,col].set_xlabel('Frequency (Hz)')
        if col==0:
            for ax in axes[:,col]:
                ax.set_xscale('linear');ax.xaxis.set_major_locator(FixedLocator([24000,24500,25000,25500,26000]));ax.xaxis.set_major_formatter(FuncFormatter(lambda x,_:f'{x/1000:g}'));ax.xaxis.set_minor_locator(NullLocator())
            axes[1,col].set_xlabel('Frequency (kHz)')
    axes[0,0].set_ylabel('Magnitude (dB re 1 ohm)');axes[1,0].set_ylabel('Phase (degree)');axes[0,0].legend(fontsize=8);fig.suptitle('Case1, generation 300: same three random seeds, analytic curves')
    path=OUT/'figures/case1_zoom_300.png';fig.savefig(path,dpi=170);plt.close(fig)
    md=OUT/'case1.md';text=md.read_text(encoding='utf8')
    if '## 末代两个局部结构' not in text:text=text.replace('## 每个seed的阶段诊断',f'## 末代两个局部结构\n\n![局部结构]({link(path)})\n\n左列把约25.23kHz的目标窄峰和相位转变放大；右列对应100–316kHz的主要相位残差频带。图中的极窄拟合尖峰来自解析电路响应，原训练点折线可能看不到。\n\n## 每个seed的阶段诊断')
    md.write_text(text,encoding='utf8')
    print(path)
if __name__=='__main__':main()
