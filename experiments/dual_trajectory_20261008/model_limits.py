from analyze import Model,np,weights,OUT,link,write,dump
def main():
    m=Model('case2');theta=np.angle(m.target);excess=np.maximum(abs(theta)-np.pi/2,0);mask=excess>0
    result=dict(points=len(theta),outside_passive_phase_points=int(mask.sum()),frequency_hz=m.f[mask].tolist(),phase_deg=np.rad2deg(theta[mask]).tolist(),necessary_training_phase_mse_lower_bound_rad2=float(np.mean(excess**2)),necessary_diagnostic_phase_rmse_lower_bound_deg=float(np.rad2deg(np.sqrt(np.sum(weights(m.f)*excess**2)))))
    dump(OUT/'MODEL_LIMITS.json',result)
    write(OUT/'MODEL_LIMITS.md',['# Case2：先区分模型限制与指标限制',
        'Case2的原始201个目标点中，16MHz处相位为107°，实部为负。当前电路只包含正值R/L/C，其无源输入阻抗的实部非负，相位只能落在−90°到90°之间。因此这个点的相位误差至少为17°；不能宣称这个实测Case存在当前模型下的精确零误差解。',
        f'仅这一个点给等权PHA MSE带来必要下界{result["necessary_training_phase_mse_lower_bound_rad2"]:.8g} rad²；按本轮log f诊断积分，相位RMSE的必要下界约{result["necessary_diagnostic_phase_rmse_lower_bound_deg"]:.5g}°。这是单点限制给出的下界，不是整个电路能达到的最优值。它明显低于当前末代平均相位RMSE约16.95°，不能用它解释全部残差。',
        '这个结果直接来自冻结目标和当前模型的相位范围，尚未判断负实部点来自测量噪声、去嵌误差、数据处理还是目标本身。实验没有删除该点或更改目标。Case2局部阈值事件的漏失可能还包含测量细节和模型可表示性的影响，应与Case1的无噪声解析目标分别判断。',
        f'[冻结数据复算记录]({link(OUT/"MODEL_LIMITS.json")})；[返回结构结论]({link(OUT/"FINDINGS.md")})'])
    for name in ['REPORT.md','FINDINGS.md']:
        p=OUT/name;text=p.read_text(encoding='utf8')
        if 'MODEL_LIMITS.md' not in text:text+=f'\n\n[Case2模型限制：16MHz目标相位107°，不能用正值RLC模型精确拟合]({link(OUT/"MODEL_LIMITS.md")})。这不解释全部残差，但需要避免把模型限制误判成指标限制。\n'
        p.write_text(text,encoding='utf8')
    p=OUT/'ALL_SEEDS.md';text=p.read_text(encoding='utf8').replace('末代残差不能证明双目标不存在零误差解。','Case1的真值在边界内，存在零误差解；Case2为实测目标，需另检查模型可表示性。');p.write_text(text,encoding='utf8')
    print(result)
if __name__=='__main__':main()
