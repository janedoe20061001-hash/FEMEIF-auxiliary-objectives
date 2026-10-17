"""Human-readable target definitions derived from the frozen model adapters."""
from core import *
from circuits import expression

def main():
    names={"case1":"Case1基准", "ladder8":"梯形网络倍率8", "mixed0p25":"混合支路倍率0.25", "control0p25":"单并联RLC窄峰对照", "three0p25":"三级网络倍率0.25"}
    diagrams={
        "three0p25":ROOT/"output/mse_structure_complex_20261010_server/retrieved/other/inputs_core/three_level/lc_reciprocal_0p25/circuit.svg",
        "case1":ROOT/"output/mse_structure_step1_20261008/schematics/case1.svg",
        "ladder8":ROOT/"output/mse_structure_complex_20261010_server/retrieved/other/inputs_core/rlc_ladder/lc_reciprocal_8p0/circuit.svg",
        "mixed0p25":ROOT/"output/mse_structure_complex_20261010_server/retrieved/other/inputs_core/mixed_resonators/lc_reciprocal_0p25/circuit.svg",
        "control0p25":ROOT/"output/mse_structure_complex_20261010_server/retrieved/first/inputs_core/control_rlc/lc_reciprocal_0p25/circuit.svg"}
    note="# 第三步使用的目标电路、参数和检查窗口\n\n以下直接从冻结模型读取。R是电阻（Ω），L是电感（H），C是电容（F）；+表示串联，//表示并联。图是普通元件符号的电路图，参数表是生成目标的真值，不是优化后的拟合参数。\n\n"
    note+="倍率名称沿用第二步：指定L乘倍率、C除以倍率，LC乘积保持不变。它不保证端口阻抗的峰宽、峰高或中心频率不变；不要把倍率8直接理解成峰宽变8倍。本轮每一组搜索中，所有元件参数仍同时未知。\n\n"
    definitions=[]
    for ident in CFG["targets"]:
        d=target(ident); c=d["circuit"]; g=Geometry(d)
        assert diagrams[ident].exists()
        note+=f'## {names[ident]}\n\n![{names[ident]}普通电路图]({diagrams[ident].as_posix()})\n\n连接表达式：`{expression(c.tree)}`。未知参数共{len(c.names)}个。\n\n'
        note+="|元件|目标真值（SI单位）|搜索下限|搜索上限|\n|---|---:|---:|---:|\n"
        for n,p,lo,hi in zip(c.names,c.p,10**d["lower"],10**d["upper"]):
            unit={"R":"Ω","L":"H","C":"F"}[n[0]]
            note+=f'|{n} / {unit}|{p:.10g}|{lo:.10g}|{hi:.10g}|\n'
        note+="\n|重点事件|目标中心 / Hz|目标3 dB宽度 / Hz|几何窗口下限 / Hz|上限 / Hz|固定阻抗尺度S / Ω|\n|---|---:|---:|---:|---:|---:|\n"
        for w in g.windows:
            e=next(e for e in d["catalog"] if e["event_id"]==w["event"])
            note+=f'|{w["event"]}|{10**e["x"]:.9g}|{e["width_3db_hz"]:.9g}|{10**w["lo"]:.9g}|{10**w["hi"]:.9g}|{w["scale"]:.9g}|\n'
        note+="\n宽度是目标曲线从峰顶向下3 dB、或从谷底向上3 dB的两个交点之间的频率差。检查窗口通常比这个宽度大，遇到邻近事件还会被裁剪。\n\n"
        definitions.append(dict(target=ident,tree=c.tree,names=c.names,parameters=c.p,lower_log10=d["lower"],upper_log10=d["upper"],windows=g.description()))
    note+="所有目标都是无噪声理论电路，真参数在搜索范围内，存在零误差解。本轮没有使用Case2实测数据来证明几何指标对现实噪声有效。\n"
    (OUT/"TARGETS.md").write_text(note,encoding="utf-8")
    # Fixed training target samples are already reconstructible from core.py.
    dump(OUT/"TARGET_DEFINITIONS.json",definitions)
    print("Target circuit and parameter note generated",flush=True)

if __name__=="__main__":
    main()
