"""Readable parameter tables and conventional circuit diagrams from the frozen trees."""
import importlib.util,sys
from pathlib import Path
spec=importlib.util.spec_from_file_location('physical_catalog_base',Path(__file__).with_name('run.py'))
b=importlib.util.module_from_spec(spec);sys.modules[spec.name]=b;spec.loader.exec_module(b)
import json
draw=b.core.load('physical_catalog_schematic',b.core.NESTED/'schematic.py')

def main():
    names={'simple_series':'简单串联RLC控制','simple_parallel':'简单并联RLC控制','case1':'Case1：六个并联RLC模块串联',
           'ladder8':'两级梯形网络（既有倍率8目标）','mixed0p25':'混合嵌套支路（既有倍率0.25目标）',
           'three0p25':'三级嵌套网络（既有倍率0.25目标）','control0p25':'既有简单并联RLC目标（倍率0.25）'}
    text=['# 本轮的电路与目标参数','',
          '2026-10-11。表和普通电路图均从本轮冻结的目标参数、串并联树自动生成；连接关系通过原电路图工具的校验。',
          '目标倍率是前期已生成的目标条件，本轮没有重新改变原优化目标。下面的参数是干预倍率1的参照；105条干预中的变化参数另见各目标details JSON。','']
    folder=b.OUT/'schematics';folder.mkdir(exist_ok=True)
    for ident,title in names.items():
        data=b.simple(ident) if ident.startswith('simple_') else b.core.target(ident);c=data['circuit']
        original=json.loads((b.OUT/'details'/f'{ident}.json').read_text(encoding='utf-8'))
        assert c.record()==original['target']['circuit']
        draw.save_schematic(c,folder/ident)
        text.extend([f'## {title}','',f'标识：{ident}。','',f'![{title}]({(folder/(ident+".png")).as_posix()})','',
                     '|元件|便于阅读的值|原始SI值|','|---|---:|---:|'])
        for n,v in zip(c.names,c.p):text.append(f'|{n}|{draw.engineering(n,v)}|{v:.12g} {dict(R="Ω",L="H",C="F")[n[0]]}|')
        text.extend(['',f'原干预的电阻元件：{c.damping}；LC干预元件：{c.lc[0]}、{c.lc[1]}。',
                     '这是选定干预元件的名称，不表示该电阻唯一决定全电路阻尼，也不表示这对LC唯一决定端口谐振。',
                     '',f'[完整目标和逐曲线状态]({(b.OUT/"details"/(ident+".json")).as_posix()})',''])
    (b.OUT/'TARGETS.md').write_text('\n'.join(text)+'\n',encoding='utf-8')

if __name__=='__main__':main()
