"""Remap raw input paths after the explicitly scoped PowerShell move."""
from reselect import *

def main():
    for batch,name in BATCHES.items():
        old=f'output/{name}/batch/';new=f'output/mse_structure_step3_knee_20261011/source_populations/{batch}/'
        p=OUT/batch/'INPUT_VERIFICATION.json';v=json.loads(p.read_text())
        v['raw_checksums']={k.replace(old,new,1):h for k,h in v['raw_checksums'].items()};v['source_migration']={'original_prefix':old,'current_prefix':new,'bytes_unchanged':True};dump(p,v)
        for file in (OUT/batch/'selections').glob('*.json'):
            s=json.loads(file.read_text());s['original_source']=s['source'];s['source']=s['source'].replace(old,new,1);dump(file,s)
            dfile=OUT/batch/f"analysis/details/{s['target']}_{s['method']}_{s['seed']}_knee.json"
            d=json.loads(dfile.read_text(encoding='utf-8'));d['selection_provenance']=s;dump(dfile,d)
        for readme in [ROOT/'experiments'/name/'README.md']:
            readme.write_text('# 冻结优化源：旧后处理停用\n\n本目录仅保留原优化与审计源作为新版复算依据，不再使用旧加权代表后处理入口。\n\n现行入口：[膝点后处理](D:/develop/FEMEIF/experiments/mse_structure_step3_knee_20261011/README.md)。现行结果：[通俗报告](D:/develop/FEMEIF/output/mse_structure_step3_knee_20261011/FINDINGS.md)。原完整种群已移入新版output/source_populations/'+batch+'；原协议中的selection、旧primary和balanced字段已停用。不要启动原run.py或旧report.py重新生成过时结果。\n',encoding='utf-8')

if __name__=='__main__':main()
