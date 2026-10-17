from pathlib import Path
r=Path('experiments/teacher_doubts_validation_20261011');d=Path('experiments/case2_30mhz_20261011');d.mkdir(exist_ok=False)
s=(r/'core.py').read_text()
s=s.replace("OUT=ROOT/'output/teacher_doubts_validation_20261011'","OUT=ROOT/'output/case2_30mhz_20261011'")
s=s.replace("if condition=='paper_band':","if condition in ['paper_band','upto30m']:")
s=s.replace("mask=(m.f>=1e4)&(m.f<=3e7);","mask=(m.f<=3e7) if condition=='upto30m' else (m.f>=1e4)&(m.f<=3e7);")
s=s.replace("assert case=='case2' and len(m.f)==108","assert case=='case2' and len(m.f)==(183 if condition=='upto30m' else 108)")
(d/'core.py').write_text(s,encoding='utf8')
s=(r/'run.py').read_text().split('def preflight():')[0]
(d/'run.py').write_text(s,encoding='utf8')
