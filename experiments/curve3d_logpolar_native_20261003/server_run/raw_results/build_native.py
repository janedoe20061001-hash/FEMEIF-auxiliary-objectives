"""Build auditable CPU arithmetic, no fast math, no scientific approximation."""
from pathlib import Path
import subprocess,json,hashlib,shutil
root=Path(__file__).resolve().parent
flags=['-O3','-fno-fast-math','-ffp-contract=off','-shared','-fPIC']
command=['gcc',*flags,str(root/'orientation_native.c'),'-lm','-o',str(root/'orientation_native.so')]
subprocess.run(command,check=True)
record=dict(compiler=subprocess.check_output(['gcc','--version'],text=True).splitlines()[0],
            flags=flags,source_sha256=hashlib.sha256((root/'orientation_native.c').read_bytes()).hexdigest(),
            binary_sha256=hashlib.sha256((root/'orientation_native.so').read_bytes()).hexdigest())
for case in ['case1','case2']:
    shutil.copy2(root/'orientation_native.so',root/case/'orientation_native.so')
    (root/case/'build_native.json').write_text(json.dumps(record,indent=2))
(root/'build_native.json').write_text(json.dumps(record,indent=2))
print(json.dumps(record))
