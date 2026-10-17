"""Copy verified baseline/singles unchanged; never relabel their source fingerprint."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import shutil
import numpy as np
import experiment as e


def reference_groups(cfg):
    return ['baseline']+list(dict.fromkeys(a for g in cfg['groups'] for a in g))


def validate_compatibility(source, cfg, manifest, expected_fingerprint=None):
    if expected_fingerprint is None:
        expected_fingerprint=json.loads((e.HERE/'SOURCE.json').read_text())['reference_fingerprint']
    if source['fingerprint']!=expected_fingerprint:
        raise ValueError('Unexpected reference source fingerprint')
    fields={k:v for k,v in cfg.items() if k!='groups'}
    if fields!={k:v for k,v in source['config'].items() if k!='groups'}:
        raise ValueError('Reference budget, data, seeds, operators or selection differs')
    for key in ['frequency_hz','target_real','target_imag','parameter_names','lower_log10','upper_log10','versions']:
        if json.loads(json.dumps(manifest[key]))!=source[key]:
            raise ValueError(f'Reference differs: {key}')
    for name in ['case2.py','objectives.py','final_selection.py','requirements-lock.txt']:
        if source['code_sha256'][name]!=manifest['code_sha256'][name]:
            raise ValueError(f'Reference evaluation code differs: {name}')


def validate_reference(root, cfg, manifest, expected_fingerprint=None):
    root=Path(root)
    source=json.loads((root/'manifest.json').read_text())
    validate_compatibility(source,cfg,manifest,expected_fingerprint)
    seen=set()
    expected={(g,s) for g in reference_groups(cfg) for s in range(cfg['seeds']['start'],cfg['seeds']['start']+cfg['seeds']['count'])}
    for path in root.glob('*/seed_*/summary.json'):
        r=json.loads(path.read_text());key=r['group'],r['seed']
        if key in seen or key not in expected or r['fingerprint']!=source['fingerprint'] or r['status']!='complete':
            raise ValueError(f'Invalid reference record: {path}')
        seen.add(key)
        initial=np.load(path.parent/'initial_log_parameters.npy')
        np.testing.assert_array_equal(initial,e.initial_population(cfg,r['seed']))
        if hashlib.sha256(initial.tobytes()).hexdigest()!=r['initial_population_sha256']:
            raise ValueError('Corrupt reference initialization')
        if r['evaluations']!=cfg['population']*cfg['generations'] or r['generations']!=cfg['generations']:
            raise ValueError('Reference evaluation budget differs')
        for name in ['final_population.npz','history.csv']:
            if not (path.parent/name).is_file():raise ValueError(f'Missing {name}')
    if seen!=expected:raise ValueError('Missing reference seeds/groups')
    return source


def prepare(source_root,destination,cfg,manifest,expected_fingerprint=None):
    source_root,destination=Path(source_root),Path(destination)
    source=json.loads((source_root/'manifest.json').read_text())
    validate_compatibility(source,cfg,manifest,expected_fingerprint)
    if destination.exists():raise FileExistsError(destination)
    destination.mkdir(parents=True)
    shutil.copy2(source_root/'manifest.json',destination/'manifest.json')
    for group in reference_groups(cfg):
        for seed in range(cfg['seeds']['start'],cfg['seeds']['start']+cfg['seeds']['count']):
            name=Path(group)/f'seed_{seed:03d}'
            shutil.copytree(source_root/name,destination/name)
    validate_reference(destination,cfg,manifest,expected_fingerprint)
    e.dump(destination/'reuse_audit.json',dict(status='passed',reference_fingerprint=source['fingerprint'],
        reused_runs=len(reference_groups(cfg))*cfg['seeds']['count'],original_records_preserved=True))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--source',type=Path,required=True)
    ap.add_argument('--output',type=Path,default=e.HERE/'reference');args=ap.parse_args()
    cfg,f=e.config(e.HERE/'config.json');mf=e.manifest(cfg,f)
    spec=json.loads((e.HERE/'SOURCE.json').read_text())
    nodes={n.name:n for n in ast.parse((e.HERE/'experiment.py').read_text()).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
    for name,digest in spec['search_ast_sha256'].items():
        if hashlib.sha256(ast.dump(nodes[name],include_attributes=False).encode()).hexdigest()!=digest:
            raise ValueError(f'Search framework changed: {name}')
    prepare(args.source,args.output,cfg,mf)
    print('Verified and copied 600 reference runs without changing their records.')


if __name__=='__main__':main()
