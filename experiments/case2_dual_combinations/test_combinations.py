import ast
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import experiment as e
from prepare_reference import prepare,validate_reference,validate_compatibility,reference_groups
import report_results
from cross_case import compare


class CombinationChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg,cls.f=e.config(e.HERE/'config.json')

    def test_plan_and_unchanged_framework(self):
        self.assertEqual(e.plan(self.cfg)['planned_runs'],800)
        self.assertEqual(e.plan(self.cfg)['planned_evaluations'],36000000)
        self.assertEqual(len(reference_groups(self.cfg)),6)
        self.assertEqual(sum(len(g)==2 for g in e.groups(self.cfg)),6)
        self.assertEqual(sum(len(g)==3 for g in e.groups(self.cfg)),2)
        spec=json.loads((e.HERE/'SOURCE.json').read_text())
        nodes={n.name:n for n in ast.parse((e.HERE/'experiment.py').read_text()).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
        for name,digest in spec['search_ast_sha256'].items():
            self.assertEqual(hashlib.sha256(ast.dump(nodes[name],include_attributes=False).encode()).hexdigest(),digest,name)
        for name in ['case2.py','objectives.py','final_selection.py','requirements-lock.txt']:
            self.assertEqual(hashlib.sha256((e.HERE/name).read_bytes()).hexdigest(),spec['reference_code_sha256'][name])

    def test_duplicate_groups_rejected(self):
        cfg=copy.deepcopy(self.cfg);cfg['groups'].append(list(reversed(cfg['groups'][0])))
        with self.assertRaises(ValueError):e.groups(cfg)

    def test_reference_mismatch_rejected(self):
        m=e.manifest(self.cfg,self.f);source=json.loads(json.dumps(m))
        validate_compatibility(source,self.cfg,m,m['fingerprint'])
        source['config']['final_selection']['phase_weight']=25.12
        with self.assertRaises(ValueError):validate_compatibility(source,self.cfg,m,m['fingerprint'])

    def test_case1_snapshots_cover_both_sets(self):
        rows=[dict(group=e.group_name(g),magnitude_change_pct=-1.,phase_change_pct=-2.,both_means_lower=True) for g in e.groups(self.cfg)]
        with tempfile.TemporaryDirectory() as directory:
            comparisons=compare(directory,rows)
            self.assertEqual(len(comparisons),15)
            self.assertEqual(len({r['group'] for r in comparisons}),8)

    def test_full_reuse_and_reporting(self):
        cfg=copy.deepcopy(self.cfg);cfg.update(population=8,generations=3,seeds=dict(start=0,count=2))
        mf=e.manifest(cfg,self.f)
        source_cfg=dict(cfg,groups=[[] if g=='baseline' else [g] for g in reference_groups(cfg)])
        source_mf=e.manifest(source_cfg,self.f)
        with tempfile.TemporaryDirectory(prefix='case2_combo_check_') as temp:
            base=Path(temp);source=base/'source';source.mkdir();e.dump(source/'manifest.json',source_mf)
            for seed in range(2):
                for group in e.groups(source_cfg):
                    e.run_one(source_cfg,self.f,group,seed,source/e.group_name(group)/f'seed_{seed:03d}',source_mf['fingerprint'])
            reference=base/'reference';prepare(source,reference,cfg,mf,source_mf['fingerprint'])
            with self.assertRaises(FileExistsError):prepare(source,reference,cfg,mf,source_mf['fingerprint'])
            root=base/'runs';root.mkdir();e.dump(root/'manifest.json',mf)
            for seed in range(2):
                for group in e.groups(cfg):
                    e.run_one(cfg,self.f,group,seed,root/e.group_name(group)/f'seed_{seed:03d}',mf['fingerprint'])
            def fixture_validation(ref,c,m):
                return validate_reference(ref,c,m,source_mf['fingerprint'])
            with patch.object(report_results,'validate_reference',side_effect=fixture_validation),patch('sys.argv',['report_results.py',str(root),'--no-archive']):
                report_results.main()
            audit=json.loads((root/'completion_audit.json').read_text())
            self.assertEqual((audit['runs'],audit['new_runs'],audit['reused_runs']),(28,16,12))
            stability=json.loads((root/'stability.json').read_text(encoding='utf-8'))
            constituents=json.loads((root/'constituent_stability.json').read_text(encoding='utf-8'))
            self.assertEqual(len(stability['findings']),8)
            self.assertEqual(stability['settings']['familywise_components'],16)
            self.assertEqual(constituents['component_intervals'],48)
            missing=reference/'baseline/seed_000/summary.json';missing.unlink()
            with self.assertRaises(ValueError):validate_reference(reference,cfg,mf,source_mf['fingerprint'])


if __name__=='__main__':unittest.main(verbosity=2)
