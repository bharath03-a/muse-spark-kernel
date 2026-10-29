import copy
import json
from pathlib import Path
import tempfile
import unittest
from bench.experiment_loop import load_plan

class PlanTests(unittest.TestCase):
    def setUp(self):
        self.plan=json.loads(Path('bench/loop_manifest.json').read_text())
    def load(self,plan):
        with tempfile.NamedTemporaryFile(mode='w',suffix='.json') as f:
            json.dump(plan,f);f.flush();return load_plan(f.name)
    def test_real_sources_pinned(self):
        self.assertEqual(self.load(self.plan),self.plan)
    def test_changed_sources_rejected(self):
        self.plan['baseline']['sha256']='0'*64
        with self.assertRaises(ValueError):self.load(self.plan)
    def test_unbalanced_plan_rejected(self):
        self.plan['reps']=9
        with self.assertRaises(ValueError):self.load(self.plan)
    def test_single_cohort_rejected(self):
        self.plan['cohorts']=1
        with self.assertRaises(ValueError):self.load(self.plan)
    def test_duplicate_shapes_rejected(self):
        self.plan['ms']=[128,128]
        with self.assertRaises(ValueError):self.load(self.plan)
    def test_invalid_layer_rejected(self):
        self.plan['layers']=['fake']
        with self.assertRaises(ValueError):self.load(self.plan)
    def test_empty_shapes_rejected(self):
        self.plan['ms']=[]
        with self.assertRaises(ValueError):self.load(self.plan)
    def test_default_shapes_materialized(self):
        del self.plan['ms'];del self.plan['layers']
        result=self.load(self.plan)
        self.assertEqual(result['ms'],[1,8,16,32,128,256])
        self.assertEqual(len(result['layers']),5)

if __name__=='__main__':unittest.main()
