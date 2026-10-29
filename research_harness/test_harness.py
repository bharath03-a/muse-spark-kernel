import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from .stats import paired_summary, weighted_load_speedup
from . import loop
from .environment import configure_h200_cache
import os


class StatisticsTests(unittest.TestCase):
    def test_cpu_gpu_cache_arch_consistent(self):
        with patch.dict(os.environ, {'TORCH_CUDA_ARCH_LIST': '9.0'}):
            env = configure_h200_cache('/test-cache')
            self.assertEqual(os.environ['TORCH_CUDA_ARCH_LIST'], '9.0a')
            self.assertEqual(env['SGLANG_JIT_CACHE_DIR'], '/test-cache/sglang_jit')

    def test_order_bias_cancels_in_pairs(self):
        samples = [{'baseline_us': 110, 'candidate_us': 50},
                   {'baseline_us': 100, 'candidate_us': 55}] * 10
        report = paired_summary(samples)
        self.assertAlmostEqual(report['speedup'], 2)
        self.assertEqual(report['decision'], 'promising')
        self.assertLessEqual(report['speedup_ci95'][0], 1.97)

    def test_near_tie_not_promoted(self):
        report = paired_summary([{'baseline_us': 100.1, 'candidate_us': 100}] * 20)
        self.assertEqual(report['decision'], 'inconclusive')

    def test_invalid_samples_rejected(self):
        for value in (0, -1, float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                paired_summary([{'baseline_us': 100, 'candidate_us': value}] * 8)
        with self.assertRaises(ValueError):
            paired_summary([{'baseline_us': 4, 'candidate_us': 3}] * 8, replay_overhead_us=4)

    def test_load_weights_not_shape_geomean(self):
        rows = [dict(load='decode', weight=10, baseline_us=100, candidate_us=50),
                dict(load='decode', weight=1, baseline_us=100, candidate_us=200),
                dict(load='verify', weight=1, baseline_us=100, candidate_us=100)]
        self.assertAlmostEqual(weighted_load_speedup(rows), (1100 / 700) ** .5)


class QueueTests(unittest.TestCase):
    def test_completed_source_not_repeated_and_changes_invalidate(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(loop, 'ROOT', Path(tmp)):
            root = Path(tmp)
            source = root / 'candidate.py'
            source.write_text('version=1')
            task = dict(id='smoke', command=['python3', '-c', 'print("PASS")'], sources=['candidate.py'],
                        success_markers=['PASS'], hypothesis='Queue smoke test')
            manifest = root / 'queue.json'
            manifest.write_text(json.dumps(dict(tasks=[task])))
            self.assertEqual(loop.run_queue(manifest), 0)
            self.assertEqual(loop.run_queue(manifest), 0)
            journal = root / 'research_harness/results/journal.jsonl'
            self.assertEqual(len(journal.read_text().splitlines()), 2)
            record = json.loads(journal.read_text().splitlines()[-1])
            self.assertEqual((Path(record['source_snapshot']) / 'candidate.py').read_text(), 'version=1')
            source.write_text('version=2')
            self.assertEqual(loop.run_queue(manifest), 0)
            self.assertEqual(len(journal.read_text().splitlines()), 4)

    def test_exit_zero_without_results_is_failure(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(loop, 'ROOT', Path(tmp)):
            root = Path(tmp)
            (root / 'candidate.py').write_text('')
            manifest = root / 'queue.json'
            manifest.write_text(json.dumps(dict(tasks=[dict(id='empty', command=['python3', '-c', 'pass'],
                sources=['candidate.py'], success_markers=['PASS'], hypothesis='Missing result') ])))
            self.assertEqual(loop.run_queue(manifest), 1)
            self.assertEqual(loop.run_queue(manifest), 1)

    def test_source_mutation_during_run_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(loop, 'ROOT', Path(tmp)):
            root = Path(tmp)
            (root / 'candidate.py').write_text('version=1')
            manifest = root / 'queue.json'
            manifest.write_text(json.dumps(dict(tasks=[dict(id='mutation', command=['python3', '-c',
                'from pathlib import Path; Path("candidate.py").write_text("version=2"); print("PASS")'],
                sources=['candidate.py'], success_markers=['PASS'], hypothesis='Source integrity') ])))
            self.assertEqual(loop.run_queue(manifest), 1)
            record = json.loads((root / 'research_harness/results/journal.jsonl').read_text().splitlines()[-1])
            self.assertEqual(record['status'], 'source_changed_during_run')

    def test_interrupted_dispatch_never_silently_restarts(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(loop, 'ROOT', Path(tmp)):
            root = Path(tmp)
            (root / 'candidate.py').write_text('version=1')
            task = dict(id='interrupted', command=['python3', '-c', 'print("PASS")'],
                        sources=['candidate.py'], success_markers=['PASS'], hypothesis='No duplicate dispatch')
            out = root / 'research_harness/results'
            out.mkdir(parents=True)
            (out / 'journal.jsonl').write_text(json.dumps(dict(id=task['id'],
                fingerprint=loop.fingerprint(task), status='started')) + '\n')
            manifest = root / 'queue.json'
            manifest.write_text(json.dumps(dict(tasks=[task])))
            with patch.object(loop.subprocess, 'Popen') as launch:
                self.assertEqual(loop.run_queue(manifest), 1)
                launch.assert_not_called()


if __name__ == '__main__':
    unittest.main()
