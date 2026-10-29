import json
import math
import random
import unittest
from pathlib import Path

try:
    from .loop_stats import analyze_pairs, score_loads, project_judge_score
except ImportError:
    from loop_stats import analyze_pairs, score_loads, project_judge_score


class LoopStatsTests(unittest.TestCase):
    def test_multiplicative_order_bias_cancels_in_blocks(self):
        samples = []
        for _ in range(12):
            samples.extend([{'baseline_us': 12 * 1.2, 'candidate_us': 10, 'order': 'AB'},
                            {'baseline_us': 12, 'candidate_us': 10 * 1.2, 'order': 'BA'}])
        result = analyze_pairs(samples)
        self.assertAlmostEqual(result['speedup'], 1.2)
        self.assertTrue(result['accepted'])
        self.assertEqual(result, analyze_pairs(samples))
        json.dumps(result, allow_nan=False)

    def test_paired_drift_and_known_improvement(self):
        rng = random.Random(9)
        samples = []
        for _ in range(40):
            drift = rng.uniform(.8, 1.2)
            samples.append({'baseline_us': 20 * drift + 4, 'candidate_us': 16 * drift + 4})
        result = analyze_pairs(samples, overhead_us=4)
        self.assertAlmostEqual(result['speedup'], 1.25)
        self.assertTrue(result['accepted'])
        self.assertEqual(result['overhead_us'], 4)

    def test_noise_does_not_create_acceptance(self):
        samples = [{'baseline_us': 10, 'candidate_us': c} for c in [9, 11] * 20]
        result = analyze_pairs(samples)
        self.assertFalse(result['accepted'])
        self.assertEqual(result['gain_us'], 0)

    def test_floor_and_unbalanced_rejected(self):
        for value in [0, -1, 4, float('nan'), float('inf')]:
            with self.assertRaises(ValueError):
                analyze_pairs([{'baseline_us': 10, 'candidate_us': value}] * 2, overhead_us=4)
        with self.assertRaises(ValueError):
            analyze_pairs([{'baseline_us': 10, 'candidate_us': 9}])
        with self.assertRaises(ValueError):
            analyze_pairs([{'baseline_us': 10, 'candidate_us': 9, 'order': 'AB'}] * 2)

    def test_single_block_cannot_approve(self):
        self.assertFalse(analyze_pairs([{'baseline_us': 20, 'candidate_us': 10}] * 2)['accepted'])

    def test_positive_median_with_uncertain_blocks_rejected(self):
        samples = []
        for gain in [-2, -1, 1, 1, 1, 2]:
            samples.extend([{'baseline_us': 20, 'candidate_us': 20 - gain}] * 2)
        result = analyze_pairs(samples, bootstrap_samples=2000)
        self.assertGreater(result['gain_us'], .3)
        self.assertLess(result['gain_us_ci95'][0], 0)
        self.assertFalse(result['accepted'])

    def test_score_is_geomean_of_load_sums(self):
        result = score_loads([
            {'load': 1, 'layer': 'a', 'baseline_us': 100, 'candidate_us': 100},
            {'load': 1, 'layer': 'b', 'baseline_us': 1, 'candidate_us': .1},
            {'load': 8, 'layer': 'a', 'baseline_us': 20, 'candidate_us': 10, 'weight': 62},
        ])
        self.assertAlmostEqual(result['score'], math.sqrt((101 / 100.1) * 2))
        self.assertEqual(result['loads']['8']['baseline_us'], 1240)
        with self.assertRaises(ValueError):
            score_loads([{'load': 1, 'layer': 'a', 'baseline_us': 1, 'candidate_us': 1}] * 2)

    def test_project_judge_absolute_gains_and_regression(self):
        profile = {'benchmarks': [
            {'case': {'m': 1, 'layer': 'a', 'load': 'decode', 'weight': 62},
             'baseline_ns': 20000, 'median_ns': 10000},
            {'case': {'m': 1, 'layer': 'b', 'load': 'decode', 'weight': 62},
             'baseline_ns': 10000, 'median_ns': 10000},
            {'case': {'m': 8, 'layer': 'a', 'load': 'verify', 'weight': 62},
             'baseline_ns': 20000, 'median_ns': 10000}]}
        result = project_judge_score(profile, {(1, 'a'): 2, (8, 'a'): -2})
        self.assertTrue(result['estimated_not_judge'])
        self.assertAlmostEqual(result['baseline_score'], math.sqrt(1.5 * 2))
        self.assertAlmostEqual(result['projected_score'], math.sqrt(30 / 18 * 20 / 12))
        self.assertEqual(result['projected_loads']['decode']['candidate_us'], 18 * 62)
        json.dumps(result, allow_nan=False)
        for gains in [{(2, 'a'): 1}, {(1, 'a'): 10}, {(1, 'a'): float('nan')}]:
            with self.assertRaises(ValueError):
                project_judge_score(profile, gains)

    def test_cp15_profile_score(self):
        path = Path(__file__).parent / 'profiles' / 'top50_candidate.03960c2e352c40e3.json'
        result = project_judge_score(json.loads(path.read_text()), {})
        self.assertAlmostEqual(result['baseline_score'], 1.426996, places=6)
        self.assertEqual(result['projected_score'], result['baseline_score'])
        self.assertEqual(result['load_count'], 6)


if __name__ == '__main__':
    unittest.main()
