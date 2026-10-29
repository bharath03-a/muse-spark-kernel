"""CPU-only balanced cold-graph statistics; never subtract hidden overhead.

Input order is AB, BA, AB, BA. Each adjacent pair forms one bootstrap
unit, so the resampler preserves the order balance of an ABBA block.
"""
from __future__ import annotations

import math
import random
import statistics


def _number(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f'{name} must be a finite number')
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f'{name} must be finite')
    return value


def _percentile(values, fraction):
    values = sorted(values)
    index = fraction * (len(values) - 1)
    lo = int(index)
    hi = min(lo + 1, len(values) - 1)
    return values[lo] + (values[hi] - values[lo]) * (index - lo)


def analyze_pairs(pairs: list[dict], overhead_us: float = 0,
                  min_gain_us: float = .3, min_speedup: float = 1.01,
                  bootstrap_samples: int = 1000) -> dict:
    """Analyze balanced repetitions, resampling complete ABBA blocks.

    Optional ``order`` fields must alternate 'AB', 'BA' (A=baseline).
    Without them the caller is responsible for preserving that order.
    ``accepted`` requires >=4 blocks and both lower CI bounds to exceed
    the requested practical thresholds. Confidence intervals are paired
    bootstrap intervals, not guarantees about the independent judge.
    """
    overhead_us = _number(overhead_us, 'overhead_us')
    min_gain_us = _number(min_gain_us, 'min_gain_us')
    min_speedup = _number(min_speedup, 'min_speedup')
    if overhead_us < 0 or min_gain_us < 0 or min_speedup < 1:
        raise ValueError('overhead/gain must be nonnegative; speedup >= 1')
    if isinstance(bootstrap_samples, bool) or not isinstance(bootstrap_samples, int) or bootstrap_samples < 100:
        raise ValueError('bootstrap_samples must be an integer >= 100')
    if not pairs or len(pairs) % 2:
        raise ValueError('need a nonempty even number of AB/BA repetitions')
    has_order = any('order' in p for p in pairs)
    raw_b, raw_c = [], []
    for i, pair in enumerate(pairs):
        if has_order and pair.get('order') != ('AB' if i % 2 == 0 else 'BA'):
            raise ValueError('order must alternate AB then BA')
        b = _number(pair['baseline_us'], 'baseline_us')
        c = _number(pair['candidate_us'], 'candidate_us')
        if min(b, c) <= overhead_us:
            raise ValueError('every raw sample must exceed explicit overhead')
        raw_b.append(b)
        raw_c.append(c)
    bs = [v - overhead_us for v in raw_b]
    cs = [v - overhead_us for v in raw_c]
    # Geometric ratios neutralize symmetric multiplicative order bias.
    # Gains remain in microseconds, with both orders equally represented.
    ratios, gains = [], []
    for i in range(0, len(pairs), 2):
        ratios.append(math.exp((math.log(bs[i] / cs[i]) + math.log(bs[i + 1] / cs[i + 1])) / 2))
        gains.append(((bs[i] - cs[i]) + (bs[i + 1] - cs[i + 1])) / 2)
    rng = random.Random(20261027)
    boot_ratio, boot_gain = [], []
    for _ in range(bootstrap_samples):
        indices = [rng.randrange(len(ratios)) for _ in ratios]
        boot_ratio.append(statistics.median(ratios[i] for i in indices))
        boot_gain.append(statistics.median(gains[i] for i in indices))
    ratio_ci = [_percentile(boot_ratio, .025), _percentile(boot_ratio, .975)]
    gain_ci = [_percentile(boot_gain, .025), _percentile(boot_gain, .975)]
    enough = len(ratios) >= 4
    accepted = enough and ratio_ci[0] > min_speedup and gain_ci[0] > min_gain_us
    reasons = []
    if not enough:
        reasons.append('fewer than four balanced blocks')
    if ratio_ci[0] <= min_speedup:
        reasons.append('speedup lower confidence bound does not exceed threshold')
    if gain_ci[0] <= min_gain_us:
        reasons.append('absolute gain lower confidence bound does not exceed threshold')
    return {
        'raw_median_us': {'baseline': statistics.median(raw_b), 'candidate': statistics.median(raw_c)},
        'adjusted_median_us': {'baseline': statistics.median(bs), 'candidate': statistics.median(cs)},
        'overhead_us': overhead_us,
        'speedup': statistics.median(ratios), 'speedup_ci95': ratio_ci,
        'gain_us': statistics.median(gains), 'gain_us_ci95': gain_ci,
        'repetitions': len(pairs), 'balanced_blocks': len(ratios),
        'bootstrap_samples': bootstrap_samples, 'bootstrap_seed': 20261027,
        'order_verified': has_order, 'min_gain_us': min_gain_us,
        'min_speedup': min_speedup, 'accepted': accepted, 'reasons': reasons,
    }


def score_loads(records: list[dict]) -> dict:
    """Sum layer times per load, then geometric-mean load speedups.

    Records: load, layer, baseline_us, candidate_us, optional weight=1.
    No shape-level geometric mean; duplicate (load, layer) is rejected.
    """
    if not records:
        raise ValueError('no score records')
    loads, seen = {}, set()
    for record in records:
        load = str(record['load'])
        key = (load, str(record['layer']))
        if key in seen:
            raise ValueError(f'duplicate load/layer: {key}')
        seen.add(key)
        b = _number(record['baseline_us'], 'baseline_us')
        c = _number(record['candidate_us'], 'candidate_us')
        w = _number(record.get('weight', 1), 'weight')
        if min(b, c, w) <= 0:
            raise ValueError('score times and weights must be positive')
        total = loads.setdefault(load, {'baseline_us': 0., 'candidate_us': 0.})
        total['baseline_us'] += b * w
        total['candidate_us'] += c * w
    for total in loads.values():
        total['speedup'] = total['baseline_us'] / total['candidate_us']
    score = math.exp(statistics.mean(math.log(v['speedup']) for v in loads.values()))
    return {'score': score, 'loads': loads, 'load_count': len(loads)}


def project_judge_score(profile: dict, gains: dict[tuple[int, str], float]) -> dict:
    """Hypothetically transfer local absolute gains onto saved judge times.

    Gains are microseconds; negative values model regressions. No confidence
    or cross-machine equivalence is implied. Judge times already have their
    own overhead adjustment, so no additional subtraction is performed.
    """
    benchmarks = profile.get('benchmarks')
    if not isinstance(benchmarks, list) or not benchmarks:
        raise ValueError('profile must contain nonempty benchmarks')
    known = {(r['case']['m'], r['case']['layer']) for r in benchmarks}
    unknown = set(gains) - known
    if unknown:
        raise ValueError(f'unknown shapes: {sorted(unknown, key=str)}')
    gains = {key: _number(value, 'gain_us') for key, value in gains.items()}
    original, projected = [], []
    for record in benchmarks:
        case = record['case']
        key = (case['m'], case['layer'])
        baseline_us = _number(record['baseline_ns'], 'baseline_ns') / 1000
        current_us = _number(record['median_ns'], 'median_ns') / 1000
        projected_us = current_us - gains.get(key, 0.)
        if current_us <= 0 or projected_us <= 0:
            raise ValueError(f'nonpositive current/projected time for {key}')
        common = {'load': case['load'], 'layer': case['layer'],
                  'baseline_us': baseline_us, 'weight': case.get('weight', 1)}
        original.append({**common, 'candidate_us': current_us})
        projected.append({**common, 'candidate_us': projected_us})
    before = score_loads(original)
    after = score_loads(projected)
    return {'estimated_not_judge': True, 'baseline_score': before['score'],
            'projected_score': after['score'], 'baseline_loads': before['loads'],
            'projected_loads': after['loads'], 'load_count': before['load_count'],
            'applied_gains_us': [{'m': m, 'layer': layer, 'gain_us': value}
                                 for (m, layer), value in sorted(gains.items())]}
