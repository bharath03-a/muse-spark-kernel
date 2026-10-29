"""Pure statistics for paired, order-balanced CUDA graph experiments."""
import math
import random
import statistics


def paired_summary(samples, *, replay_overhead_us=0.0, bootstrap=2000, seed=0):
    """Each adjacent AB/BA pair is one independent bootstrap unit.

    Raw times remain in the report. Overhead defaults to zero: an unmeasured
    fixed subtraction can turn tiny kernels into arbitrarily large speedups.
    """
    if len(samples) < 8 or len(samples) % 2:
        raise ValueError('Need an even number of at least 8 paired repetitions')
    if not math.isfinite(replay_overhead_us) or replay_overhead_us < 0:
        raise ValueError('Invalid replay overhead')
    if bootstrap < 100:
        raise ValueError('Need at least 100 bootstrap replicates')
    ratios = []
    for row in samples:
        a, b = row['baseline_us'], row['candidate_us']
        if not all(math.isfinite(x) and x > replay_overhead_us for x in (a, b)):
            raise ValueError('Nonfinite, nonpositive or overhead-dominated timing')
        ratios.append((a - replay_overhead_us) / (b - replay_overhead_us))
    blocks = [math.sqrt(ratios[i] * ratios[i + 1]) for i in range(0, len(ratios), 2)]
    rng = random.Random(seed)
    draws = sorted(statistics.median(rng.choices(blocks, k=len(blocks))) for _ in range(bootstrap))
    speedup = statistics.median(blocks)
    low, high = draws[int(.025 * bootstrap)], draws[min(bootstrap - 1, int(.975 * bootstrap))]
    # Match the judge's minimum uncertainty guard, even for very stable clocks.
    low, high = min(low, speedup * .985), max(high, speedup * 1.015)
    return dict(baseline_us=statistics.median(r['baseline_us'] for r in samples),
                candidate_us=statistics.median(r['candidate_us'] for r in samples),
                paired_candidate_us=statistics.median(r['baseline_us'] for r in samples) / speedup,
                speedup=speedup, speedup_ci95=[low, high],
                decision='promising' if low > 1 else ('slower' if high < 1 else 'inconclusive'),
                replay_overhead_us=replay_overhead_us, samples=samples)


def weighted_load_speedup(shapes):
    """Geomean of load speedups, using weighted sums within each load."""
    loads = {}
    for row in shapes:
        weight = row['weight']
        candidate = row.get('paired_candidate_us', row['candidate_us'])
        if weight <= 0 or not all(math.isfinite(v) and v > 0 for v in (weight, row['baseline_us'], candidate)):
            raise ValueError('Positive shape weights and times required')
        sums = loads.setdefault(row['load'], [0.0, 0.0])
        sums[0] += weight * row['baseline_us']
        sums[1] += weight * candidate
    if not loads:
        raise ValueError('No measured loads')
    return math.exp(statistics.mean(math.log(a / b) for a, b in loads.values()))
