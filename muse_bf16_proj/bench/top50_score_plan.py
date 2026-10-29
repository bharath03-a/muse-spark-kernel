"""CPU-only score impact calculator using saved judge results.

Example: .venv/bin/python bench/top50_score_plan.py --save 128:router:1 --save 256:router:2
Savings are hypothetical microseconds, never measured results.
"""
import argparse
import json
import math
from collections import defaultdict
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', default='bench/profiles/cand_D.9d8ae0d9547747ce.json')
    parser.add_argument('--target', type=float, default=1.5271)
    parser.add_argument('--save', action='append', default=[], help='m:layer:microseconds')
    args = parser.parse_args()
    records = json.loads(Path(args.profile).read_text())['benchmarks']
    changes = {}
    for spec in args.save:
        m, layer, us = spec.split(':')
        changes[(int(m), layer)] = float(us)
    loads = defaultdict(lambda: [0., 0., 0.])
    for record in records:
        case = record['case']
        m, layer = case['m'], case['layer']
        old = record['median_ns'] / 1000
        saving = changes.pop((m, layer), 0.)
        if saving < 0 or saving >= old:
            raise ValueError(f'Invalid saving for {m}/{layer}: {saving}')
        loads[m][0] += record['baseline_ns'] / 1000
        loads[m][1] += old
        loads[m][2] += old - saving
    if changes:
        raise ValueError(f'Unknown shapes: {changes}')
    score = math.exp(sum(math.log(base / old) for base, old, new in loads.values()) / len(loads))
    projected = math.exp(sum(math.log(base / new) for base, old, new in loads.values()) / len(loads))
    reduction = 1 - score / args.target
    print('Saved-profile score:', round(score, 6))
    print('Hypothetical score:', round(projected, 6))
    print('Target:', args.target, 'uniform required time reduction:', f'{reduction:.2%}')
    print('m baseline_us current_us hypothetical_us uniform_target_saving_us')
    for m in sorted(loads):
        base, old, new = loads[m]
        print(m, *(f'{v:.3f}' for v in (base, old, new, old * reduction)))


if __name__ == '__main__':
    main()
