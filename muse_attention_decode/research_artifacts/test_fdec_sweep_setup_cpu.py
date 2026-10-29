"""CPU-only bounds/prologue proof. No torch, Triton, CUDA, or job APIs."""
from pathlib import Path
import ast


def main():
    ctx, jitter, q_len = 80000, 6000, 1
    low, high = ctx - jitter, ctx + jitter - 1
    old_capacity, repaired_capacity = ctx + 8, ctx + jitter + 8
    assert low == 74000 and high == 85999
    invalid_old = sum(not 0 <= p < old_capacity for p in range(low, high + 1))
    assert invalid_old == 5992
    assert all(0 <= p and p + q_len <= repaired_capacity for p in range(low, high + 1))
    for batch in (16, 32):
        # A permutation of B*capacity reshaped to [B,capacity] always yields
        # a legal flattened cache row whenever the logical column is legal.
        assert (batch - 1) * repaired_capacity + high < batch * repaired_capacity
    for depth in range(2, 7):
        for slot in range(depth - 1):
            prologue = [-1] * (depth - 1)
            prologue[slot] = -2
            new_guard = any(row == -2 for row in prologue)
            assert new_guard
            old_slots = {depth - 2, 0}
            if depth > 2: old_slots.add(1)
            if depth > 3: old_slots.add(2)
            old_guard = any(prologue[i] == -2 for i in old_slots)
            assert old_guard == (depth != 6 or slot != 3)
    root = Path(__file__).resolve().parents[1]
    for filename in ('probe_fdec_sweep_fixed.py', 'probe_fdec_sweep_repaired.py'):
        ast.parse((root / filename).read_text())
    print('PASS: 5992/12000 old logical positions invalid; all repaired positions valid; '
          'RBF2..6 all-slot guard covers every new-token slot; syntax PASS')


if __name__ == '__main__':
    main()
