"""Compose explicitly selected fdec prefetch/balance settings; never select a winner.

Run only after the corrected sweep shows >5% paired improvement and correctness.
PFF affects every fdec instantiation; B1/full controls remain necessary before promotion.
No CUDA, projection, MoE, or PDL source is changed by this builder.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def configure(source, prefix, prefetch, balanced_batches):
    tree = ast.parse(source)
    assignments = {n.targets[0].id: n for n in tree.body
                   if isinstance(n, ast.Assign) and len(n.targets) == 1
                   and isinstance(n.targets[0], ast.Name)}
    debug_name, balance_name = prefix + '_DBG', prefix + 'DFL'
    debug_node, balance_node = assignments[debug_name], assignments[balance_name]
    debug = ast.literal_eval(debug_node.value)
    balance = ast.literal_eval(balance_node.value)
    assert not any(flag.startswith('-DPFF=') for flag in debug), 'base already overrides PFF'
    if prefetch:
        debug.append(f'-DPFF={prefetch}')
    for batch in balanced_batches:
        # Preserve gate prefetch + one-pass merge, add balanced full-context partition.
        balance[(batch, False)] = balance.get((batch, False), 5) | 8
    updates = {debug_name: debug, balance_name: balance}
    result = source
    for name, value in updates.items():
        literal = ast.get_source_segment(source, assignments[name].value)
        # Restrict replacement to the selected assignment, avoiding equal empty literals.
        old = ast.get_source_segment(source, assignments[name])
        new = old[:old.index('=') + 1] + ' ' + repr(value)
        assert result.count(old) == 1, name
        result = result.replace(old, new, 1)
    compiled = ast.parse(result)
    def scrub(t):
        for node in t.body:
            if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name) and node.targets[0].id in updates:
                node.value = ast.Constant('<selected fdec settings>')
        return ast.dump(t, include_attributes=False)
    assert scrub(tree) == scrub(compiled), 'builder changed code beyond two setting assignments'
    compile(result, '<fdec candidate>', 'exec')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prefetch', type=int, choices=(0, 4, 8), required=True)
    parser.add_argument('--balanced-batches', type=int, choices=(16, 32), nargs='*', default=[])
    parser.add_argument('--standalone-base', type=Path, default=ROOT / 'muse_attention_decode/candidate_openkernels_v75_f16pv.py')
    parser.add_argument('--whole-base', type=Path, default=ROOT / 'muse_decode_layer/candidate_v35_wxpin32_strict.py')
    parser.add_argument('--standalone-output', type=Path, required=True)
    parser.add_argument('--whole-output', type=Path, required=True)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    for base, output, prefix in ((args.standalone_base, args.standalone_output, ''), (args.whole_base, args.whole_output, 'A_')):
        assert output.resolve() != base.resolve() and output.name != 'submission.py', 'preserve the baseline/best file'
        source = base.read_text()
        result = configure(source, prefix, args.prefetch, args.balanced_batches)
        record = dict(base=str(base), output=str(output), base_sha256=hashlib.sha256(source.encode()).hexdigest(),
                      output_sha256=hashlib.sha256(result.encode()).hexdigest(), prefetch=args.prefetch,
                      balanced_batches=args.balanced_batches, dry_run=args.dry_run)
        print(json.dumps(record))
        if not args.dry_run:
            assert not output.exists(), 'choose a new candidate path'
            output.write_text(result)


if __name__ == '__main__':
    main()
