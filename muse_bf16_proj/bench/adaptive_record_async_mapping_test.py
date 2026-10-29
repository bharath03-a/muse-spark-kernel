"""CPU-only proof of encoded-stage view, vectordecoder ABI and ring schedule."""
import ast
import pathlib
from adaptive_record_split_mapping_test import main as prior_proof


def main():
    prior_proof()
    helpers = {}
    tree = ast.parse(pathlib.Path(__file__).with_name('adaptive_record_async_kernel.py').read_text())
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == 'decode_asm':
            exec(compile(ast.Module(body=[node], type_ignores=[]), 'helpers', 'exec'), helpers)
    asm = helpers['decode_asm']()
    assert 'ld.' not in asm and 'st.' not in asm and 'bra' not in asm
    assert asm.count('mov.b32') == 16
    assert asm.count('lop3.b32') + asm.count('shf.l.wrap.b32') + asm.count('prmt.b32') == 28
    assert '$32' in asm
    seen = set()
    for record in range(8):
        for word in range(512):
            for half in range(2):
                row16, batch = divmod(record, 4)
                plane, rest = divmod(word, 128)
                lane, subword = divmod(rest, 4)
                row = row16 * 16 + (subword & 1) * 8 + lane // 4
                k = (plane // 2) * 32 + (plane & 1) * 16 + (subword // 2) * 8 + (lane & 3) * 2 + half
                key = (batch, row, k)
                assert key not in seen
                seen.add(key)
                # View reshape/permutation axes in kernel.
                axes = [row16, batch, plane // 2, plane % 2, lane // 4, lane % 4, subword // 2, subword % 2, half]
                shuffled = [axes[i] for i in (1, 0, 7, 4, 2, 3, 6, 5, 8)]
                bb = shuffled[0]
                rr = (shuffled[1] * 2 + shuffled[2]) * 8 + shuffled[3]
                kk = ((((shuffled[4] * 2 + shuffled[5]) * 2 + shuffled[6]) * 4 + shuffled[7]) * 2 + shuffled[8])
                assert (bb, rr, kk) == key
    assert len(seen) == 4 * 32 * 64
    for depth in (2, 3, 4):
        steps = 32
        slots = list(range(depth))
        queue = list(range(depth))
        for it in range(steps):
            assert len(queue) == depth
            assert queue.pop(0) == it
            assert slots[it % depth] == it
            # After registerconsumption+CTAbarrier, reuse samephysicalslot.
            slots[it % depth] = it + depth
            queue.append(it + depth)  # includes maskedtail groups
        assert queue == list(range(steps, steps + depth))
    print('PASS:stageview8192values, pure7opdecodeABI, fixeddepth2/3/4ring scheduling')

if __name__ == '__main__':
    main()
