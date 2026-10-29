"""CPU-only proof: vector decode emits canonical Gluon DotA register order."""
import ast
import pathlib
import random
import struct
from adaptive_record_codec import pack_records, unpack_records, rebase_word


def main():
    source = pathlib.Path(__file__).with_name('adaptive_record_kernel.py').read_text()
    helpers = {}
    for node in ast.parse(source).body:
        if isinstance(node, ast.FunctionDef) and node.name in ('dot_record_order', 'vector_asm'):
            exec(compile(ast.Module(body=[node], type_ignores=[]), 'helpers', 'exec'), helpers)
    order = helpers['dot_record_order']()
    assert sorted(order) == list(range(1024))
    rng = random.Random(123)
    words = [((120 + rng.randrange(16)) << 7) | rng.randrange(128) | (rng.randrange(2) << 15)
             for _ in range(4096)]
    words[1031] = (145 << 7) | 31
    packed = pack_records(words, base=60, rebase_raw=True, order=order)
    assert list(unpack_records(packed)) == words
    assert 0 < packed['raw_records'] < 4

    def prmt(a, b, select):
        src = [(a >> (8 * i)) & 255 for i in range(4)] + [(b >> (8 * i)) & 255 for i in range(4)]
        return sum(src[(select >> (4 * i)) & 7] << (8 * i) for i in range(4))

    for record in range(4):
        meta = packed['metadata'][record]
        raw, base = bool(meta >> 31), (meta & 0x7fffffff) * 16
        for lane in range(32):
            values = []
            for j in range(4):
                if raw:
                    regs = struct.unpack_from('<4I', packed['payload'], base + j * 512 + lane * 16)
                else:
                    a, b, c = [struct.unpack_from('<I', packed['payload'], base + plane * 512 + lane * 16 + j * 4)[0]
                               for plane in range(3)]
                    h0 = (c & 0x87878787) | 0x38383838
                    rotate = ((c << 4) | (c >> 28)) & 0xffffffff
                    h1 = (rotate & 0x87878787) | 0x38383838
                    regs = [prmt(a, h0, 0x5140), prmt(a, h0, 0x7362),
                            prmt(b, h1, 0x5140), prmt(b, h1, 0x7362)]
                for value in regs:
                    values.extend([value & 65535, value >> 16])
            for reg, value in enumerate(values):
                row = (lane >> 2) + ((reg >> 1) & 1) * 8
                k = (reg & 1) + (lane & 3) * 2 + ((reg >> 2) & 1) * 8 + ((reg >> 3) & 1) * 16 + ((reg >> 4) & 1) * 32
                expected = rebase_word(words[record * 1024 + row * 64 + k], -8)
                assert value == expected, (record, lane, reg)
    asm = helpers['vector_asm']()
    assert asm.count('ld.global.v4.u32') == 7
    assert asm.count('lop3.b32') + asm.count('shf.l.wrap.b32') + asm.count('prmt.b32') == 28
    assert 'bra' not in asm
    print('PASS:4096 compressed/raw fragment values; vector loads,7decode ops/8weights; no format branch')


if __name__ == '__main__':
    main()
