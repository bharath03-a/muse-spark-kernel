"""Two serialized experiments sharing one bounded Modal allocation."""
import runpy
import sys
import time
import traceback

start = time.monotonic()
for name, args in [('top50_qkv_mma', []), ('top50_large_pipeline', ['--layer', 'o', '--m', '256', '--reps', '20'])]:
    if time.monotonic() - start > 135:
        print('Combined runtime cutoff before', name, flush=True)
        break
    print('START', name, flush=True)
    sys.argv = [name] + args
    try:
        runpy.run_path('/root/proj/bench/' + name + '.py', run_name='__main__')
    except Exception:
        traceback.print_exc()
    print('FINISHED', name, 'elapsed', time.monotonic() - start, flush=True)
