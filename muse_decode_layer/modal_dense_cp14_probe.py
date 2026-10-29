"""Bounded whole-layer cp14 integration probe on the judge H200 image."""
from pathlib import Path
import modal
ROOT = Path(__file__).resolve().parent
image = (modal.Image.from_registry('lmsysorg/sglang:dev-muse-spark-open')
    .run_commands("echo /usr/local/lib/python3.12/dist-packages > /opt/sglang/lib/python3.12/site-packages/zz_system.pth", "echo /usr/lib/python3/dist-packages >> /opt/sglang/lib/python3.12/site-packages/zz_system.pth")
    .entrypoint([])
    .add_local_dir(ROOT, remote_path='/root/decode', ignore=['profiles', '__pycache__', '*.pyc', '.venv']))
app = modal.App('muse-decode-cp14-probe', image=image)
cache = modal.Volume.from_name('muse-ext-cache', create_if_missing=True)
@app.function(gpu='H200', timeout=420, volumes={'/cache': cache})
def probe():
    import os, sys, time, json
    os.environ['TORCH_EXTENSIONS_DIR'] = '/cache/torch_ext'
    os.environ['TRITON_CACHE_DIR'] = '/cache/triton'
    os.environ['TVM_FFI_CACHE_DIR'] = '/cache/tvm-ffi'
    sys.path.insert(0, '/root/decode')
    import torch
    import reference
    import candidate_dense_cp14 as sub
    print('DEVICE', torch.cuda.get_device_name(), flush=True)
    start = time.monotonic()
    weights = reference.generate_weights(5353)
    print('WEIGHTS_SECONDS', time.monotonic()-start, flush=True)
    started = time.monotonic()
    state = sub.prepare(weights)
    torch.cuda.synchronize()
    elapsed = time.monotonic()-started
    print('PREPARE_SECONDS', elapsed, 'UNDER_300', elapsed < 300, flush=True)
    data = reference.generate_input(batch=1, ctx=80000, layer='sliding', seed=617, q_len=1, page=1, dist='soft')
    outputs = sub.kernel_fn(data, state)
    torch.cuda.synchronize()
    raw = reference.collect(data, outputs)
    good, message = reference.check_implementation(data, raw, weights)
    print('CORRECTNESS', json.dumps({'passed': good, 'message': message}), flush=True)
    assert good, message
    cache.commit()
