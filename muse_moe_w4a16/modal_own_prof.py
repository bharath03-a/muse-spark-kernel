from pathlib import Path

import modal

PROJECT_DIR = Path(__file__).parent
REMOTE_DIR = "/root/own"
FILES = ["reference.py", "checks.py", "nvfp4.py", "own_moe_v3.py", "own_moe_v4.py", "own_moe_v5.py"]

image = (
    modal.Image.from_registry("lmsysorg/sglang:dev-muse-spark-open")
    .run_commands(
        "echo /usr/local/lib/python3.12/dist-packages > /opt/sglang/lib/python3.12/site-packages/zz_system.pth",
        "echo /usr/lib/python3/dist-packages >> /opt/sglang/lib/python3.12/site-packages/zz_system.pth",
    )
    .entrypoint([])
)
for f in FILES:
    image = image.add_local_file(PROJECT_DIR / f, remote_path=f"{REMOTE_DIR}/{f}")

app = modal.App("muse-moe-own-prof", image=image)

TARGET = '''
import sys, importlib, torch
sys.path.insert(0, "/root/own")
import reference as R
w = R.generate_weights(R.WEIGHTS_SEED)
for name in ("own_moe_v3", "own_moe_v4"):
    m = importlib.import_module(name)
    m.SWEEP_BUDGET_S = 0.0
    st = m.prepare(w)
    for bs in (16, 256):
        d = R.generate_input(bs, "ep8", 777 + bs)
        m.kernel_fn(d, st)
        torch.cuda.synchronize()
'''

METRICS = ",".join([
    "gpu__time_duration.sum",
    "dram__throughput.avg.pct_of_peak_sustained_elapsed",
    "dram__bytes_read.sum",
    "sm__inst_executed.sum",
    "sm__instruction_throughput.avg.pct_of_peak_sustained_active",
    "sm__warps_active.avg.pct_of_peak_sustained_active",
    "launch__registers_per_thread",
    "launch__occupancy_limit_registers",
    "launch__occupancy_limit_shared_mem",
    "l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum",
    "smsp__average_warp_latency_issue_stalled_long_scoreboard.ratio",
    "smsp__average_warp_latency_issue_stalled_barrier.ratio",
    "smsp__average_warp_latency_issue_stalled_wait.ratio",
    "smsp__average_warp_latency_issue_stalled_short_scoreboard.ratio",
    "smsp__average_warp_latency_issue_stalled_math_pipe_throttle.ratio",
    "smsp__average_warp_latency_issue_stalled_mio_throttle.ratio",
    "smsp__average_warp_latency_issue_stalled_lg_throttle.ratio",
])


@app.function(gpu="H200", timeout=900)
def run():
    import subprocess
    import time
    t0 = time.time()
    print(subprocess.run("which ncu; ncu --version", shell=True, capture_output=True, text=True).stdout, flush=True)
    open("/tmp/target.py", "w").write(TARGET)
    cmd = (f"ncu --clock-control none --kernel-name regex:'_w13_kernel|_w2_kernel' --metrics {METRICS} --csv "
           f"python /tmp/target.py")
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    print(r.stdout[-30000:], flush=True)
    print(r.stderr[-3000:], flush=True)
    print(f"[prof] gpu wall {time.time() - t0:.0f}s", flush=True)


@app.local_entrypoint()
def main():
    run.remote()
