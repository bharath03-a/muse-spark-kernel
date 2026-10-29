"""Serialized, budgeted Modal H200 runner shared by all agents.

    .venv/bin/python bench/gpu_q.py <owner> micro <bench_script_name> [args...]
    .venv/bin/python bench/gpu_q.py status

One job at a time (file lock). Wall time of every job is charged to the ledger;
jobs are refused once the 30-minute budget is spent. Each job is killed after JOB_CAP_S.
"""

from __future__ import annotations

import fcntl
import json
import subprocess as sp
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LEDGER = ROOT / "bench" / "gpu_ledger.json"
LOCK = ROOT / "bench" / ".gpu_q.lock"
BUDGET_S = 30 * 60
JOB_CAP_S = 300


def _load() -> dict:
    if LEDGER.exists():
        return json.loads(LEDGER.read_text())
    return {"used_s": 0.0, "jobs": []}


def main() -> None:
    if len(sys.argv) >= 2 and sys.argv[1] == "status":
        led = _load()
        print(f"used {led['used_s']:.0f}s of {BUDGET_S}s; jobs {len(led['jobs'])}")
        return
    if len(sys.argv) < 4 or sys.argv[2] != "micro":
        sys.exit(__doc__)
    owner, name, args = sys.argv[1], sys.argv[3], " ".join(sys.argv[4:])
    with open(LOCK, "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        led = _load()
        if led["used_s"] >= BUDGET_S:
            sys.exit(f"GPU budget exhausted ({led['used_s']:.0f}s)")
        cmd = [str(ROOT / ".venv/bin/modal"), "run", str(ROOT / "modal_app.py") + "::micro", "--name", name]
        if args:
            cmd += ["--args", args]
        t0 = time.time()
        try:
            res = sp.run(cmd, cwd=ROOT, timeout=JOB_CAP_S, capture_output=True, text=True)
            out, rc = res.stdout + res.stderr, res.returncode
        except sp.TimeoutExpired as exc:
            out, rc = f"{exc.stdout or ''}{exc.stderr or ''}\nJOB TIMEOUT {JOB_CAP_S}s", -1
        dt = time.time() - t0
        led["used_s"] += dt
        led["jobs"].append({"owner": owner, "name": name, "args": args, "s": round(dt, 1), "rc": rc,
                            "t": time.strftime("%H:%M:%S", time.gmtime())})
        LEDGER.write_text(json.dumps(led, indent=1))
    print(out[-20000:])
    print(f"[gpu_q] {owner} {name}: {dt:.0f}s, rc={rc}; budget used {led['used_s']:.0f}/{BUDGET_S}s")


if __name__ == "__main__":
    main()
