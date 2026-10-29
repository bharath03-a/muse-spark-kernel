"""Resumable bounded research queue. No automatic leaderboard submissions.

python3 -m research_harness.loop research_harness/prefill_queue.json
"""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import shutil

ROOT = Path(__file__).resolve().parents[1]


def fingerprint(task):
    digest = hashlib.sha256(json.dumps(task, sort_keys=True).encode())
    for relative in sorted(task['sources']):
        path = ROOT / relative
        digest.update(relative.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def run_queue(manifest_path, *, retry_failed=False):
    manifest = json.loads(Path(manifest_path).read_text())
    out = ROOT / manifest.get('output', 'research_harness/results')
    out.mkdir(parents=True, exist_ok=True)
    # Hold the lock for the entire invocation; prevent two heartbeats launching
    # identical expensive work. OS releases flock after crash or interruption.
    import fcntl
    lock_dir = ROOT / 'research_harness/results'
    lock_dir.mkdir(parents=True, exist_ok=True)
    with (lock_dir / 'experiments.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        journal = out / 'journal.jsonl'
        history = [json.loads(s) for s in journal.read_text().splitlines()] if journal.exists() else []
        skipped_failure = False
        for task in manifest['tasks']:
            key = fingerprint(task)
            previous = [r for r in history if r['fingerprint'] == key]
            if previous and (previous[-1]['status'] == 'passed' or not retry_failed):
                print(json.dumps(dict(id=task['id'], status='already_recorded', previous=previous[-1]['status'])), flush=True)
                if previous[-1]['status'] != 'passed':
                    skipped_failure = True
                    if task.get('stop_on_failure', True):
                        return 1
                continue
            cmd = task['command']
            if not isinstance(cmd, list) or not all(isinstance(v, str) for v in cmd):
                raise ValueError('Commands must be argv arrays, never shell strings')
            if any(v in ('submit', 'withdraw') for v in cmd):
                raise ValueError('This research loop does not submit or withdraw leaderboard entries')
            if not task.get('success_markers'):
                raise ValueError('Each experiment needs explicit success markers')
            log = out / (task['id'] + '-' + key[:12] + '.log')
            snapshot = out / 'sources' / key
            snapshot.mkdir(parents=True, exist_ok=True)
            for relative in task['sources']:
                source = (ROOT / relative).resolve()
                if not source.is_relative_to(ROOT.resolve()):
                    raise ValueError('Source snapshots must stay inside workspace')
                dest = snapshot / relative
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, dest)
            (snapshot / 'experiment.json').write_text(json.dumps(task, indent=2) + '\n')
            started = dict(id=task['id'], fingerprint=key, status='started', log=str(log),
                           source_snapshot=str(snapshot), command=cmd,
                           utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
            # Persist before dispatch. If the coordinator dies, a later wakeup
            # must inspect the remote app, not silently launch the task twice.
            with journal.open('a') as stream:
                stream.write(json.dumps(started) + '\n')
                stream.flush()
                os.fsync(stream.fileno())
            history.append(started)
            print(json.dumps(dict(id=task['id'], status='started', log=str(log))), flush=True)
            start = time.monotonic()
            status, code = 'failed', None
            with log.open('w') as stream:
                try:
                    process = subprocess.Popen(cmd, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
                    try:
                        code = process.wait(timeout=task.get('timeout_seconds', 600))
                    except (subprocess.TimeoutExpired, KeyboardInterrupt):
                        import signal
                        os.killpg(process.pid, signal.SIGTERM)
                        try:
                            process.wait(timeout=10)
                        except subprocess.TimeoutExpired:
                            os.killpg(process.pid, signal.SIGKILL)
                            process.wait()
                        raise
                except subprocess.TimeoutExpired:
                    status = 'timeout'
            content = log.read_text()
            markers = {s: content.count(s) for s in task['success_markers']}
            if code == 0 and all(n >= task.get('minimum_marker_counts', {}).get(s, 1) for s, n in markers.items()):
                status = 'passed'
            if fingerprint(task) != key:
                status = 'source_changed_during_run'
            record = dict(id=task['id'], fingerprint=key, status=status, returncode=code,
                          elapsed_s=time.monotonic()-start, log=str(log), markers=markers,
                          utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                          hypothesis=task['hypothesis'], source_snapshot=str(snapshot))
            with journal.open('a') as stream:
                stream.write(json.dumps(record) + '\n')
                stream.flush()
                os.fsync(stream.fileno())
            history.append(record)
            print(json.dumps(record), flush=True)
            if status != 'passed' and task.get('stop_on_failure', True):
                return 1
    return 1 if skipped_failure else 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('manifest')
    parser.add_argument('--retry-failed', action='store_true')
    args = parser.parse_args()
    raise SystemExit(run_queue(args.manifest, retry_failed=args.retry_failed))
