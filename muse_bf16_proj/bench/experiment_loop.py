"""Budgeted projection experiment loop. CPU dry-run requires no torch.

GPU execution is explicit; this tool never starts Modal, submits judge jobs,
changes a submission, or changes the GPU ledger. All timing is captured graphs,
paired AB/BA, with identical fresh inputs, read-only cold-L2 flushes, and raw
samples retained. Acceptance requires correctness and two independent cohorts.
"""
from __future__ import annotations
import argparse
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parent.parent
LAYERS = ('qkv_gate', 'o', 'pooler', 'unpooler', 'router')
MS = (1, 8, 16, 32, 128, 256)


def load_plan(path):
    plan = json.loads(Path(path).read_text())
    plan.setdefault('ms', list(MS))
    plan.setdefault('layers', list(LAYERS))
    if not plan['ms'] or not plan['layers']:
        raise ValueError('At least one token count and layer required')
    for name in ['baseline', *plan.get('candidates', {})]:
        spec = plan['baseline'] if name == 'baseline' else plan['candidates'][name]
        source = ROOT / spec['path']
        if not source.is_file():
            raise ValueError(f'Missing source {source}')
        content = source.read_bytes()
        ast.parse(content)
        sha = hashlib.sha256(content).hexdigest()
        if spec.get('sha256') and spec['sha256'] != sha:
            raise ValueError(f'Source changed for {name}: expected {spec["sha256"]}, actual {sha}')
        spec['sha256'] = sha
    if 'baseline' in plan.get('candidates', {}):
        raise ValueError('Candidate name baseline is reserved')
    if len(set(plan.get('ms', MS))) != len(plan.get('ms', MS)):
        raise ValueError('Duplicate token counts')
    if len(set(plan.get('layers', LAYERS))) != len(plan.get('layers', LAYERS)):
        raise ValueError('Duplicate layers')
    if plan.get('max_wall_seconds', 600) <= 0:
        raise ValueError('Positive wall-time budget required')
    if not plan.get('candidates'):
        raise ValueError('No candidates')
    for m in plan.get('ms', MS):
        if not isinstance(m, int) or m <= 0:
            raise ValueError(f'Invalid m={m}')
    for layer in plan.get('layers', LAYERS):
        if layer not in LAYERS:
            raise ValueError(f'Unknown layer {layer}')
    if plan.get('reps', 40) < 8 or plan.get('reps', 40) % 2:
        raise ValueError('reps must be even and >=8 for balanced blocks')
    if plan.get('cohorts', 2) < 2:
        raise ValueError('At least two independent cohorts required')
    return plan


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, ROOT/path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--manifest', default='bench/loop_manifest.json')
    ap.add_argument('--output', default='bench/loop_results.jsonl')
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()
    plan = load_plan(args.manifest)
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        print('Plan/sources valid; no GPU used, no performance claim')
        return
    try:
        run(plan, Path(args.output))
    except Exception as exc:
        # Partial JSONL samples remain usable for diagnosis, never acceptance.
        with Path(args.output).open('a') as stream:
            stream.write(json.dumps({'event':'run_failed','error':type(exc).__name__,
                                     'detail':str(exc)})+'\n')
        raise


def run(plan, output):
    import torch
    import triton
    sys.path.insert(0, str(ROOT/'bench'))
    from loop_correctness import build_inputs, validate_output
    from loop_stats import analyze_pairs
    # This runner is external to submissions; importing the reference here is permitted.
    ref = module('reference.py', 'loop_external_reference')
    run_id = uuid.uuid4().hex
    started = time.monotonic()
    deadline = started + plan.get('max_wall_seconds', 600)
    output.parent.mkdir(parents=True, exist_ok=True)

    def emit(record):
        record['run_id'] = run_id
        record['elapsed_s'] = round(time.monotonic()-started, 3)
        with output.open('a') as stream:
            stream.write(json.dumps(record, allow_nan=False)+'\n')
        print(json.dumps(record, allow_nan=False), flush=True)

    def budget():
        if time.monotonic() >= deadline:
            raise TimeoutError('Experiment wall budget reached; no candidate promoted')

    emit({'event':'environment','torch':torch.__version__,'triton':triton.__version__,
          'gpu':torch.cuda.get_device_name(),'plan':plan})
    weights = ref.generate_weights(plan.get('weight_seed', 20261027))
    baseline = module(plan['baseline']['path'], 'loop_baseline')
    budget()
    baseline_state = baseline.prepare({k:v.clone() for k,v in weights.items()})
    budget()
    flush = torch.empty(256*2**20//4, device='cuda', dtype=torch.float32).normal_()
    events = (torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True))

    class Capture:
        def __init__(self, mod, state, x, cfg):
            self.x=x.clone()
            self.data={'x':self.x,'config':cfg}
            for _ in range(3):
                mod.kernel_fn(self.data,state)
            torch.cuda.synchronize()
            self.graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph):
                self.out=mod.kernel_fn(self.data,state)
            torch.cuda.synchronize()
        def timed(self):
            flush.sum()  # READ flush, not dirty stores or a host wall timer.
            a,b=events
            a.record();self.graph.replay();b.record();b.synchronize()
            return a.elapsed_time(b)*1000

    decisions=[]
    for name,spec in plan['candidates'].items():
        budget()
        candidate=module(spec['path'],'loop_candidate_'+name)
        state=candidate.prepare({k:v.clone() for k,v in weights.items()})
        budget()
        all_accepted=True
        for m in plan.get('ms',MS):
            for layer in plan.get('layers',LAYERS):
                budget()
                w=weights[layer]
                n=1024 if layer=='unpooler' and m<=8 else w.shape[0]
                w64=w[:n].double()
                cfg={'m':m,'n':n,'k':w.shape[1],'layer':layer}
                x,_=build_inputs(m,w.shape[1],100+m,device='cuda')
                sides={'A':Capture(baseline,baseline_state,x,cfg),
                       'B':Capture(candidate,state,x,cfg)}
                correct=True
                for dist in ['normal','outliers','onehot','zero']:
                    fresh,probes=build_inputs(m,w.shape[1],1234+m,dist,device='cuda')
                    target=fresh.double()@w64.T
                    for side,captured in sides.items():
                        captured.x.copy_(fresh)
                        for repeat in range(3):
                            captured.graph.replay()
                            metrics=validate_output(captured.out,target,probes,layer=='router')
                            emit({'event':'correctness','candidate':name,'m':m,'layer':layer,
                                  'dist':dist,'side':side,'repeat':repeat,**metrics})
                            correct &= metrics['passed']
                if not correct:
                    all_accepted=False
                    emit({'event':'rejected_correctness','candidate':name,'m':m,'layer':layer})
                    continue
                for cohort in range(plan.get('cohorts',2)):
                    samples=[]
                    for rep in range(plan.get('reps',40)):
                        budget()
                        fresh,probes=build_inputs(m,w.shape[1],10000+cohort*1000+rep,device='cuda')
                        target=fresh.double()@w64.T
                        for captured in sides.values():
                            captured.x.copy_(fresh)
                        order='AB' if rep%2==0 else 'BA'
                        measured={side:sides[side].timed() for side in order}
                        # Validate captured outputs on EVERY fresh input, outside event timing.
                        for side,captured in sides.items():
                            metrics=validate_output(captured.out,target,probes,layer=='router')
                            if not metrics['passed']:
                                raise RuntimeError(f'Fresh-input replay failed {name}/{m}/{layer}/{side}: {metrics}')
                        sample={'baseline_us':measured['A'],'candidate_us':measured['B'],'order':order}
                        samples.append(sample)
                        emit({'event':'sample','candidate':name,'m':m,'layer':layer,
                              'cohort':cohort,'rep':rep,**sample})
                    result=analyze_pairs(samples,overhead_us=plan.get('overhead_us',0),
                        min_gain_us=plan.get('min_gain_us',.35),min_speedup=plan.get('min_speedup',1.015))
                    all_accepted &= result['accepted']
                    emit({'event':'cohort','candidate':name,'m':m,'layer':layer,'cohort':cohort,**result})
        decisions.append({'name':name,'accepted_all_shapes':all_accepted})
        emit({'event':'candidate_decision',**decisions[-1]})
        del state,candidate
    emit({'event':'finished','decisions':decisions,
          'note':'Local acceptance is not judge proof. No source or leaderboard changed.'})


if __name__=='__main__':
    main()
