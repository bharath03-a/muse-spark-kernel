"""CPU-only: reconstruct decisions from raw loop samples, not claimed summaries.
Projected scores transfer local absolute gains onto a saved judge profile;
they are explicitly estimates, never judge results or promotion authority.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import statistics
try:
    from .loop_stats import analyze_pairs,project_judge_score
except ImportError:
    from loop_stats import analyze_pairs,project_judge_score


def summarize(records, profile=None):
    if not records:
        raise ValueError('Empty experiment log')
    if records[-1].get('event')=='run_failed':
        raise ValueError('Latest run failed; partial samples cannot approve a candidate')
    run_id=records[-1].get('run_id')
    if not run_id:
        raise ValueError('Latest event missing run_id')
    run=[r for r in records if r.get('run_id')==run_id]
    env=[r for r in run if r.get('event')=='environment']
    if len(env)!=1 or run[-1].get('event')!='finished':
        raise ValueError('Incomplete or ambiguous run')
    plan=env[0]['plan']
    expected_cohorts=plan.get('cohorts',2)
    if expected_cohorts<2:
        raise ValueError('Two independent cohorts required')
    report={'run_id':run_id,'estimated_not_judge':True,'candidates':{}}
    for name in plan['candidates']:
        shapes=[];point_gains={};lower_gains={}
        for m in plan['ms']:
            for layer in plan['layers']:
                selected=[r for r in run if r.get('candidate')==name and r.get('m')==m and r.get('layer')==layer]
                checks=[r for r in selected if r.get('event')=='correctness']
                expected={(d,side,rep) for d in ['normal','outliers','onehot','zero'] for side in 'AB' for rep in range(3)}
                observed={(r['dist'],r['side'],r['repeat']) for r in checks}
                if len(checks)!=len(expected) or observed!=expected or not all(r.get('passed') is True for r in checks):
                    shapes.append({'m':m,'layer':layer,'accepted':False,'reason':'correctness gate incomplete or failed'})
                    continue
                analyses=[]
                for cohort in range(expected_cohorts):
                    samples=[r for r in selected if r.get('event')=='sample' and r.get('cohort')==cohort]
                    if len(samples)!=plan.get('reps',40) or [r.get('rep') for r in samples]!=list(range(plan.get('reps',40))):
                        raise ValueError(f'Incomplete/duplicate samples: {name}/{m}/{layer}/{cohort}')
                    analyses.append(analyze_pairs(samples,overhead_us=plan.get('overhead_us',0),
                        min_gain_us=plan.get('min_gain_us',.35),min_speedup=plan.get('min_speedup',1.015)))
                accepted=all(a['accepted'] for a in analyses)
                shapes.append({'m':m,'layer':layer,'accepted':accepted,'cohorts':analyses})
                point_gains[(m,layer)]=statistics.median(a['gain_us'] for a in analyses)
                lower_gains[(m,layer)]=min(a['gain_us_ci95'][0] for a in analyses)
        candidate={'shapes':shapes,'accepted_shapes':[(s['m'],s['layer']) for s in shapes if s['accepted']],
                   'all_correct':all('reason' not in s for s in shapes)}
        if profile is not None and candidate['all_correct']:
            # Include measured regressions; do not cherry-pick a full candidate's projection.
            candidate['score_point_estimate']=project_judge_score(profile,point_gains)
            candidate['score_lower_gain_scenario']=project_judge_score(profile,lower_gains)
        report['candidates'][name]=candidate
    return report


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('log')
    ap.add_argument('--judge-profile')
    args=ap.parse_args()
    records=[json.loads(line) for line in Path(args.log).read_text().splitlines() if line.strip()]
    profile=json.loads(Path(args.judge_profile).read_text()) if args.judge_profile else None
    print(json.dumps(summarize(records,profile),indent=2,allow_nan=False))

if __name__=='__main__':main()
