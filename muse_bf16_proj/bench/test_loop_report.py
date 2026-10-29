import copy
import unittest
from bench.loop_report import summarize


def fixture():
    plan={'candidates':{'v':{}},'ms':[128],'layers':['unpooler'],'reps':8,'cohorts':2}
    records=[{'event':'environment','plan':plan,'run_id':'r'}]
    for d in ['normal','outliers','onehot','zero']:
        for side in 'AB':
            for repeat in range(3):
                records.append(dict(event='correctness',candidate='v',m=128,layer='unpooler',dist=d,side=side,repeat=repeat,passed=True,run_id='r'))
    for cohort in range(2):
        for rep in range(8):
            records.append(dict(event='sample',candidate='v',m=128,layer='unpooler',cohort=cohort,rep=rep,order='AB' if rep%2==0 else 'BA',baseline_us=20,candidate_us=18,run_id='r'))
    records.append({'event':'finished','run_id':'r'})
    return records

class ReportTests(unittest.TestCase):
    def test_rebuild_from_raw_samples(self):
        r=summarize(fixture())
        self.assertEqual(r['candidates']['v']['accepted_shapes'],[(128,'unpooler')])
    def test_regression_in_second_cohort_not_accepted(self):
        r=fixture()
        for x in r:
            if x.get('event')=='sample' and x['cohort']==1:x['candidate_us']=22
        self.assertFalse(summarize(r)['candidates']['v']['accepted_shapes'])
    def test_incomplete_or_failed_not_accepted(self):
        r=fixture()
        with self.assertRaises(ValueError):summarize(r[:-1])
        r.append({'event':'run_failed','error':'compile error'})
        with self.assertRaises(ValueError):summarize(r)
    def test_correctness_failure_prevents_approval(self):
        r=fixture();r[1]['passed']=False
        self.assertFalse(summarize(r)['candidates']['v']['all_correct'])
    def test_duplicate_or_missing_sample_rejected(self):
        r=fixture();r.pop(-2)
        with self.assertRaises(ValueError):summarize(r)
    def test_duplicate_correctness_not_accepted(self):
        r=fixture();r.insert(2,copy.deepcopy(r[1]))
        self.assertFalse(summarize(r)['candidates']['v']['all_correct'])

if __name__=='__main__':unittest.main()
