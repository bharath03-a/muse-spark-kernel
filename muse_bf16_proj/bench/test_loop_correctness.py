"""CPU validation of numerical gates; does not validate GPU kernels."""
import unittest
import torch
from bench.loop_correctness import build_inputs, validate_output


class CorrectnessGateTests(unittest.TestCase):
    def test_input_distributions_and_freshness(self):
        for dist in ('normal','outliers','onehot','zero'):
            x, probes = build_inputs(16,64,123,dist,'cpu')
            same, _ = build_inputs(16,64,123,dist,'cpu')
            self.assertEqual(x.dtype,torch.bfloat16)
            self.assertTrue(torch.equal(x,same))
            self.assertTrue(torch.equal(probes,((x!=0).sum(1)==1).nonzero().flatten()))
            if dist=='onehot':self.assertEqual(probes.numel(),16)
            if dist=='zero':self.assertEqual(x.count_nonzero(),0)
        a,_=build_inputs(16,64,123,device='cpu')
        b,_=build_inputs(16,64,124,device='cpu')
        self.assertFalse(torch.equal(a,b))

    def test_bf16_exact_weight_probes_pass(self):
        x,probes=build_inputs(16,64,123,'onehot','cpu')
        w=torch.randn(32,64).bfloat16()
        ref=x.double()@w.double().T
        self.assertTrue(validate_output(ref.bfloat16(),ref,probes,False)['passed'])

    def test_router_tolerance_stricter(self):
        ref=torch.ones(8,8,dtype=torch.float64)
        self.assertFalse(validate_output(torch.full((8,8),1.0005),ref,[],True)['passed'])
        self.assertTrue(validate_output(ref.float(),ref,[],True)['passed'])

    def test_probe_gate_catches_small_corruption(self):
        ref=torch.ones(128,8,dtype=torch.float64)
        out=ref.bfloat16();out[0]*=1.0078125
        result=validate_output(out,ref,[0],False)
        self.assertLess(result['relative_l2'],.01)
        self.assertFalse(result['passed'])
        self.assertIn('one-hot probe relative L2 exceeds tolerance',result['failures'])

    def test_row_gate_catches_sparse_corruption(self):
        ref=torch.ones(1024,8,dtype=torch.float64)
        out=ref.bfloat16();out[0]*=1.125
        result=validate_output(out,ref,[],False)
        self.assertLess(result['relative_l2'],.01)
        self.assertFalse(result['passed'])
        self.assertIn('per-row relative L2 exceeds tolerance',result['failures'])

    def test_zero_and_invalid_outputs(self):
        ref=torch.zeros(8,8,dtype=torch.float64)
        self.assertTrue(validate_output(ref.bfloat16(),ref,[],False)['passed'])
        for out in (torch.zeros(8,7,dtype=torch.bfloat16),ref.float(),torch.full((8,8),float('nan'),dtype=torch.bfloat16)):
            self.assertFalse(validate_output(out,ref,[],False)['passed'])

    def test_bad_probe_indices_rejected(self):
        ref=torch.ones(8,8,dtype=torch.float64)
        for probes in ([8],[.5],torch.ones(7,dtype=torch.bool)):
            self.assertFalse(validate_output(ref.bfloat16(),ref,probes,False)['passed'])

if __name__=='__main__':unittest.main()
