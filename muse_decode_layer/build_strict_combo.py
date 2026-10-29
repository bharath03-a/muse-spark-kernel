"""Compose checked dense/attention and CPU/GPU-validated BF16 MoE backends."""
from pathlib import Path
import ast,io,tokenize
root=Path(__file__).resolve().parent.parent
source=root/'muse_moe_w4a16/research_moe_full_bf16_gk.py'
text=source.read_text();names=set()
for n in ast.parse(text).body:
 if isinstance(n,(ast.FunctionDef,ast.ClassDef)):names.add(n.name)
 elif isinstance(n,(ast.Assign,ast.AnnAssign)):
  for target in n.targets if isinstance(n,ast.Assign) else [n.target]:
   names.update(x.id for x in ast.walk(target) if isinstance(x,ast.Name))
tokens=[]
for token in tokenize.generate_tokens(io.StringIO(text).readline):
 if token.type==tokenize.NAME and token.string in names:token=token._replace(string='SW_'+token.string)
 tokens.append(token)
embed=tokenize.untokenize(tokens)
s=(root/'muse_decode_layer/candidate_cp14_open_attention.py').read_text()
for old,new in [('"bank": _marlin_bank(weights),','"bank": None,'),('"marlin_mod": _jit_moe_wna16_marlin_module(torch.bfloat16, True, False),','"marlin_mod": None,')]:
 assert s.count(old)==1;s=s.replace(old,new)
a=s.index('        block_size_m = BLOCK_SIZE_M_MAP.get(tokens, 32)',s.index('def kernel_fn(data: Inputs, state: dict):'));b=s.index('    expert_input = _fast_gemm',a)
s=s[:a]+'''        # The persistent backend performs route partitioning within its timed call.
        block_size_m = 16
        sorted_ids = block_experts = n_padded = None

'''+s[b:]
s=s.replace('for tensor in (topk_weights, topk_ids, local_ids, sorted_ids, block_experts, n_padded):','for tensor in (topk_weights, topk_ids, local_ids):')
s+='\n# Apache-2.0 persistent backend, copyright 2026 Lee Penkman; strict BF16 modifications by this research team.\n'+embed
s+='''
_strict_combo_prepare = prepare

def prepare(weights):
    from concurrent.futures import ThreadPoolExecutor
    import time
    start=time.monotonic()
    # Compile independent MoE modules alongside the existing projection preparation.
    with ThreadPoolExecutor(4) as pool:
        builds=[pool.submit(SW__extS,0),pool.submit(SW__extL,0),
                pool.submit(SW__extG,"-DFENCE1 -DRT=16"),pool.submit(SW__extK,"-DSW=4 -DKS2=2")]
        state=_strict_combo_prepare(weights)
        for build in builds:build.result()
    raw={k:weights[k] for k in ("w13","w13_sf","w13_gs","w2","w2_sf","w2_gs")}
    raw["post_norm_w"]=weights["post_expert_norm"]
    state["strict_moe"]=SW_prepare(raw)
    print("strict combo prepare seconds",time.monotonic()-start)
    return state

def _optimized_experts(x,topk_weights,local_ids,sorted_ids,block_experts,n_padded,block_size_m,state):
    return SW_kernel_fn({"hidden_states":x,"topk_ids":local_ids,"topk_weights":topk_weights,
                         "config":{"local_start":0,"local_count":32}},state["strict_moe"])
'''
p=root/'muse_decode_layer/candidate_cp14_attention_strict_moe.py';p.write_text(s);compile(s,str(p),'exec');print(p)
