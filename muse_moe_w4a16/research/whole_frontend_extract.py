"""External research only: AST-copy verified whole frontend, stop before experts.
Never submit this harness. It does not modify the source module or installed packages.
"""
from pathlib import Path
import ast
import hashlib
import types
import sys


def load_frontend(path):
    path = Path(path)
    source = path.read_text()
    tree = ast.parse(source)
    # Keep original whole+projection+attention code, omit embedded MoE and wrappers.
    marker = source.index('# okc: problem=muse_moe_w4a16')
    cutoff = source[:marker].count('\n') + 1
    nodes = [node for node in tree.body if node.lineno < cutoff]
    functions = [n for n in nodes if isinstance(n, ast.FunctionDef) and n.name == 'kernel_fn']
    assert len(functions) == 1
    fn = functions[0]
    stop = next(i for i, stmt in enumerate(fn.body)
                if isinstance(stmt, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'expert_output' for t in stmt.targets))
    assert any(isinstance(n, ast.Name) and n.id == 'normed_expert_in' for n in ast.walk(ast.Module(body=fn.body[:stop],type_ignores=[])))
    fn.name = 'extract_frontend'
    fn.body = fn.body[:stop] + ast.parse("return {'hidden_states': normed_expert_in, 'topk_ids': local_ids, 'topk_weights': topk_weights, 'config': {'local_start': 0, 'local_count': 32}} ").body
    module = types.ModuleType('research_whole_frontend')
    module.__file__ = str(path)
    sys.modules[module.__name__] = module
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])), str(path), 'exec'), module.__dict__)
    module.source_sha256 = hashlib.sha256(source.encode()).hexdigest()
    return module


def load_compile_only(path):
    """Load only literal CUDA/build constants and standalone builder definitions on CPU.
    Avoid SGLang runtime imports that require libcuda.so.1; compile identical source/flags.
    """
    source=Path(path).read_text();tree=ast.parse(source)
    namespace={'hashlib':hashlib,'cp14_hashlib':hashlib}
    import torch
    from torch.utils.cpp_extension import load_inline
    namespace.update(torch=torch,load_inline=load_inline)
    builders={'_build_cuda','cp14__build_cuda','cp14__build_zip','cp14__r9_build','cp14__r9_gen_wgmma','cp14__r15_build','cp14__r15_cublas_includes','OKA__ext'}
    for node in tree.body:
        if isinstance(node,ast.Assign):
            try:value=ast.literal_eval(node.value)
            except (ValueError,TypeError):
                if any(isinstance(t,ast.Name) and t.id=='cp14_R9_SRC' for t in node.targets):
                    exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),namespace)
                continue
            for target in node.targets:
                if isinstance(target,ast.Name):namespace[target.id]=value
        if isinstance(node,ast.FunctionDef) and node.name in builders:
            exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),namespace)
    namespace['OKA__big']=lambda:True
    namespace['OKA__gencode']=lambda:['-gencode=arch=compute_90a,code=sm_90a']
    namespace['source_sha256']=hashlib.sha256(source.encode()).hexdigest()
    return types.SimpleNamespace(**namespace)
