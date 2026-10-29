# okc: problem=muse_moe_w4a16
import flashinfer.fused_moe as fi_moe

def prepare(weights):
    print("=== fi_moe attributes ===")
    print([x for x in dir(fi_moe) if not x.startswith('_')])
    if hasattr(fi_moe, 'QuantFormat'):
        print("QuantFormat:", dir(fi_moe.QuantFormat))
    if hasattr(fi_moe, 'DataType'):
        print("DataType:", dir(fi_moe.DataType))

    return {}

def kernel_fn(data, state):
    return data["hidden_states"]
