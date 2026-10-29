"""Untimed correctness helpers for independent projection experiments.

Importing this module does not import torch. Callers compute the reference as
``x.double() @ w.double().T`` (slice w to the requested output width first).
Checks deliberately compare with that unrounded FP64 reference, a stronger
condition than comparison with the judge's output-dtype-rounded reference.
These helpers synchronize and must never be called inside a timed kernel call.
"""
from __future__ import annotations


def validate_output(out, reference_fp64, probe_rows, router: bool) -> dict:
    """Return JSON-compatible metrics and failures; never silently cast output.

    probe_rows may be a sequence of integer indices or a 1D integer/bool tensor.
    Malformed inputs fail validation instead of passing via broadcasting.
    """
    import torch

    tol = 1e-4 if router else 1e-2
    result = {
        "passed": False, "failures": [], "tolerance": tol,
        "row_tolerance": 5 * tol, "probe_tolerance": 1e-3,
        "relative_l2": None, "worst_row_error": None,
        "worst_row": None, "worst_probe_error": None,
        "worst_probe_row": None, "probe_count": 0,
    }
    fail = result["failures"].append
    if not isinstance(reference_fp64, torch.Tensor):
        fail("reference must be a torch.Tensor")
        return result
    if reference_fp64.dtype != torch.float64 or reference_fp64.ndim != 2 or reference_fp64.numel() == 0:
        fail("reference must be a nonempty 2D FP64 tensor")
        return result
    if not isinstance(out, torch.Tensor):
        fail("output must be a torch.Tensor")
        return result
    expected_dtype = torch.float32 if router else torch.bfloat16
    if out.shape != reference_fp64.shape:
        fail(f"shape mismatch: {tuple(out.shape)} != {tuple(reference_fp64.shape)}")
    if out.dtype != expected_dtype:
        fail(f"dtype mismatch: {out.dtype} != {expected_dtype}")
    if out.device != reference_fp64.device:
        fail(f"device mismatch: {out.device} != {reference_fp64.device}")
    if result["failures"]:
        return result
    with torch.no_grad():
        if not bool(torch.isfinite(reference_fp64).all()):
            fail("reference contains NaN or Inf")
            return result
        if not bool(torch.isfinite(out).all()):
            fail("output contains NaN or Inf")
            return result
        diff = out.double() - reference_fp64
        norms = reference_fp64.norm(dim=-1)
        relative = float(diff.norm() / reference_fp64.norm().clamp(min=1e-12))
        floor = (norms.mean() * 0.1).clamp(min=1e-12)
        row_errors = diff.norm(dim=-1) / torch.maximum(norms, floor)
        worst = int(row_errors.argmax())
        result.update(relative_l2=relative, worst_row=worst,
                      worst_row_error=float(row_errors[worst]))
        if relative > tol:
            fail("global relative L2 exceeds tolerance")
        if result["worst_row_error"] > 5 * tol:
            fail("per-row relative L2 exceeds tolerance")
        try:
            rows = torch.as_tensor(probe_rows, device=out.device)
            if rows.ndim != 1:
                raise ValueError("probe_rows must be one-dimensional")
            if rows.dtype == torch.bool:
                if rows.numel() != out.shape[0]:
                    raise ValueError("probe mask length differs from output rows")
                rows = rows.nonzero().flatten()
            elif rows.numel() and rows.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
                raise ValueError("probe indices must be integers")
            rows = rows.to(torch.int64)
            if rows.numel() and (bool((rows < 0).any()) or bool((rows >= out.shape[0]).any())):
                raise ValueError("probe index out of bounds")
        except (TypeError, ValueError, RuntimeError) as exc:
            fail(f"invalid probe rows: {exc}")
            return result
        result["probe_count"] = rows.numel()
        if rows.numel():
            errors = diff[rows].norm(dim=-1) / norms[rows].clamp(min=1e-30)
            worst_probe = int(errors.argmax())
            result.update(worst_probe_error=float(errors[worst_probe]),
                          worst_probe_row=int(rows[worst_probe]))
            if result["worst_probe_error"] > 1e-3:
                fail("one-hot probe relative L2 exceeds tolerance")
        result["passed"] = not result["failures"]
    return result


def build_inputs(m: int, k: int, seed: int, dist: str = "normal", device="cuda"):
    """Return ``(x_bf16, probe_row_indices)`` from a device-local seeded RNG.

    normal/outliers include judge-style random power-of-two one-hot probes;
    onehot makes every row a probe; zero is entirely zero, without probes.
    Outliers multiply 1% of channels by 30 before BF16 conversion.
    """
    import torch

    if not isinstance(m, int) or not isinstance(k, int) or m <= 0 or k <= 0:
        raise ValueError("m and k must be positive integers")
    if dist not in ("normal", "outlier", "outliers", "onehot", "zero"):
        raise ValueError(f"unsupported input distribution: {dist}")
    gen = torch.Generator(device=device).manual_seed(seed)
    if dist == "zero":
        return (torch.zeros((m, k), dtype=torch.bfloat16, device=device),
                torch.empty(0, dtype=torch.int64, device=device))
    x = torch.randn((m, k), generator=gen, device=device)
    if dist in ("outlier", "outliers"):
        channels = torch.randperm(k, generator=gen, device=device)[:max(1, k // 100)]
        x[:, channels] *= 30
    if dist == "onehot":
        mask = torch.ones(m, dtype=torch.bool, device=device)
    else:
        mask = torch.rand(m, generator=gen, device=device) < 1 / 8
        if m >= 8:
            mask[torch.randint(m, (1,), generator=gen, device=device)] = True
    cols = torch.randint(k, (m,), generator=gen, device=device)
    signs = torch.randint(2, (m,), generator=gen, device=device) * 2 - 1
    powers = torch.randint(3, 8, (m,), generator=gen, device=device)
    values = signs * (2.0 ** powers)
    spikes = torch.zeros_like(x).scatter_(1, cols[:, None], values.float()[:, None])
    x = torch.where(mask[:, None], spikes, x).to(torch.bfloat16)
    # Recompute after conversion: covers k=1 and avoids metadata assumptions.
    probes = ((x != 0).sum(dim=-1) == 1).nonzero().flatten()
    return x, probes
