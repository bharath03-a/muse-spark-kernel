"""Shared helpers for the local judge replica: module loading, shapes, weights."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch

PROJ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ))

import reference  # noqa: E402

LAYER_ORDER = ("qkv_gate", "o", "pooler", "unpooler", "router")
LOADS = {
    "decode 1": 1,
    "decode 16": 16,
    "decode 32": 32,
    "verify 1": 8,
    "verify 16": 128,
    "verify 32": 256,
}
TESTS = [
    (1, "qkv_gate", "normal"),
    (7, "pooler", "normal"),
    (33, "o", "normal"),
    (130, "unpooler", "normal"),
    (5, "unpooler", "normal"),
    (8, "router", "normal"),
    (9, "router", "outliers"),
    (512, "qkv_gate", "outliers"),
    (1000, "unpooler", "outliers"),
]


def load_module(path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, PROJ / path)
    if spec is None or spec.loader is None:
        raise FileNotFoundError(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def weights(seed: int = reference.WEIGHTS_SEED) -> dict[str, torch.Tensor]:
    return reference.generate_weights(seed)


def scored_shapes() -> list[tuple[str, int, str]]:
    return [(load, m, layer) for load, m in LOADS.items() for layer in LAYER_ORDER]
