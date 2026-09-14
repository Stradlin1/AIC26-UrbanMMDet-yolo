#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Smoke test the Exp04 5ch -> dynamic-expert stem conversion.

Run from project root:
    python scripts/test_exp04_emoe_dynamic_conversion.py

This test intentionally uses the trained stable Exp04 checkpoint because the
main design guarantee is parity at initialization before the router learns.
"""

from __future__ import annotations

from pathlib import Path

import torch
from ultralytics import YOLO

from exp04_emoe_dynamic_model import get_dynamic_stem
from exp04_emoe_dynamic_trainer import EMOEDynamicMultimodalTrainer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_MODEL_PATH = (
    PROJECT_ROOT
    / "runs"
    / "exp04_rgbid_early5_yolo11s_960"
    / "weights"
    / "best.pt"
)
DATA_YAML = PROJECT_ROOT / "yolo_views" / "rgb_v1" / "data_exp04_5ch.yaml"
TMP_RUNS = PROJECT_ROOT / "runs"


def main() -> None:
    if not SOURCE_MODEL_PATH.is_file():
        raise FileNotFoundError(f"Missing Exp04 best checkpoint: {SOURCE_MODEL_PATH}")
    if not DATA_YAML.is_file():
        raise FileNotFoundError(f"Missing data YAML: {DATA_YAML}")

    print("=" * 88)
    print("Exp04 EMOE dynamic conversion smoke test")
    print("=" * 88)

    source_wrapper = YOLO(str(SOURCE_MODEL_PATH))
    source_model = source_wrapper.model.float().eval()
    source_stem = source_model.model[0]

    overrides = {
        "model": str(SOURCE_MODEL_PATH),
        "data": str(DATA_YAML),
        "task": "detect",
        "pretrained": True,
        "imgsz": 960,
        "batch": 1,
        "device": "cpu",
        "workers": 0,
        "cache": False,
        "plots": False,
        "project": str(TMP_RUNS),
        "name": "_smoke_exp04_emoe_dynamic_conversion",
        "exist_ok": True,
    }

    trainer = EMOEDynamicMultimodalTrainer(overrides=overrides)
    trainer.setup_model()
    dynamic_model = trainer.model.float().eval()
    dynamic_stem = get_dynamic_stem(dynamic_model)

    torch.manual_seed(2026)
    x = torch.rand(2, 5, 160, 160, dtype=torch.float32)

    with torch.no_grad():
        source_out = source_stem(x)
        dynamic_out = dynamic_stem(x)
        router_w = dynamic_stem.route(x)

    max_abs_diff = (source_out - dynamic_out).abs().max().item()
    mean_abs_diff = (source_out - dynamic_out).abs().mean().item()

    expected_router = torch.full_like(router_w, 1.0 / 3.0)
    router_diff = (router_w - expected_router).abs().max().item()

    print("Source output shape :", tuple(source_out.shape))
    print("Dynamic output shape:", tuple(dynamic_out.shape))
    print("Stem max abs diff   :", f"{max_abs_diff:.9g}")
    print("Stem mean abs diff  :", f"{mean_abs_diff:.9g}")
    print("Router weights      :", router_w[0].tolist())
    print("Router max diff 1/3 :", f"{router_diff:.9g}")

    if max_abs_diff > 2e-5:
        raise AssertionError(
            "5ch -> dynamic-expert conversion is not sufficiently close: "
            f"max_abs_diff={max_abs_diff}"
        )
    if router_diff > 1e-7:
        raise AssertionError("Router is not exactly neutral/uniform at initialization.")

    dynamic_stem.train()
    dynamic_stem.zero_grad(set_to_none=True)
    y = dynamic_stem(x)
    loss = y.square().mean()
    loss.backward()

    grad = dynamic_stem.router_fc2.weight.grad
    grad_abs_sum = 0.0 if grad is None else grad.abs().sum().item()
    print("Router grad abs sum :", f"{grad_abs_sum:.9g}")

    if grad is None or grad_abs_sum == 0.0:
        raise AssertionError("Router did not receive a nonzero gradient.")

    print("[PASS] source/dynamic stem near-parity")
    print("[PASS] uniform router initialization")
    print("[PASS] router receives gradient")
    print("STATUS = PASS")


if __name__ == "__main__":
    main()
