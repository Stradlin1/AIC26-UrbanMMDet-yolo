#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""AIC2026 Exp04 - EMOE-inspired dynamic modality experts.

This is an additive experimental variant. It does not overwrite the stable
Exp04 Early-Fusion run.

Default initialization:
    runs/exp04_rgbid_early5_yolo11s_960/weights/best.pt

Why warm-start from Exp04 best:
    The 5-channel first convolution can be decomposed exactly into RGB/IR/Depth
    expert convolutions. With a zero-logit router and auxiliary_scale=3, the
    new stem starts as a near-parity transformation of the existing Exp04
    model and then learns sample-specific auxiliary modality weights.

Run from project root:
    python scripts/train_exp04_emoe_dynamic_yolo11s_960.py
"""

from __future__ import annotations

from pathlib import Path

import torch
import ultralytics

from exp04_emoe_dynamic_model import MODALITY_NAMES, get_dynamic_stem
from exp04_emoe_dynamic_trainer import EMOEDynamicMultimodalTrainer
from multimodal_dataset import CHANNEL_NAMES, MULTIMODAL_CHANNELS


PROJECT_ROOT = Path(__file__).resolve().parents[1]

SOURCE_MODEL_PATH = (
    PROJECT_ROOT
    / "runs"
    / "exp04_rgbid_early5_yolo11s_960"
    / "weights"
    / "best.pt"
)

DATA_YAML = (
    PROJECT_ROOT
    / "yolo_views"
    / "rgb_v1"
    / "data_exp04_5ch.yaml"
)

RUNS_DIR = PROJECT_ROOT / "runs"
EXPERIMENT_NAME = "exp04_emoe_dynamic_experts_yolo11s_960"
RUN_DIR = RUNS_DIR / EXPERIMENT_NAME

ROUTER_POOL_SIZE = 4
ROUTER_HIDDEN_DIM = 32
ROUTER_TEMPERATURE = 1.0
AUXILIARY_SCALE = 3.0

IMAGE_SIZE = 960
BATCH_SIZE = 8
EPOCHS = 60
PATIENCE = 15
SEED = 2026
WORKERS = 4
MAX_DET_TRAIN_VAL = 300

OPTIMIZER = "AdamW"
LR0 = 5e-4
LRF = 0.10
WEIGHT_DECAY = 5e-4
WARMUP_EPOCHS = 1.0


def section(title: str) -> None:
    print()
    print("=" * 88)
    print(title)
    print("=" * 88)


def preflight() -> None:
    section("Exp04 EMOE Dynamic Experts - Preflight")

    if not SOURCE_MODEL_PATH.is_file():
        raise FileNotFoundError(
            "The stable Exp04 best checkpoint is required for the default "
            "parity warm-start:\n"
            f"  {SOURCE_MODEL_PATH}\n\n"
            "Copy/restore the Exp04 run before launching this experiment."
        )
    if not DATA_YAML.is_file():
        raise FileNotFoundError(f"Dataset YAML not found:\n  {DATA_YAML}")
    if RUN_DIR.exists():
        raise RuntimeError(
            "Target experiment directory already exists; refusing to mix runs:\n"
            f"  {RUN_DIR}"
        )
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; formal training should not start on CPU.")

    active_ultralytics = Path(ultralytics.__file__).resolve()
    expected_ultralytics_root = (PROJECT_ROOT / "ultralytics").resolve()
    try:
        active_ultralytics.relative_to(expected_ultralytics_root)
    except ValueError as exc:
        raise RuntimeError(
            "Active Ultralytics is not the project-local editable source.\n"
            f"Active:   {active_ultralytics}\n"
            f"Expected: {expected_ultralytics_root}"
        ) from exc

    RUNS_DIR.mkdir(parents=True, exist_ok=True)

    print("Project root       :", PROJECT_ROOT)
    print("Ultralytics        :", active_ultralytics)
    print("Source checkpoint  :", SOURCE_MODEL_PATH)
    print("Dataset YAML       :", DATA_YAML)
    print("Output             :", RUN_DIR)
    print("Input channels     :", CHANNEL_NAMES)
    print("Modalities         :", MODALITY_NAMES)
    print("Router pool        :", ROUTER_POOL_SIZE)
    print("Router hidden      :", ROUTER_HIDDEN_DIM)
    print("Router temperature :", ROUTER_TEMPERATURE)
    print("Auxiliary scale    :", AUXILIARY_SCALE)
    print("Image size         :", IMAGE_SIZE)
    print("Batch              :", BATCH_SIZE)
    print("Epochs             :", EPOCHS)
    print("Optimizer          :", OPTIMIZER)
    print("lr0                :", LR0)
    print("GPU                :", torch.cuda.get_device_name(0))
    print("[PASS] project-local editable Ultralytics")


def build_trainer() -> EMOEDynamicMultimodalTrainer:
    overrides = {
        "model": str(SOURCE_MODEL_PATH),
        "data": str(DATA_YAML),
        "task": "detect",
        "pretrained": True,
        "cls_remap": True,
        "epochs": EPOCHS,
        "patience": PATIENCE,
        "imgsz": IMAGE_SIZE,
        "batch": BATCH_SIZE,
        "rect": False,
        "multi_scale": 0.0,
        "device": "0",
        "workers": WORKERS,
        "cache": False,
        "amp": True,
        "optimizer": OPTIMIZER,
        "lr0": LR0,
        "lrf": LRF,
        "weight_decay": WEIGHT_DECAY,
        "warmup_epochs": WARMUP_EPOCHS,
        "warmup_bias_lr": 0.0,
        "seed": SEED,
        "deterministic": True,
        "mosaic": 1.0,
        "close_mosaic": 10,
        "fliplr": 0.5,
        "flipud": 0.0,
        "mixup": 0.0,
        "cutmix": 0.0,
        "copy_paste": 0.0,
        "val": True,
        "iou": 0.70,
        "max_det": MAX_DET_TRAIN_VAL,
        "project": str(RUNS_DIR),
        "name": EXPERIMENT_NAME,
        "exist_ok": False,
        "save": True,
        "save_period": -1,
        "plots": False,
    }

    return EMOEDynamicMultimodalTrainer(
        overrides=overrides,
        router_pool_size=ROUTER_POOL_SIZE,
        router_hidden_dim=ROUTER_HIDDEN_DIM,
        router_temperature=ROUTER_TEMPERATURE,
        auxiliary_scale=AUXILIARY_SCALE,
    )


def verify_constructed_model(trainer: EMOEDynamicMultimodalTrainer) -> None:
    section("Construct and verify dynamic-expert model")
    trainer.setup_model()
    stem = get_dynamic_stem(trainer.model)

    print("Stem class         :", type(stem).__name__)
    print("RGB expert weight  :", tuple(stem.rgb_expert.weight.shape))
    print("IR expert weight   :", tuple(stem.ir_expert.weight.shape))
    print("Depth expert weight:", tuple(stem.depth_expert.weight.shape))
    print("Router fc1         :", tuple(stem.router_fc1.weight.shape))
    print("Router fc2         :", tuple(stem.router_fc2.weight.shape))

    if stem.in_channels != MULTIMODAL_CHANNELS:
        raise AssertionError("Dynamic stem is not 5-channel.")

    with torch.no_grad():
        if torch.count_nonzero(stem.router_fc2.weight).item() != 0:
            raise AssertionError("Router output weights are not zero-initialized.")
        if torch.count_nonzero(stem.router_fc2.bias).item() != 0:
            raise AssertionError("Router output bias is not zero-initialized.")

    print("[PASS] dynamic expert stem")
    print("[PASS] neutral sample-wise router initialization")


def main() -> None:
    preflight()
    trainer = build_trainer()
    verify_constructed_model(trainer)

    section("TRAIN Exp04 EMOE Dynamic Experts")
    trainer.train()

    section("Training finished")
    run_dir = Path(trainer.save_dir).resolve()
    best_path = run_dir / "weights" / "best.pt"
    last_path = run_dir / "weights" / "last.pt"
    results_csv = run_dir / "results.csv"

    for path in (best_path, last_path, results_csv):
        if not path.is_file():
            raise AssertionError(f"Required output missing: {path}")
        print("[PASS]", path)

    print()
    print("Next comparison protocol:")
    print("  imgsz=960, conf=0.001, iou=0.70, max_det=100, fixed val400")
    print("Do not compare only the trainer's internal max_det=300 validation.")


if __name__ == "__main__":
    main()
