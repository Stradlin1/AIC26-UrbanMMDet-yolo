#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import json
from copy import copy
from pathlib import Path

import torch
from ultralytics import YOLO
from ultralytics.models.yolo.detect.val import DetectionValidator

from multimodal_config import (
    channel_names_for_modalities,
    channels_for_modalities,
)
from multimodal_trainer import MultimodalDetectionTrainer


MODALITIES = ("rgb", "ir")
CHANNEL_NAMES = channel_names_for_modalities(MODALITIES)
EXPECTED_CHANNELS = channels_for_modalities(MODALITIES)

EXPECTED_CLASSES = 12
EXPECTED_VAL_IMAGES = 400

IMAGE_SIZE = 960
BATCH_SIZE = 8
CONF = 0.001
IOU = 0.70
MAX_DET = 100


PROJECT_ROOT = Path(__file__).resolve().parents[1]

RUN_DIR = (
    PROJECT_ROOT
    / "runs"
    / "exp04_ablation_rgbi_early4_yolo11s_960"
)

MODEL_PATH = RUN_DIR / "weights" / "best.pt"

DATA_YAML = (
    PROJECT_ROOT
    / "yolo_views"
    / "rgb_v1"
    / "data_exp04_rgbi_4ch.yaml"
)

VAL_DIR = RUN_DIR / "val_official_960"
METRICS_JSON = VAL_DIR / "official_metrics.json"


def section(title: str) -> None:
    print()
    print("=" * 80)
    print(title)
    print("=" * 80)


def main() -> None:
    section("RGBI 4ch Official Val - Preflight")

    if not MODEL_PATH.is_file():
        raise FileNotFoundError(MODEL_PATH)

    if not DATA_YAML.is_file():
        raise FileNotFoundError(DATA_YAML)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable.")

    VAL_DIR.mkdir(parents=True, exist_ok=True)

    print("Model      :", MODEL_PATH)
    print("Dataset    :", DATA_YAML)
    print("Modalities :", MODALITIES)
    print("Channels   :", CHANNEL_NAMES)
    print("imgsz      :", IMAGE_SIZE)
    print("batch      :", BATCH_SIZE)
    print("rect       :", True)
    print("conf       :", CONF)
    print("iou        :", IOU)
    print("max_det    :", MAX_DET)
    print("GPU        :", torch.cuda.get_device_name(0))

    section("Load RGBI best.pt")

    wrapper = YOLO(str(MODEL_PATH))
    model = wrapper.model

    first_conv = model.model[0].conv
    detect_head = model.model[-1]

    print("First conv weight:", tuple(first_conv.weight.shape))
    print("Detect nc        :", getattr(detect_head, "nc", None))

    if first_conv.in_channels != EXPECTED_CHANNELS:
        raise RuntimeError(
            f"Expected {EXPECTED_CHANNELS} channels, "
            f"got {first_conv.in_channels}"
        )

    if getattr(detect_head, "nc", None) != EXPECTED_CLASSES:
        raise RuntimeError("Detect head is not 12-class.")

    print("[PASS] RGBI 4-channel checkpoint")
    print("[PASS] 12-class checkpoint")

    section("Build RGBI val400 DataLoader")

    overrides = {
        "model": str(MODEL_PATH),
        "data": str(DATA_YAML),
        "task": "detect",
        "imgsz": IMAGE_SIZE,
        "batch": BATCH_SIZE,
        "device": "0",
        "workers": 4,
        "cache": False,
        "rect": True,
        "conf": CONF,
        "iou": IOU,
        "max_det": MAX_DET,
        "plots": False,
        "save_json": False,
        "save_txt": False,
        "project": str(RUN_DIR),
        "name": "val_official_960",
        "exist_ok": True,
    }

    trainer = MultimodalDetectionTrainer(
        overrides=overrides,
        modalities=MODALITIES,
    )

    trainer.model = wrapper.model.to(trainer.device)
    trainer.set_model_attributes()

    if trainer.data.get("channels") != EXPECTED_CHANNELS:
        raise RuntimeError(
            f"Dataset channels={trainer.data.get('channels')}, expected=4"
        )

    if trainer.data.get("nc") != EXPECTED_CLASSES:
        raise RuntimeError(
            f"Dataset nc={trainer.data.get('nc')}, expected=12"
        )

    val_loader = trainer.get_dataloader(
        trainer.data["val"],
        batch_size=BATCH_SIZE,
        rank=-1,
        mode="val",
    )

    dataset = val_loader.dataset

    print("Dataset class :", type(dataset).__name__)
    print("Samples       :", len(dataset))
    print("Modalities    :", getattr(dataset, "modalities", None))

    if type(dataset).__name__ != "MultimodalYOLODataset":
        raise RuntimeError("Not using MultimodalYOLODataset.")

    if len(dataset) != EXPECTED_VAL_IMAGES:
        raise RuntimeError(
            f"Val samples={len(dataset)}, expected={EXPECTED_VAL_IMAGES}"
        )

    first_batch = next(iter(val_loader))
    images = first_batch["img"]

    print("First batch   :", tuple(images.shape))
    print("dtype         :", images.dtype)

    if images.ndim != 4 or images.shape[1] != EXPECTED_CHANNELS:
        raise RuntimeError(
            f"Expected BCHW 4-channel batch, got {tuple(images.shape)}"
        )

    print("[PASS] fixed val400")
    print("[PASS] RGBI MultimodalYOLODataset")
    print("[PASS] 4-channel batch")

    section("RUN RGBI OFFICIAL VAL")

    validator_args = copy(trainer.args)

    validator_args.mode = "val"
    validator_args.model = str(MODEL_PATH)
    validator_args.data = str(DATA_YAML)
    validator_args.split = "val"
    validator_args.imgsz = IMAGE_SIZE
    validator_args.batch = BATCH_SIZE
    validator_args.rect = True
    validator_args.conf = CONF
    validator_args.iou = IOU
    validator_args.max_det = MAX_DET
    validator_args.device = "0"
    validator_args.plots = False
    validator_args.save_json = False
    validator_args.save_txt = False

    validator = DetectionValidator(
        dataloader=val_loader,
        save_dir=VAL_DIR,
        args=validator_args,
    )

    validator(model=str(MODEL_PATH))

    box = validator.metrics.box

    precision = float(box.mp)
    recall = float(box.mr)
    map50 = float(box.map50)
    map75 = float(box.map75)
    map5095 = float(box.map)

    names = trainer.data["names"]

    per_class = {}

    maps = box.maps

    for class_id in range(EXPECTED_CLASSES):
        class_name = names[class_id]
        per_class[str(class_id)] = {
            "name": class_name,
            "map50_95": float(maps[class_id]),
        }

    result = {
        "experiment": "exp04_ablation_rgbi_early4_yolo11s_960",
        "modalities": list(MODALITIES),
        "channels": list(CHANNEL_NAMES),
        "protocol": {
            "imgsz": IMAGE_SIZE,
            "batch": BATCH_SIZE,
            "rect": True,
            "conf": CONF,
            "iou": IOU,
            "max_det": MAX_DET,
            "val_images": EXPECTED_VAL_IMAGES,
        },
        "metrics": {
            "precision": precision,
            "recall": recall,
            "map50": map50,
            "map75": map75,
            "map5095": map5095,
        },
        "per_class": per_class,
    }

    METRICS_JSON.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    section("RGBI OFFICIAL VAL RESULT")

    print(f"Precision : {precision:.6f}")
    print(f"Recall    : {recall:.6f}")
    print(f"mAP50     : {map50:.6f}")
    print(f"mAP75     : {map75:.6f}")
    print(f"mAP50-95  : {map5095:.6f}")

    print()
    print("Saved:")
    print(METRICS_JSON)

    section("FINAL")
    print("RGBI best.pt             PASS")
    print("RGBI 4-channel           PASS")
    print("fixed val400             PASS")
    print("Rect 960                 PASS")
    print("conf=0.001               PASS")
    print("iou=0.70                 PASS")
    print("max_det=100              PASS")


if __name__ == "__main__":
    main()
