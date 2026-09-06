#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

import infer_exp04_rgbid_early5_960_submit as submit_base
import infer_exp04_rgbid_early5_960_submit_rect as impl
import test_exp04_modality_ablation_val400 as base
import test_exp04_rect_inference_parity_val400 as rect


PROJECT_ROOT = Path(__file__).resolve().parents[1]

RUN_DIR = (
    PROJECT_ROOT
    / "runs"
    / "exp04_ablation_rgbi_early4_yolo11s_960"
)

MODEL_PATH = RUN_DIR / "weights" / "best.pt"

OUTPUT_ROOT = RUN_DIR / "submission_test_rect"

TXT_DIR = OUTPUT_ROOT / "txt"

ZIP_PATH = (
    OUTPUT_ROOT
    / "exp04_ablation_rgbi_early4_yolo11s_960_submit_rect.zip"
)

EXPECTED_CHANNELS = 4
EXPECTED_CLASSES = 12


def load_rgbi(
    rgb_path: Path,
    ir_path: Path,
    depth_path: Path | None = None,
):
    """
    Build:
        H x W x 4 uint8

    Channel order:
        [R, G, B, IR]

    depth_path is accepted only because the already-tested Rect
    infrastructure passes all three modality paths. It is deliberately
    ignored for this RGBI model.
    """

    rgb = base.load_rgb(rgb_path)
    ir = base.load_ir_gray(ir_path)

    h, w = rgb.shape[:2]

    if ir.shape != (h, w):
        raise RuntimeError(
            "RGB/IR spatial mismatch:\n"
            f"  RGB={rgb_path}: {rgb.shape}\n"
            f"  IR ={ir_path}: {ir.shape}"
        )

    image = np.concatenate(
        [
            rgb,
            ir[..., None],
        ],
        axis=2,
    )

    if image.dtype != np.uint8:
        raise RuntimeError(
            f"RGBI dtype must be uint8, got {image.dtype}"
        )

    if image.shape[2] != EXPECTED_CHANNELS:
        raise RuntimeError(
            f"RGBI channels={image.shape[2]}, expected=4"
        )

    return image, (h, w)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)

            if not chunk:
                break

            digest.update(chunk)

    return digest.hexdigest()


# ============================================================
# Convert the already-tested 5ch infrastructure to RGBI 4ch.
# ============================================================

base.EXPECTED_CHANNELS = EXPECTED_CHANNELS
base.load_5ch = load_rgbi

rect.EXPECTED_CHANNELS = EXPECTED_CHANNELS

impl.EXPECTED_CHANNELS = EXPECTED_CHANNELS

impl.MODEL_PATH = MODEL_PATH
impl.OUTPUT_ROOT = OUTPUT_ROOT
impl.TXT_DIR = TXT_DIR
impl.ZIP_PATH = ZIP_PATH


def main() -> None:

    print("=" * 88)
    print("AIC2026 RGBI 4ch Rect Submission")
    print("=" * 88)

    print("Model       :", MODEL_PATH)
    print("Input       : [R, G, B, IR]")
    print("Channels    :", EXPECTED_CHANNELS)
    print("imgsz       :", rect.IMAGE_SIZE)
    print("rect group  :", rect.RECT_GROUP_BATCH)
    print("forward     :", impl.FORWARD_BATCH)
    print("rect pad    :", rect.RECT_PAD)
    print("conf        :", base.CONF_THRESHOLD)
    print("iou         :", base.IOU_THRESHOLD)
    print("max_det     :", base.MAX_DET)
    print("multi_label : True")
    print("NMS         : per-image")
    print()

    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"RGBI best.pt not found:\n  {MODEL_PATH}"
        )

    # Existing implementation handles:
    #   --check-only
    #   --test-root
    #   test discovery
    #   Rect geometry
    #   model forward
    #   per-image NMS
    #   coordinate restoration
    #   TXT validation
    #   ZIP validation
    impl.main()

    # --check-only stops before ZIP generation.
    if not ZIP_PATH.is_file():
        return

    print()
    print("=" * 88)
    print("FINAL RGBI SUBMISSION")
    print("=" * 88)

    print("Model SHA256:")
    print(sha256_file(MODEL_PATH))

    print()
    print("ZIP SHA256:")
    print(sha256_file(ZIP_PATH))

    print()
    print("Submission ZIP:")
    print(ZIP_PATH)


if __name__ == "__main__":
    main()
