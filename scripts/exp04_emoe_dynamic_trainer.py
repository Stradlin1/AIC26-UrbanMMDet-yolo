#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Custom trainer for the EMOE-inspired Exp04 dynamic-expert detector."""

from __future__ import annotations

from typing import Any

from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils import LOGGER, RANK

from multimodal_dataset import CHANNEL_NAMES, MULTIMODAL_CHANNELS
from multimodal_trainer import MultimodalDetectionTrainer
from exp04_emoe_dynamic_model import (
    DynamicModalityExpertStem,
    detect_source_kind,
    get_dynamic_stem,
    initialize_dynamic_stem_from_source,
    inject_dynamic_expert_stem,
)


class EMOEDynamicMultimodalTrainer(MultimodalDetectionTrainer):
    """Train YOLO11s with a sample-adaptive modality-expert input stem.

    Dataset construction, RGB-only HSV augmentation, synchronous geometry,
    AMP, EMA, loss, validation, checkpointing and DDP remain delegated to the
    already-validated Exp04/Ultralytics pipeline.
    """

    def __init__(
        self,
        *args: Any,
        router_pool_size: int = 4,
        router_hidden_dim: int = 32,
        router_temperature: float = 1.0,
        auxiliary_scale: float = 3.0,
        **kwargs: Any,
    ) -> None:
        self.router_pool_size = int(router_pool_size)
        self.router_hidden_dim = int(router_hidden_dim)
        self.router_temperature = float(router_temperature)
        self.auxiliary_scale = float(auxiliary_scale)
        super().__init__(*args, **kwargs)

    def get_model(
        self,
        cfg: str | dict | None = None,
        weights=None,
        verbose: bool = True,
    ):
        """Build the 5-channel model, inject dynamic experts, then load weights.

        Supported initialization sources:

        * 3-channel YOLO11s COCO checkpoint
        * existing 5-channel Exp04 Early-Fusion checkpoint
        * previously saved dynamic-expert checkpoint (resume/fine-tune)
        """
        source_kind = detect_source_kind(weights)

        model = self.set_model_names_for_load(
            DetectionModel(
                cfg,
                nc=self.data["nc"],
                ch=MULTIMODAL_CHANNELS,
                verbose=verbose and RANK == -1,
            )
        )

        stem = inject_dynamic_expert_stem(
            model,
            router_pool_size=self.router_pool_size,
            router_hidden_dim=self.router_hidden_dim,
            router_temperature=self.router_temperature,
            auxiliary_scale=self.auxiliary_scale,
        )

        if weights is not None:
            # All compatible backbone/neck/head weights, plus dynamic stem
            # parameters when resuming a dynamic checkpoint, are restored here.
            model.load(weights)

            # Standard 3ch/5ch checkpoints have a different first-stem key, so
            # explicitly decompose that kernel into modality experts.
            initialize_dynamic_stem_from_source(stem, weights)
        else:
            LOGGER.warning(
                "EMOE dynamic trainer received no pretrained weights. "
                "This is allowed for debugging but is not the formal Exp04 path."
            )

        stem = get_dynamic_stem(model)
        detect_head = model.model[-1]
        head_nc = getattr(detect_head, "nc", None)
        expected_nc = self.data["nc"]

        if stem.in_channels != MULTIMODAL_CHANNELS:
            raise AssertionError(
                f"Dynamic stem input channels={stem.in_channels}, expected 5."
            )
        if head_nc != expected_nc:
            raise AssertionError(
                f"Detection head class count mismatch: head={head_nc}, data={expected_nc}."
            )

        LOGGER.info(
            "Exp04 EMOE dynamic experts ready: "
            f"source={source_kind}, channels={MULTIMODAL_CHANNELS} {CHANNEL_NAMES}, "
            f"router_pool={self.router_pool_size}, "
            f"router_hidden={self.router_hidden_dim}, "
            f"temperature={self.router_temperature}, "
            f"aux_scale={self.auxiliary_scale}"
        )
        LOGGER.info(
            "Fusion rule: RGB anchor + routed IR/Depth residual experts. "
            "Uniform router + aux_scale=3 reproduces the original 5ch stem."
        )

        return model

    def dynamic_stem(self) -> DynamicModalityExpertStem:
        """Convenience accessor for diagnostics."""
        return get_dynamic_stem(self.model)


__all__ = ["EMOEDynamicMultimodalTrainer"]
