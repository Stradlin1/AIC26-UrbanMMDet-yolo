#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""EMOE-inspired dynamic modality experts for AIC2026 Exp04.

This module adapts the central idea of EMOE (CVPR 2025) to the existing
5-channel YOLO11s detector without copying EMOE's task-specific code.

Input order is fixed to::

    [R, G, B, IR, Depth]

The original Exp04 uses one 5-channel convolution.  This variant decomposes
that convolution into three modality experts and adds a sample-wise router::

    RGB   -> RGB expert ------------------------------+
    IR    -> IR expert ---- x routed IR weight -------+--> shared BN/act
    Depth -> D expert ----- x routed Depth weight ----+

The RGB path is an unconditional anchor.  The router controls only auxiliary
residuals.  This is deliberate: it keeps the strong pretrained/learned RGB
representation from being suppressed by a noisy auxiliary modality.

For a standard 5-channel Exp04 checkpoint, conversion is functionally neutral
at initialization:

    router logits = 0 -> softmax = [1/3, 1/3, 1/3]
    auxiliary_scale = 3

therefore::

    F = F_rgb + 3*(1/3)*F_ir + 3*(1/3)*F_depth
      = F_rgb + F_ir + F_depth

which matches the original 5-channel convolution up to floating-point
summation order.  Training can then learn sample-specific IR/Depth weights.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


RGB_CHANNELS = 3
MULTIMODAL_CHANNELS = 5
MODALITY_NAMES = ("RGB", "IR", "Depth")

STANDARD_FIRST_CONV_KEY = "model.0.conv.weight"
DYNAMIC_RGB_EXPERT_KEY = "model.0.rgb_expert.weight"
DYNAMIC_IR_EXPERT_KEY = "model.0.ir_expert.weight"
DYNAMIC_DEPTH_EXPERT_KEY = "model.0.depth_expert.weight"


class DynamicModalityExpertStem(nn.Module):
    """RGB-anchored, sample-adaptive three-expert input stem.

    The router follows the EMOE principle of producing a unique modality
    distribution for every sample.  For image detection we avoid flattening
    the full 960x960 tensor; instead, adaptive pooling provides a compact
    spatial descriptor before a small MLP router.

    Notes
    -----
    * RGB is always added with coefficient 1.0.
    * IR and Depth are residual experts controlled by the router.
    * ``auxiliary_scale=3`` makes uniform router weights reproduce a normal
      three-expert sum, which is important when converting an existing Exp04
      5-channel checkpoint.
    * The final router linear layer is initialized to zero, so initial routing
      is exactly uniform and deterministic for every sample.
    """

    def __init__(
        self,
        original_block: nn.Module,
        router_pool_size: int = 4,
        router_hidden_dim: int = 32,
        router_temperature: float = 1.0,
        auxiliary_scale: float = 3.0,
    ) -> None:
        super().__init__()

        if not hasattr(original_block, "conv"):
            raise TypeError("Expected the original YOLO stem block to expose '.conv'.")
        if not hasattr(original_block, "bn"):
            raise TypeError("Expected the original YOLO stem block to expose '.bn'.")
        if not hasattr(original_block, "act"):
            raise TypeError("Expected the original YOLO stem block to expose '.act'.")

        source_conv = original_block.conv
        if not isinstance(source_conv, nn.Conv2d):
            raise TypeError(
                "Expected original_block.conv to be nn.Conv2d, "
                f"got {type(source_conv).__name__}."
            )
        if source_conv.in_channels != MULTIMODAL_CHANNELS:
            raise ValueError(
                "DynamicModalityExpertStem expects a 5-channel source stem, "
                f"got in_channels={source_conv.in_channels}."
            )
        if source_conv.groups != 1:
            raise ValueError(
                "Grouped/depthwise first convolutions are not supported; "
                f"got groups={source_conv.groups}."
            )
        if source_conv.bias is not None:
            raise ValueError(
                "The current YOLO stem is expected to use bias=False before BatchNorm."
            )
        if router_pool_size <= 0:
            raise ValueError("router_pool_size must be > 0.")
        if router_hidden_dim <= 0:
            raise ValueError("router_hidden_dim must be > 0.")
        if router_temperature <= 0:
            raise ValueError("router_temperature must be > 0.")
        if auxiliary_scale <= 0:
            raise ValueError("auxiliary_scale must be > 0.")

        conv_kwargs = dict(
            out_channels=source_conv.out_channels,
            kernel_size=source_conv.kernel_size,
            stride=source_conv.stride,
            padding=source_conv.padding,
            dilation=source_conv.dilation,
            groups=1,
            bias=False,
            padding_mode=source_conv.padding_mode,
            device=source_conv.weight.device,
            dtype=source_conv.weight.dtype,
        )

        self.rgb_expert = nn.Conv2d(in_channels=RGB_CHANNELS, **conv_kwargs)
        self.ir_expert = nn.Conv2d(in_channels=1, **conv_kwargs)
        self.depth_expert = nn.Conv2d(in_channels=1, **conv_kwargs)

        # Reuse the exact original BatchNorm and activation modules.  When a
        # pretrained checkpoint is loaded, BN parameters keep their standard
        # state-dict names (model.0.bn.*) and transfer normally.
        self.bn = original_block.bn
        self.act = original_block.act

        self.router_pool_size = int(router_pool_size)
        self.router_hidden_dim = int(router_hidden_dim)
        self.router_temperature = float(router_temperature)
        self.auxiliary_scale = float(auxiliary_scale)

        router_input_dim = MULTIMODAL_CHANNELS * self.router_pool_size**2
        self.router_fc1 = nn.Linear(router_input_dim, self.router_hidden_dim)
        self.router_fc2 = nn.Linear(self.router_hidden_dim, len(MODALITY_NAMES))

        # Deterministic neutral routing.  fc1 can use its default random
        # initialization because the zero final layer makes all logits zero.
        nn.init.zeros_(self.router_fc2.weight)
        nn.init.zeros_(self.router_fc2.bias)

        # Start from the currently constructed 5-channel stem.  This makes the
        # replacement neutral even before a checkpoint is copied in.
        with torch.no_grad():
            self.rgb_expert.weight.copy_(source_conv.weight[:, 0:3])
            self.ir_expert.weight.copy_(source_conv.weight[:, 3:4])
            self.depth_expert.weight.copy_(source_conv.weight[:, 4:5])

        # Ultralytics' graph executor expects modules to carry these fields.
        for attr in ("i", "f"):
            if hasattr(original_block, attr):
                setattr(self, attr, getattr(original_block, attr))
        self.type = self.__class__.__name__
        self.np = sum(p.numel() for p in self.parameters())

        # Runtime-only diagnostics.  Do not register as a buffer: it should not
        # become part of checkpoints or influence state-dict parity.
        self.last_router_mean: torch.Tensor | None = None

    @property
    def in_channels(self) -> int:
        return MULTIMODAL_CHANNELS

    @property
    def out_channels(self) -> int:
        return self.rgb_expert.out_channels

    def route(self, x: torch.Tensor) -> torch.Tensor:
        """Return per-sample softmax weights in RGB/IR/Depth order."""
        if x.ndim != 4 or x.shape[1] != MULTIMODAL_CHANNELS:
            raise RuntimeError(
                "Router expected BCHW 5-channel input, "
                f"received shape={tuple(x.shape)}."
            )

        pooled = F.adaptive_avg_pool2d(
            x,
            output_size=(self.router_pool_size, self.router_pool_size),
        ).flatten(1)

        hidden = F.silu(self.router_fc1(pooled))
        logits = self.router_fc2(hidden) / self.router_temperature
        weights = torch.softmax(logits, dim=1)

        self.last_router_mean = weights.detach().mean(dim=0)
        return weights

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Fuse the three modality experts and apply the original BN/act."""
        weights = self.route(x)

        rgb_feature = self.rgb_expert(x[:, 0:3])
        ir_feature = self.ir_expert(x[:, 3:4])
        depth_feature = self.depth_expert(x[:, 4:5])

        ir_weight = weights[:, 1].view(-1, 1, 1, 1)
        depth_weight = weights[:, 2].view(-1, 1, 1, 1)

        fused = rgb_feature + self.auxiliary_scale * (
            ir_weight * ir_feature + depth_weight * depth_feature
        )

        return self.act(self.bn(fused))

    def router_mean_dict(self) -> dict[str, float] | None:
        """Return the most recent batch-mean router weights for logging."""
        if self.last_router_mean is None:
            return None
        values = self.last_router_mean.detach().float().cpu().tolist()
        return {name: float(value) for name, value in zip(MODALITY_NAMES, values)}


def _source_model_from_weights(weights: Any) -> nn.Module | None:
    if weights is None:
        return None
    source_model = weights.get("model") if isinstance(weights, dict) else weights
    return source_model if isinstance(source_model, nn.Module) else None


def source_state_dict(weights: Any) -> dict[str, torch.Tensor] | None:
    """Extract an FP32 source state dict from an Ultralytics checkpoint object."""
    source_model = _source_model_from_weights(weights)
    if source_model is None:
        return None
    return source_model.float().state_dict()


def detect_source_kind(weights: Any) -> str:
    """Classify the source checkpoint architecture used for initialization."""
    state = source_state_dict(weights)
    if state is None:
        return "none"
    if DYNAMIC_RGB_EXPERT_KEY in state:
        return "dynamic_experts"
    if STANDARD_FIRST_CONV_KEY in state:
        channels = int(state[STANDARD_FIRST_CONV_KEY].shape[1])
        if channels == 3:
            return "standard_3ch"
        if channels == 5:
            return "standard_5ch"
        return f"standard_{channels}ch"
    return "unknown"


def inject_dynamic_expert_stem(
    model: nn.Module,
    router_pool_size: int = 4,
    router_hidden_dim: int = 32,
    router_temperature: float = 1.0,
    auxiliary_scale: float = 3.0,
) -> DynamicModalityExpertStem:
    """Replace YOLO layer 0 with :class:`DynamicModalityExpertStem`."""
    if not hasattr(model, "model") or len(model.model) == 0:
        raise RuntimeError("Detection model has no layer-0 stem to replace.")

    if isinstance(model.model[0], DynamicModalityExpertStem):
        return model.model[0]

    original_block = model.model[0]
    dynamic_stem = DynamicModalityExpertStem(
        original_block=original_block,
        router_pool_size=router_pool_size,
        router_hidden_dim=router_hidden_dim,
        router_temperature=router_temperature,
        auxiliary_scale=auxiliary_scale,
    )
    model.model[0] = dynamic_stem
    return dynamic_stem


def initialize_dynamic_stem_from_source(
    stem: DynamicModalityExpertStem,
    weights: Any,
) -> str:
    """Initialize expert kernels from a standard source checkpoint.

    ``model.load(weights)`` should be called before this helper.  For a
    previously trained dynamic-expert checkpoint, normal state-dict loading
    already restores the router and expert weights and this function leaves
    them untouched.
    """
    state = source_state_dict(weights)
    kind = detect_source_kind(weights)

    if state is None:
        return kind

    if kind == "dynamic_experts":
        # Already restored by DetectionModel.load().  Verify the expected keys
        # exist rather than resetting any learned router/expert parameters.
        for key in (
            DYNAMIC_RGB_EXPERT_KEY,
            DYNAMIC_IR_EXPERT_KEY,
            DYNAMIC_DEPTH_EXPERT_KEY,
        ):
            if key not in state:
                raise RuntimeError(f"Dynamic checkpoint is missing key: {key}")
        return kind

    if STANDARD_FIRST_CONV_KEY not in state:
        raise RuntimeError(
            "Source checkpoint has neither a standard first convolution nor "
            "the dynamic expert stem."
        )

    source_weight = state[STANDARD_FIRST_CONV_KEY]
    if source_weight.ndim != 4:
        raise RuntimeError(
            "Unexpected first-conv tensor shape: "
            f"{tuple(source_weight.shape)}"
        )
    if source_weight.shape[0] != stem.out_channels:
        raise RuntimeError(
            "Source/target first-stem output channels differ: "
            f"source={source_weight.shape[0]}, target={stem.out_channels}."
        )

    source_channels = int(source_weight.shape[1])
    if source_channels not in {3, 5}:
        raise RuntimeError(
            "Only 3-channel COCO or 5-channel Exp04 checkpoints are supported, "
            f"got {source_channels} channels."
        )

    with torch.no_grad():
        stem.rgb_expert.weight.copy_(source_weight[:, 0:3])

        if source_channels == 5:
            stem.ir_expert.weight.copy_(source_weight[:, 3:4])
            stem.depth_expert.weight.copy_(source_weight[:, 4:5])
        else:
            stem.ir_expert.weight.zero_()
            stem.depth_expert.weight.zero_()

        # A standard checkpoint has no router.  Re-establish the deterministic
        # neutral router in case this helper is reused on an existing object.
        stem.router_fc2.weight.zero_()
        stem.router_fc2.bias.zero_()

    return kind


def get_dynamic_stem(model: nn.Module) -> DynamicModalityExpertStem:
    """Return and validate the dynamic stem from a detection model."""
    if not hasattr(model, "model") or len(model.model) == 0:
        raise RuntimeError("Model has no layer-0 stem.")
    stem = model.model[0]
    if not isinstance(stem, DynamicModalityExpertStem):
        raise TypeError(
            "Model layer 0 is not DynamicModalityExpertStem: "
            f"{type(stem).__name__}"
        )
    return stem


__all__ = [
    "RGB_CHANNELS",
    "MULTIMODAL_CHANNELS",
    "MODALITY_NAMES",
    "DynamicModalityExpertStem",
    "detect_source_kind",
    "inject_dynamic_expert_stem",
    "initialize_dynamic_stem_from_source",
    "get_dynamic_stem",
    "source_state_dict",
]
