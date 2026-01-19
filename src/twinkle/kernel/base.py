# Copyright (c) ModelScope Contributors. All rights reserved.
"""Kernel module base - Base classes, env vars, device detection."""
import os
from typing import Optional, Literal

from ..utils import exists


def _kernels_enabled() -> bool:
    """Check if kernels are enabled (default: enabled)."""
    env_val = os.getenv("TWINKLE_USE_KERNELS", "YES").upper()
    return env_val in ("YES", "TRUE", "1", "ON")


def _trust_remote_code() -> bool:
    """Check if remote code is trusted (default: not trusted)."""
    env_val = os.getenv("TWINKLE_TRUST_REMOTE_CODE", "NO").upper()
    return env_val in ("YES", "TRUE", "1", "ON")


ModeType = Literal["train", "inference", "compile"]
DeviceType = Literal["cuda", "npu", "mps", "cpu", "rocm", "metal"]


def get_device_type() -> Optional[DeviceType]:
    """Auto-detect current device type."""
    if exists("torch"):
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch, "npu") and torch.npu.is_available():
            return "npu"
        if hasattr(torch, "backends") and hasattr(torch.backends, "mps"):
            if torch.backends.mps.is_available():
                return "mps"
    return None


def detect_backend() -> Optional[str]:
    """Detect training framework backend: "transformers" | "megatron" | None."""
    if exists("transformers"):
        return "transformers"
    return None


def is_kernels_available() -> bool:
    """Check if HF kernels package is available."""
    return exists("kernels")


def is_kernels_enabled() -> bool:
    """Check if kernels are enabled by env var."""
    return _kernels_enabled() and is_kernels_available()


def to_kernels_mode(mode: ModeType) -> str:
    """Convert Twinkle mode to HF kernels mode."""
    if not is_kernels_available():
        return None
    from kernels import Mode
    mode_map = {
        "train": Mode.TRAINING,
        "inference": Mode.INFERENCE,
        "compile": Mode.TORCH_COMPILE,
    }
    return mode_map.get(mode, Mode.INFERENCE)
