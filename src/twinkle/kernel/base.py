# Copyright (c) ModelScope Contributors. All rights reserved.
"""Kernel module base - 基础类、环境变量、设备检测"""
import os
from typing import Optional, Literal

from ..utils import exists


def _kernels_enabled() -> bool:
    """检查是否启用 kernels（默认启用）"""
    env_val = os.getenv("TWINKLE_USE_KERNELS", "YES").upper()
    return env_val in ("YES", "TRUE", "1", "ON")


def _trust_remote_code() -> bool:
    """检查是否信任远程代码（默认不信任）"""
    env_val = os.getenv("TWINKLE_TRUST_REMOTE_CODE", "NO").upper()
    return env_val in ("YES", "TRUE", "1", "ON")


ModeType = Literal["train", "inference", "compile"]
DeviceType = Literal["cuda", "npu", "mps", "cpu", "rocm", "metal"]


def get_device_type() -> Optional[DeviceType]:
    """自动检测当前设备类型"""
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
    """检测当前使用的训练框架后端: "transformers" | "megatron" | None"""
    if exists("transformers"):
        return "transformers"
    return None


def is_kernels_available() -> bool:
    """检查 HF kernels 包是否可用"""
    return exists("kernels")


def is_kernels_enabled() -> bool:
    """检查 kernels 是否被环境变量启用"""
    return _kernels_enabled() and is_kernels_available()


def to_kernels_mode(mode: ModeType) -> str:
    """将 Twinkle mode 转换为 HF kernels mode"""
    if not is_kernels_available():
        return None
    from kernels import Mode
    mode_map = {
        "train": Mode.TRAINING,
        "inference": Mode.INFERENCE,
        "compile": Mode.TORCH_COMPILE,
    }
    return mode_map.get(mode, Mode.INFERENCE)
