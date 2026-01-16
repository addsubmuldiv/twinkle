# Copyright (c) ModelScope Contributors. All rights reserved.
"""Kernel module layer - 层级别替换（HF kernels 集成）"""
from logging import getLogger
from pathlib import Path
from typing import Optional, Union, Any

from .base import (
    DeviceType,
    ModeType,
    is_kernels_available,
    is_kernels_enabled,
    to_kernels_mode,
    get_device_type,
)
from .registry import register_layer

logger = getLogger(__name__)


def register_layer_kernel(
    kernel_name: str,
    repo_id: Optional[str] = None,
    repo_path: Optional[Union[str, Path]] = None,
    package_name: Optional[str] = None,
    layer_name: Optional[str] = None,
    version: Optional[str] = None,
    device: DeviceType = "cuda",
    mode: Optional[ModeType] = None,
) -> None:
    """注册层级别 kernel，支持从 HuggingFace Hub 或本地路径加载"""
    if not is_kernels_available():
        logger.warning(f"HF kernels package not available. Skipping registration for kernel: {kernel_name}")
        return

    from kernels import LayerRepository, LocalLayerRepository

    if repo_path is not None:
        if package_name is None:
            raise ValueError(f"package_name must be provided when using repo_path for kernel: {kernel_name}")
        if isinstance(repo_path, str):
            repo_path = Path(repo_path)
        repo_spec = LocalLayerRepository(
            repo_path=repo_path,
            package_name=package_name,
            layer_name=layer_name or kernel_name,
        )
    else:
        if repo_id is None:
            raise ValueError(f"Either repo_id or repo_path must be provided for kernel: {kernel_name}")
        repo_spec = LayerRepository(
            repo_id=repo_id,
            layer_name=layer_name or kernel_name,
            version=version,
        )

    # 注册到 Twinkle 内部注册表
    register_layer(kernel_name, repo_spec, device)

    # 同步到 HF kernels 的全局映射
    from kernels import register_kernel_mapping as hf_register_kernel_mapping
    hf_mapping = {kernel_name: {device: repo_spec}}
    hf_register_kernel_mapping(hf_mapping, inherit_mapping=True)

    logger.info(f"Registered layer kernel: {kernel_name} for device: {device}")


def apply_layer_kernel(model, mode: ModeType = "inference", device: Optional[DeviceType] = None) -> Any:
    """应用层级别 kernel 到模型"""
    if not is_kernels_enabled():
        logger.debug("Kernels not enabled, returning original model")
        return model

    if device is None:
        device = get_device_type() or "cuda"

    kernel_mode = to_kernels_mode(mode)

    try:
        from kernels import kernelize
        logger.debug(f"Applying kernels with mode: {mode}, device: {device}")
        return kernelize(model, mode=kernel_mode, device=device)
    except Exception as e:
        logger.warning(f"Failed to apply kernels: {e}. Returning original model.")
        return model


def register_layer_batch(mapping: dict, default_device: DeviceType = "cuda") -> None:
    """批量注册层级别 kernel"""
    for kernel_name, spec in mapping.items():
        device = spec.pop("device", default_device)
        register_layer_kernel(kernel_name=kernel_name, device=device, **spec)
