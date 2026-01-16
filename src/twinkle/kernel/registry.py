# Copyright (c) ModelScope Contributors. All rights reserved.
"""Kernel module registry."""
from typing import Dict, Any, Optional, Type
from logging import getLogger

from .base import DeviceType, is_kernels_available

logger = getLogger(__name__)


# 层级别注册表: kernel_name -> device -> repo_spec
layer_registry: Dict[str, Dict[DeviceType, Any]] = {}
# 外部层映射: class -> kernel_name
external_layer_map: Dict[Type, str] = {}


class LayerRegistry:
    """层级别 kernel 注册表管理"""

    def __init__(self):
        self._registry: Dict[str, Dict[DeviceType, Any]] = {}

    def register(self, kernel_name: str, repo_spec: Any, device: DeviceType = "cuda") -> None:
        if kernel_name not in self._registry:
            self._registry[kernel_name] = {}
        self._registry[kernel_name][device] = repo_spec
        logger.debug(f"Registered layer kernel: {kernel_name} for device: {device}")

    def get(self, kernel_name: str, device: Optional[DeviceType] = None) -> Optional[Any]:
        if kernel_name not in self._registry:
            return None
        devices = self._registry[kernel_name]
        if device is None:
            return next(iter(devices.values()), None)
        return devices.get(device)

    def has(self, kernel_name: str, device: Optional[DeviceType] = None) -> bool:
        if kernel_name not in self._registry:
            return False
        if device is None:
            return True
        return device in self._registry[kernel_name]

    def remove(self, kernel_name: str, device: Optional[DeviceType] = None) -> None:
        if kernel_name not in self._registry:
            return
        if device is None:
            del self._registry[kernel_name]
        else:
            self._registry[kernel_name].pop(device, None)
            if not self._registry[kernel_name]:
                del self._registry[kernel_name]

    def list_kernel_names(self) -> list[str]:
        return list(self._registry.keys())

    def clear(self) -> None:
        self._registry.clear()


_global_layer_registry = LayerRegistry()


class ExternalLayerRegistry:
    """外部层映射管理: class -> kernel_name"""

    def __init__(self):
        self._map: Dict[Type, str] = {}

    def register(self, layer_class: Type, kernel_name: str) -> None:
        self._map[layer_class] = kernel_name
        logger.debug(f"Registered external layer: {layer_class.__name__} -> {kernel_name}")

    def get(self, layer_class: Type) -> Optional[str]:
        return self._map.get(layer_class)

    def has(self, layer_class: Type) -> bool:
        return layer_class in self._map

    def remove(self, layer_class: Type) -> None:
        self._map.pop(layer_class, None)

    def clear(self) -> None:
        self._map.clear()


_global_external_layer_registry = ExternalLayerRegistry()


def register_layer(kernel_name: str, repo_spec: Any, device: DeviceType = "cuda") -> None:
    _global_layer_registry.register(kernel_name, repo_spec, device)


def get_layer_spec(kernel_name: str, device: Optional[DeviceType] = None) -> Optional[Any]:
    return _global_layer_registry.get(kernel_name, device)


def register_external_layer(layer_class: Type, kernel_name: str) -> None:
    """注册外部层映射并调用 replace_kernel_forward_from_hub 添加 kernel_layer_name 属性"""
    _global_external_layer_registry.register(layer_class, kernel_name)

    if is_kernels_available():
        from kernels import replace_kernel_forward_from_hub
        replace_kernel_forward_from_hub(layer_class, kernel_name)
        logger.info(f"Registered {layer_class.__name__} -> kernel: {kernel_name}")
    else:
        logger.warning(
            f"HF kernels not available. {layer_class.__name__} mapping registered "
            f"but kernel replacement will not work without kernels package."
        )


def get_external_kernel_name(layer_class: Type) -> Optional[str]:
    return _global_external_layer_registry.get(layer_class)


def get_global_layer_registry() -> LayerRegistry:
    return _global_layer_registry


def get_global_external_layer_registry() -> ExternalLayerRegistry:
    return _global_external_layer_registry
