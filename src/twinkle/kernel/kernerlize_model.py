from __future__ import annotations

from typing import Dict, Iterable, Optional

from kernels.layer.kernelize import kernelize, register_kernel_mapping
from kernels.layer.mode import Mode
from kernels.layer.repos import RepositoryProtocol
from .function import apply_function_kernel, register_function_batch


def kernelize_model(
    model,
    *,
    mode: Mode,
    device: Optional[str] = None,
    layer_registry: Optional[
        Dict[str, Dict[str, RepositoryProtocol]]
    ] = None,
    function_registry: Optional[Iterable[dict]] = None,
    function_target_module: Optional[str] = None,
    strict_functions: bool = False,
):
    """
    Apply layer-level and function-level kernels in one entry point.

    Args:
        model: The model instance to kernelize (layer-level).
        mode: Kernelize mode (Mode.INFERENCE/Mode.TRAINING + optional TORCH_COMPILE).
        device: Device type for layer kernelize and function patch filtering.
        layer_registry: Mapping for layer kernels (same shape as register_kernel_mapping).
        function_registry: Iterable of dicts with keys:
            func_name, target_module, and one of func_impl/repo/repo_id, plus optional
            revision/version and device.
        function_target_module: Optional filter for function-level patching.
        strict_functions: Raise errors if function patching fails.
    """
    if layer_registry:
        register_kernel_mapping(layer_registry)
        model = kernelize(model, mode=mode, device=device)

    if function_registry:
        register_function_batch(function_registry)

        apply_function_kernel(
            target_module=function_target_module,
            device=device,
            strict=strict_functions,
        )

    return model
