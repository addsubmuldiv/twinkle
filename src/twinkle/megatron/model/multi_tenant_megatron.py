# Copyright (c) twinkle authors. All rights reserved.
"""
Multi-Tenant Megatron Model for LoRA training.

This module provides multi-tenant LoRA training support for Megatron models,
similar to MultiLoraTransformersModel but optimized for Megatron's architecture.

Key features:
1. Uses MultiAdapter's ContextVar mechanism for tenant isolation
2. Integrates with Megatron's parallel state and DDP
3. Supports per-tenant optimizers, schedulers, and gradient accumulation
4. Compatible with Swift Megatron's LoraParallelLinear
"""

import contextvars
import logging
import re
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Type, Union

import torch
import torch.distributed as dist
import torch.nn as nn

logger = logging.getLogger(__name__)

try:
    from megatron.core import parallel_state as mpu
    from megatron.core.distributed import DistributedDataParallel as MegatronDDP
    from megatron.core.distributed.distributed_data_parallel_config import DistributedDataParallelConfig
    from megatron.core.transformer.transformer_config import TransformerConfig
    MEGATRON_AVAILABLE = True
except ImportError:
    MEGATRON_AVAILABLE = False

try:
    from peft import LoraConfig, PeftModel
    from peft.tuners.lora import LoraLayer, LoraModel
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False


class MegatronMultiAdapter:
    """
    Megatron-compatible MultiAdapter using ContextVar for tenant isolation.
    
    This patches LoraLayer/LoraModel to use ContextVar-based adapter selection,
    enabling thread/coroutine-safe multi-tenant training.
    
    Key difference from twinkle's MultiAdapter:
    - Also patches Swift Megatron's LoraParallelLinear if present
    """
    
    _adapter_var: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
        'megatron_adapter_name', default=None
    )
    _patched: bool = False
    
    def __call__(self, module: nn.Module) -> nn.Module:
        """
        Patch LoRA layers to use ContextVar-based adapter selection.
        
        Args:
            module: Model containing LoRA layers.
            
        Returns:
            Patched model (same instance, modified in-place).
        """
        if MegatronMultiAdapter._patched:
            return module
        
        self._patch_peft_lora()
        self._patch_megatron_lora()
        
        module.set_current_adapter_name = MegatronMultiAdapter.set_current_adapter_name
        MegatronMultiAdapter._patched = True
        
        return module
    
    def _patch_peft_lora(self):
        """Patch PEFT's LoraLayer and LoraModel."""
        if not PEFT_AVAILABLE:
            return
        
        def get_active_adapter(*args, **kwargs):
            return MegatronMultiAdapter._adapter_var.get()
        
        def get_active_adapters(*args, **kwargs):
            adapter_name = MegatronMultiAdapter._adapter_var.get()
            return [adapter_name] if adapter_name else []
        
        def set_active_adapters(_, value):
            pass  # Controlled via ContextVar
        
        def set_adapter(self, adapter_names):
            pass  # Controlled via ContextVar
        
        def mark_only_adapters_trainable(self, model) -> None:
            for n, p in model.named_parameters():
                p.requires_grad = "lora_" in n
        
        # Patch LoraLayer
        LoraLayer.active_adapter = property(get_active_adapter, set_active_adapters)
        LoraLayer.active_adapters = property(get_active_adapters, set_active_adapters)
        LoraLayer.set_adapter = set_adapter
        
        # Patch LoraModel
        LoraModel.active_adapter = property(get_active_adapter, set_active_adapters)
        LoraModel.active_adapters = property(get_active_adapters, set_active_adapters)
        LoraModel.set_adapter = set_adapter
        LoraModel._mark_only_adapters_as_trainable = mark_only_adapters_trainable
        
        logger.info("Patched PEFT LoraLayer/LoraModel for multi-tenant support")
    
    def _patch_megatron_lora(self):
        """Patch Swift Megatron's LoraParallelLinear if available."""
        try:
            from swift.megatron.tuners.lora import LoraParallelLinear
            
            def get_active_adapter(self):
                return MegatronMultiAdapter._adapter_var.get()
            
            def get_active_adapters(self):
                adapter_name = MegatronMultiAdapter._adapter_var.get()
                return [adapter_name] if adapter_name else []
            
            # Patch as properties
            if not hasattr(LoraParallelLinear, '_multi_tenant_patched'):
                LoraParallelLinear.active_adapter = property(get_active_adapter)
                LoraParallelLinear.active_adapters = property(get_active_adapters)
                LoraParallelLinear._multi_tenant_patched = True
                logger.info("Patched LoraParallelLinear for multi-tenant support")
        except ImportError:
            logger.debug("Swift Megatron LoraParallelLinear not available")
    
    @staticmethod
    def set_current_adapter_name(adapter_name: Optional[str]):
        """Set the current adapter for this context."""
        MegatronMultiAdapter._adapter_var.set(adapter_name)
    
    @staticmethod
    def get_current_adapter_name() -> Optional[str]:
        """Get the current adapter name."""
        return MegatronMultiAdapter._adapter_var.get()


@dataclass
class TenantState:
    """State for a single tenant."""
    adapter_name: str
    process_group: Optional[dist.ProcessGroup] = None
    optimizer: Optional[torch.optim.Optimizer] = None
    scheduler: Optional[Any] = None
    grad_scaler: Optional[torch.cuda.amp.GradScaler] = None
    lora_config: Optional['LoraConfig'] = None
    
    # Tracking
    trainable_params: List[nn.Parameter] = field(default_factory=list)
    param_names: Dict[nn.Parameter, str] = field(default_factory=dict)


class MultiTenantMegatronModel:
    """
    Multi-Tenant Megatron Model wrapper for LoRA training.
    
    This class provides:
    1. Multi-tenant adapter management using ContextVar
    2. Per-tenant optimizer and scheduler
    3. Gradient synchronization with tenant-specific process groups
    4. Integration with Megatron's DDP
    
    Design:
    - Uses a single Megatron DDP wrapper for all tenants
    - Each tenant has isolated LoRA adapters
    - ContextVar ensures thread-safe adapter switching
    
    Example:
        >>> model = create_megatron_model(...)
        >>> multi_tenant = MultiTenantMegatronModel(model, config, ddp_config)
        >>> 
        >>> # Add tenants
        >>> multi_tenant.add_tenant('user_a', lora_config_a)
        >>> multi_tenant.add_tenant('user_b', lora_config_b)
        >>> 
        >>> # Training
        >>> with multi_tenant.tenant_context('user_a'):
        ...     output = multi_tenant(input)
        ...     loss.backward()
        ...     multi_tenant.step()
    """
    
    LORA_PARAM_PATTERN = re.compile(r'\.lora_\w+\.[^.]+\.')
    
    def __init__(
        self,
        model: nn.Module,
        config: 'TransformerConfig',
        ddp_config: Optional['DistributedDataParallelConfig'] = None,
        default_dp_group: Optional[dist.ProcessGroup] = None,
    ):
        """
        Initialize multi-tenant model.
        
        Args:
            model: Base Megatron model (can be already wrapped with PEFT).
            config: Transformer configuration.
            ddp_config: DDP configuration. If None, creates default.
            default_dp_group: Default data parallel group for tenants.
        """
        if not MEGATRON_AVAILABLE:
            raise ImportError("Megatron-Core is required")
        
        self.config = config
        self.ddp_config = ddp_config or DistributedDataParallelConfig(
            overlap_grad_reduce=True,
            use_distributed_optimizer=False,
        )
        
        # Setup multi-adapter
        self._multi_adapter = MegatronMultiAdapter()
        self.model = self._multi_adapter(model)
        
        # Tenant management
        self._tenants: Dict[str, TenantState] = {}
        self._default_dp_group = default_dp_group or mpu.get_data_parallel_group(
            with_context_parallel=True
        )
        
        # DDP wrapper (created lazily after first tenant is added)
        self._ddp: Optional[MegatronDDP] = None
        
        # Add a dummy adapter to ensure PEFT model structure is ready
        self._ensure_peft_model()
    
    def _ensure_peft_model(self):
        """Ensure the model is a PEFT model."""
        if not PEFT_AVAILABLE:
            logger.warning("PEFT not available, skipping PEFT model check")
            return
        
        if not isinstance(self.model, PeftModel):
            # Create minimal LoRA config for structure
            dummy_config = LoraConfig(
                r=1,
                target_modules='all-linear',
                init_lora_weights=False,
            )
            # Note: For Megatron models, you typically use Swift's prepare_model
            logger.warning(
                "Model is not a PeftModel. For Megatron LoRA, "
                "use Swift.prepare_model() before wrapping."
            )
    
    def _wrap_with_ddp(self):
        """Wrap model with Megatron DDP (lazy initialization)."""
        if self._ddp is not None:
            return
        
        self._ddp = MegatronDDP(
            config=self.config,
            ddp_config=self.ddp_config,
            module=self.model,
        )
        logger.info(
            f"Created Megatron DDP with {len(self._ddp.params_with_grad)} params, "
            f"{len(self._ddp.bucket_groups)} bucket groups"
        )
    
    def add_tenant(
        self,
        tenant_id: str,
        lora_config: Optional['LoraConfig'] = None,
        process_group: Optional[dist.ProcessGroup] = None,
        optimizer_cls: Type[torch.optim.Optimizer] = torch.optim.AdamW,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        scheduler_cls: Optional[Type] = None,
        scheduler_kwargs: Optional[Dict[str, Any]] = None,
    ):
        """
        Add a tenant with their LoRA configuration.
        
        Args:
            tenant_id: Unique tenant identifier.
            lora_config: LoRA configuration. If None, assumes adapter already exists.
            process_group: Custom process group for this tenant's gradient sync.
            optimizer_cls: Optimizer class.
            optimizer_kwargs: Optimizer arguments.
            scheduler_cls: LR scheduler class.
            scheduler_kwargs: Scheduler arguments.
        """
        if tenant_id in self._tenants:
            logger.warning(f"Tenant '{tenant_id}' already exists, skipping")
            return
        
        adapter_name = tenant_id
        
        # Add adapter if config provided and using PEFT
        if lora_config is not None and PEFT_AVAILABLE and isinstance(self.model, PeftModel):
            # Safety checks
            lora_config.modules_to_save = None
            lora_config.bias = 'none'
            
            self.model.add_adapter(adapter_name, lora_config)
            logger.info(f"Added LoRA adapter '{adapter_name}'")
        
        # Set adapter as active to find its params
        MegatronMultiAdapter.set_current_adapter_name(adapter_name)
        
        # Find trainable params for this adapter
        trainable_params = []
        param_names = {}
        
        for name, param in self.model.named_parameters():
            if self.LORA_PARAM_PATTERN.search(name) and f'.{adapter_name}.' in name:
                param.requires_grad = True
                trainable_params.append(param)
                param_names[param] = name
        
        # Create tenant state
        state = TenantState(
            adapter_name=adapter_name,
            process_group=process_group or self._default_dp_group,
            lora_config=lora_config,
            trainable_params=trainable_params,
            param_names=param_names,
        )
        
        # Create optimizer
        if optimizer_kwargs is None:
            optimizer_kwargs = {'lr': 1e-4, 'weight_decay': 0.01}
        
        state.optimizer = optimizer_cls(trainable_params, **optimizer_kwargs)
        
        # Create scheduler if specified
        if scheduler_cls is not None:
            scheduler_kwargs = scheduler_kwargs or {}
            state.scheduler = scheduler_cls(state.optimizer, **scheduler_kwargs)
        
        self._tenants[tenant_id] = state
        
        logger.info(
            f"Registered tenant '{tenant_id}' with {len(trainable_params)} "
            f"trainable params ({sum(p.numel() for p in trainable_params):,} elements)"
        )
        
        # Reset adapter context
        MegatronMultiAdapter.set_current_adapter_name(None)
    
    def remove_tenant(self, tenant_id: str):
        """Remove a tenant."""
        if tenant_id not in self._tenants:
            logger.warning(f"Tenant '{tenant_id}' not found")
            return
        
        state = self._tenants.pop(tenant_id)
        
        # Remove adapter from model if using PEFT
        if PEFT_AVAILABLE and isinstance(self.model, PeftModel):
            try:
                self.model.delete_adapter(state.adapter_name)
            except Exception as e:
                logger.warning(f"Failed to delete adapter: {e}")
        
        logger.info(f"Removed tenant '{tenant_id}'")
    
    @contextmanager
    def tenant_context(self, tenant_id: str):
        """
        Context manager for tenant-specific operations.
        
        All forward/backward operations within this context will use
        the specified tenant's LoRA adapter.
        """
        if tenant_id not in self._tenants:
            raise ValueError(f"Tenant '{tenant_id}' not registered")
        
        state = self._tenants[tenant_id]
        prev_adapter = MegatronMultiAdapter.get_current_adapter_name()
        
        try:
            MegatronMultiAdapter.set_current_adapter_name(state.adapter_name)
            yield state
        finally:
            MegatronMultiAdapter.set_current_adapter_name(prev_adapter)
    
    def forward(self, *args, tenant_id: Optional[str] = None, **kwargs):
        """
        Forward pass with tenant selection.
        
        Args:
            *args: Model inputs.
            tenant_id: Tenant to use. If None, uses current context.
            **kwargs: Additional arguments.
        """
        if tenant_id is not None:
            MegatronMultiAdapter.set_current_adapter_name(tenant_id)
        
        # Ensure DDP is initialized
        if self._ddp is None:
            self._wrap_with_ddp()
        
        return self._ddp(*args, **kwargs)
    
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
    
    def backward(self, loss: torch.Tensor, tenant_id: Optional[str] = None):
        """
        Backward pass with optional tenant selection.
        
        Args:
            loss: Loss tensor.
            tenant_id: Tenant for gradient accumulation.
        """
        if tenant_id is not None:
            MegatronMultiAdapter.set_current_adapter_name(tenant_id)
        
        loss.backward()
        
        # Sync gradients for this tenant
        self._reduce_tenant_gradients(tenant_id)
    
    def _reduce_tenant_gradients(self, tenant_id: Optional[str] = None):
        """
        Reduce gradients for a specific tenant.
        
        For now, uses Megatron DDP's finish_grad_sync which syncs all params.
        A more optimized version could filter to only tenant's params.
        """
        tenant_id = tenant_id or MegatronMultiAdapter.get_current_adapter_name()
        
        if self._ddp is not None:
            self._ddp.finish_grad_sync()
    
    def step(self, tenant_id: Optional[str] = None):
        """
        Optimizer step for a tenant.
        
        Args:
            tenant_id: Tenant to update. If None, uses current context.
        """
        tenant_id = tenant_id or MegatronMultiAdapter.get_current_adapter_name()
        
        if tenant_id is None:
            raise ValueError("No tenant specified and no current tenant context")
        
        state = self._tenants.get(tenant_id)
        if state is None:
            raise ValueError(f"Tenant '{tenant_id}' not registered")
        
        if state.optimizer is not None:
            state.optimizer.step()
    
    def zero_grad(self, tenant_id: Optional[str] = None):
        """Zero gradients for a tenant."""
        tenant_id = tenant_id or MegatronMultiAdapter.get_current_adapter_name()
        
        if tenant_id is None:
            # Zero all
            if self._ddp is not None:
                self._ddp.zero_grad_buffer()
            return
        
        state = self._tenants.get(tenant_id)
        if state is not None and state.optimizer is not None:
            state.optimizer.zero_grad()
    
    def lr_step(self, tenant_id: Optional[str] = None):
        """LR scheduler step for a tenant."""
        tenant_id = tenant_id or MegatronMultiAdapter.get_current_adapter_name()
        
        if tenant_id is None:
            return
        
        state = self._tenants.get(tenant_id)
        if state is not None and state.scheduler is not None:
            state.scheduler.step()
    
    def clip_grad_norm(
        self,
        max_norm: float = 1.0,
        norm_type: float = 2.0,
        tenant_id: Optional[str] = None,
    ) -> torch.Tensor:
        """Clip gradients for a tenant."""
        tenant_id = tenant_id or MegatronMultiAdapter.get_current_adapter_name()
        
        if tenant_id is None:
            raise ValueError("No tenant specified")
        
        state = self._tenants.get(tenant_id)
        if state is None:
            raise ValueError(f"Tenant '{tenant_id}' not registered")
        
        return torch.nn.utils.clip_grad_norm_(
            state.trainable_params, max_norm, norm_type
        )
    
    def get_tenant_state(self, tenant_id: str) -> Optional[TenantState]:
        """Get state for a tenant."""
        return self._tenants.get(tenant_id)
    
    def list_tenants(self) -> List[str]:
        """List all registered tenants."""
        return list(self._tenants.keys())
    
    @property
    def ddp(self) -> Optional[MegatronDDP]:
        """Get the DDP wrapper."""
        return self._ddp
    
    @property
    def unwrapped_model(self) -> nn.Module:
        """Get the unwrapped model."""
        return self.model
