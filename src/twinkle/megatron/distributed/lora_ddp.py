# Copyright (c) twinkle authors. All rights reserved.
"""
LoRA-aware Distributed Data Parallel wrapper for Megatron models.

This module provides a DDP wrapper that:
1. Only creates gradient buffers for LoRA parameters (trainable)
2. Supports communication-computation overlap
3. Supports multi-tenant LoRA training with separate process groups
4. Inherits from Megatron DDP to reuse optimized communication logic
"""

import logging
from contextlib import contextmanager
from typing import Dict, List, Optional, Set, Union

import torch
import torch.distributed as dist
import torch.nn as nn

logger = logging.getLogger(__name__)

try:
    from megatron.core import parallel_state as mpu
    from megatron.core.distributed import DistributedDataParallel as MegatronDDP
    from megatron.core.distributed.distributed_data_parallel_config import DistributedDataParallelConfig
    from megatron.core.distributed.param_and_grad_buffer import _ParamAndGradBuffer, partition_buckets
    from megatron.core.distributed.data_parallel_base import _BaseDataParallel
    from megatron.core.transformer.transformer_config import TransformerConfig
    from megatron.core.process_groups_config import ProcessGroupCollection
    MEGATRON_AVAILABLE = True
except ImportError:
    MEGATRON_AVAILABLE = False
    MegatronDDP = object
    _BaseDataParallel = object


class LoRADistributedDataParallel(_BaseDataParallel):
    """
    Distributed Data Parallel wrapper for LoRA/PEFT models.
    
    This class inherits from Megatron's _BaseDataParallel and implements
    DDP functionality specifically for LoRA parameters. Key features:
    
    1. **Selective Parameter Registration**: Only LoRA parameters (trainable)
       are registered for gradient synchronization, reducing memory overhead.
    
    2. **Communication-Computation Overlap**: When overlap_grad_reduce=True,
       gradient all-reduce operations are overlapped with backward computation.
    
    3. **Gradient Bucketing**: Parameters are grouped into buckets for efficient
       communication, reducing kernel launch overhead.
    
    4. **Multi-Tenant Support**: Each tenant can have its own process group
       for gradient synchronization.
    
    5. **Dynamic Parameter Updates**: Supports adding/removing LoRA parameters
       at runtime (requires buffer rebuild).
    
    Args:
        config: Transformer configuration.
        ddp_config: DDP configuration controlling overlap, bucketing, etc.
        module: The model containing LoRA layers.
        disable_bucketing: If True, all parameters go into a single bucket.
        lora_param_patterns: Set of patterns to identify LoRA parameters.
        tenant_id: Identifier for multi-tenant scenarios.
        tenant_process_group: Custom process group for this tenant.
    
    Example:
        >>> # Create DDP wrapper for LoRA model
        >>> ddp_config = DistributedDataParallelConfig(
        ...     overlap_grad_reduce=True,
        ...     bucket_size=10000000,
        ... )
        >>> ddp_model = LoRADistributedDataParallel(
        ...     config=transformer_config,
        ...     ddp_config=ddp_config,
        ...     module=lora_model,
        ... )
        >>> 
        >>> # Training loop
        >>> for batch in dataloader:
        ...     output = ddp_model(batch)
        ...     loss = compute_loss(output)
        ...     loss.backward()
        ...     ddp_model.finish_grad_sync()  # Wait for async grad sync
        ...     optimizer.step()
        ...     ddp_model.zero_grad_buffer()
    """
    
    # Default patterns to identify LoRA parameters
    DEFAULT_LORA_PATTERNS = {'lora_A', 'lora_B', 'lora_'}
    
    def __init__(
        self,
        config: 'TransformerConfig',
        ddp_config: 'DistributedDataParallelConfig',
        module: nn.Module,
        disable_bucketing: bool = False,
        lora_param_patterns: Optional[Set[str]] = None,
        tenant_id: str = 'default',
        tenant_process_group: Optional[dist.ProcessGroup] = None,
    ):
        if not MEGATRON_AVAILABLE:
            raise ImportError("Megatron-Core is required for LoRADistributedDataParallel")
        
        super().__init__(config=config, module=module)
        
        self.ddp_config = ddp_config
        self.tenant_id = tenant_id
        self.lora_param_patterns = lora_param_patterns or self.DEFAULT_LORA_PATTERNS
        self._disable_bucketing = disable_bucketing
        
        # Setup process groups
        self._setup_process_groups(tenant_process_group)
        
        # Configure bucket size
        if ddp_config.bucket_size is None:
            # Use smaller default for LoRA (fewer parameters)
            ddp_config.bucket_size = max(
                10000000, 500000 * self.dp_group.size()
            )
        if not ddp_config.overlap_grad_reduce:
            ddp_config.bucket_size = None
        
        self.bucket_size = ddp_config.bucket_size
        if disable_bucketing:
            self.bucket_size = None
        
        # Initialize data structures
        self.param_to_bucket_group = {}
        self.params_with_grad = []
        self.buffers: List['_ParamAndGradBuffer'] = []
        self.bucket_groups = []
        self.expert_parallel_buffers = []
        self.expert_parallel_bucket_groups = []
        self.grad_accs = []
        
        # Register LoRA parameters and hooks
        self._register_lora_params()
        self._register_backward_hooks()
        
        # Forward hooks for param gather overlap (usually not needed for LoRA)
        self.use_forward_hook = (
            ddp_config.use_distributed_optimizer and ddp_config.overlap_param_gather
        )
        self.remove_forward_pre_hook_handles = {}
        self.overlap_param_gather_with_optimizer_step = False
        
        logger.info(
            f"LoRADistributedDataParallel initialized for tenant '{tenant_id}' "
            f"with {len(self.params_with_grad)} LoRA parameters, "
            f"{len(self.bucket_groups)} bucket groups"
        )
    
    def _setup_process_groups(self, tenant_process_group: Optional[dist.ProcessGroup]):
        """
        Setup process groups for gradient communication.
        
        If tenant_process_group is provided, use it for DP communication.
        Otherwise, use the default Megatron parallel state groups.
        """
        if tenant_process_group is not None:
            # Use custom tenant process group
            self.dp_group = tenant_process_group
            self.dp_cp_group = tenant_process_group
            self.intra_dp_cp_group = tenant_process_group
            # Expert groups use defaults (MoE multi-tenant not supported yet)
            try:
                self.expt_dp_group = mpu.get_expert_data_parallel_group()
                self.intra_expt_dp_group = mpu.get_expert_data_parallel_group(
                    partial_expert_data_parallel=True
                )
            except:
                self.expt_dp_group = None
                self.intra_expt_dp_group = None
        else:
            # Use default Megatron process groups
            self.dp_group = mpu.get_data_parallel_group(
                with_context_parallel=False, partial_data_parallel=False
            )
            self.dp_cp_group = mpu.get_data_parallel_group(
                with_context_parallel=True, partial_data_parallel=False
            )
            self.intra_dp_cp_group = mpu.get_data_parallel_group(
                with_context_parallel=True, partial_data_parallel=True
            )
            try:
                self.expt_dp_group = mpu.get_expert_data_parallel_group()
                self.intra_expt_dp_group = mpu.get_expert_data_parallel_group(
                    partial_expert_data_parallel=True
                )
            except:
                self.expt_dp_group = None
                self.intra_expt_dp_group = None
        
        self.tp_group = mpu.get_tensor_model_parallel_group()
        self.pp_group = mpu.get_pipeline_model_parallel_group()
        try:
            self.ep_group = mpu.get_expert_model_parallel_group()
        except:
            self.ep_group = None
    
    def _is_lora_param(self, name: str) -> bool:
        """Check if a parameter is a LoRA parameter based on name patterns."""
        for pattern in self.lora_param_patterns:
            if pattern in name:
                return True
        return False
    
    def _register_lora_params(self):
        """
        Register LoRA parameters to gradient buffers.
        
        This method:
        1. Identifies LoRA parameters by name patterns
        2. Groups them by dtype
        3. Creates gradient buffers for efficient communication
        4. Sets up bucket groups for overlapped communication
        """
        param_to_name = {}
        lora_params = []
        
        for name, param in self.module.named_parameters():
            if not param.requires_grad:
                continue
            
            # Only process LoRA parameters
            if not self._is_lora_param(name):
                continue
            
            self.params_with_grad.append(param)
            param.grad_added_to_main_grad = False
            param_to_name[param] = name
            lora_params.append(param)
        
        if not lora_params:
            logger.warning(
                f"No LoRA parameters found for tenant '{self.tenant_id}'. "
                f"Patterns used: {self.lora_param_patterns}"
            )
            return
        
        # Calculate gradient scaling factor
        if self.config.calculate_per_token_loss:
            gradient_scaling_factor = 1.0
        else:
            if self.ddp_config.average_in_collective:
                gradient_scaling_factor = 1.0
            else:
                gradient_scaling_factor = 1.0 / self.dp_cp_group.size()
        
        # Group parameters by dtype
        param_and_grad_dtype_to_params = {}
        param_and_grad_dtype_to_indices = {}
        
        for param in lora_params:
            param_dtype = param.dtype
            grad_dtype = torch.float if self.ddp_config.grad_reduce_in_fp32 else param.dtype
            
            key = (param_dtype, grad_dtype)
            if key not in param_and_grad_dtype_to_params:
                param_and_grad_dtype_to_params[key] = []
                param_and_grad_dtype_to_indices[key] = []
            param_and_grad_dtype_to_params[key].append(param)
            param_and_grad_dtype_to_indices[key].append(len(param_and_grad_dtype_to_params[key]) - 1)
        
        # Create gradient buffers for each dtype combination
        pg_collection = ProcessGroupCollection()
        pg_collection.tp = self.tp_group
        pg_collection.dp_cp = self.dp_cp_group
        
        for (param_dtype, grad_dtype), params in param_and_grad_dtype_to_params.items():
            indices = param_and_grad_dtype_to_indices[(param_dtype, grad_dtype)]
            
            buffer = _ParamAndGradBuffer(
                self.ddp_config,
                param_dtype,
                grad_dtype,
                params,
                self.intra_dp_cp_group,
                self.bucket_size,
                param_to_name,
                gradient_scaling_factor,
                indices,
                getattr(self.ddp_config, 'nccl_ub', False),
                pg_collection,
            )
            self.buffers.append(buffer)
        
        # Create bucket groups
        self.bucket_groups = partition_buckets(
            self.buffers, 
            force_single_bucket_group=self._disable_bucketing
        )
        
        # Build param to bucket group mapping
        for bucket_group in self.bucket_groups:
            for bucket in bucket_group.buckets:
                for param in bucket.params:
                    self.param_to_bucket_group[param] = bucket_group
    
    def _register_backward_hooks(self):
        """
        Register backward hooks for LoRA parameters.
        
        These hooks:
        1. Accumulate gradients to main_grad buffer
        2. Trigger async gradient communication when a bucket is ready
        """
        for param in self.params_with_grad:
            if param not in self.param_to_bucket_group:
                continue
            
            # Get gradient accumulator and register hook
            param_tmp = param.expand_as(param)
            grad_acc = param_tmp.grad_fn.next_functions[0][0]
            grad_acc.register_hook(self._make_backward_post_hook(param))
            self.grad_accs.append(grad_acc)
    
    def _make_backward_post_hook(self, param: nn.Parameter):
        """
        Create a backward post-hook for a parameter.
        
        When the parameter's gradient is computed:
        1. Accumulate it to the main_grad buffer
        2. If overlap is enabled AND this is the last microbatch, start async communication
        
        Note: register_grad_ready() internally checks is_last_microbatch, so we don't
        need to check it here. The bucket_group will only start communication when
        all params are ready AND it's the last microbatch.
        """
        def hook(*unused):
            if param in self.param_to_bucket_group:
                # Accumulate gradient to main_grad
                if param.grad is not None and not param.grad_added_to_main_grad:
                    param.main_grad.add_(param.grad.data)
                param.grad = None
                
                # If overlap enabled, notify bucket that param is ready
                # Note: register_grad_ready internally checks is_last_microbatch
                # and only registers when processing the last microbatch
                if self.ddp_config.overlap_grad_reduce:
                    bucket_group = self.param_to_bucket_group[param]
                    # Only register if this is the last microbatch
                    # (bucket_group.is_last_microbatch controls this)
                    if bucket_group.is_last_microbatch:
                        bucket_group.register_grad_ready(param)
        
        return hook
    
    @contextmanager
    def no_sync(self):
        """
        Context manager that turns off gradient synchronization.
        
        Use this for gradient accumulation - only sync on the last microbatch.
        """
        for bucket_group in self.bucket_groups + self.expert_parallel_bucket_groups:
            bucket_group.is_last_microbatch = False
        try:
            yield
        finally:
            for bucket_group in self.bucket_groups + self.expert_parallel_bucket_groups:
                bucket_group.is_last_microbatch = True
    
    def start_grad_sync(self, *unused):
        """
        Start gradient synchronization (all-reduce or reduce-scatter).
        
        When overlap_grad_reduce=True, this dispatches async operations.
        When overlap_grad_reduce=False, this is a no-op (finish_grad_sync does sync).
        """
        for bucket_group in self.bucket_groups + self.expert_parallel_bucket_groups:
            bucket_group.start_grad_sync()
    
    def finish_grad_sync(self):
        """
        Finish gradient synchronization.
        
        When overlap_grad_reduce=True, waits for async operations to complete.
        When overlap_grad_reduce=False, performs synchronous communication.
        """
        for bucket_group in self.bucket_groups + self.expert_parallel_bucket_groups:
            bucket_group.finish_grad_sync()
    
    def scale_gradients(self, scaling_factor: float):
        """Scale all gradients in buffers by the given factor."""
        for buffer in self.buffers + self.expert_parallel_buffers:
            buffer.scale_gradients(scaling_factor)
    
    def zero_grad_buffer(self):
        """
        Zero out all gradient buffers.
        
        Must be called at the beginning of each training iteration.
        """
        for param in self.params_with_grad:
            param.grad_added_to_main_grad = False
        for buffer in self.buffers + self.expert_parallel_buffers:
            buffer.reset()
        for bucket_group in self.bucket_groups + self.expert_parallel_bucket_groups:
            bucket_group.reset()
    
    def broadcast_params(self):
        """Broadcast parameters from rank 0 to all DP ranks."""
        for param in self.params_with_grad:
            dist.broadcast(
                param.data,
                src=dist.get_global_rank(self.dp_cp_group, 0),
                group=self.dp_cp_group,
            )
    
    def add_lora_params(self, new_params: Dict[str, nn.Parameter]):
        """
        Dynamically add LoRA parameters.
        
        Note: This requires rebuilding gradient buffers, which is expensive.
        Use sparingly.
        
        Args:
            new_params: Dictionary mapping parameter names to parameters.
        """
        for name, param in new_params.items():
            if param.requires_grad:
                self.params_with_grad.append(param)
                param.grad_added_to_main_grad = False
        
        self._rebuild_buffers()
    
    def remove_lora_params(self, param_names: Set[str]):
        """
        Remove LoRA parameters.
        
        Note: This requires rebuilding gradient buffers, which is expensive.
        
        Args:
            param_names: Set of parameter names to remove.
        """
        new_params_with_grad = []
        for param in self.params_with_grad:
            # Find param name in module
            for name, p in self.module.named_parameters():
                if p is param and name not in param_names:
                    new_params_with_grad.append(param)
                    break
        
        self.params_with_grad = new_params_with_grad
        self._rebuild_buffers()
    
    def _rebuild_buffers(self):
        """Rebuild gradient buffers after parameter changes."""
        # Clear old hooks and buffers
        self.grad_accs.clear()
        self.param_to_bucket_group.clear()
        self.buffers.clear()
        self.bucket_groups.clear()
        
        # Re-register
        self._register_lora_params()
        self._register_backward_hooks()
        
        logger.info(
            f"Rebuilt buffers for tenant '{self.tenant_id}': "
            f"{len(self.params_with_grad)} params, {len(self.bucket_groups)} bucket groups"
        )
    
    def get_lora_param_count(self) -> int:
        """Get the number of registered LoRA parameters."""
        return len(self.params_with_grad)
    
    def get_lora_param_numel(self) -> int:
        """Get the total number of elements in LoRA parameters."""
        return sum(p.numel() for p in self.params_with_grad)


def wrap_model_with_lora_ddp(
    model: nn.Module,
    config: 'TransformerConfig',
    ddp_config: Optional['DistributedDataParallelConfig'] = None,
    lora_param_patterns: Optional[Set[str]] = None,
    tenant_id: str = 'default',
    tenant_process_group: Optional[dist.ProcessGroup] = None,
    overlap_grad_reduce: bool = True,
    bucket_size: Optional[int] = None,
) -> LoRADistributedDataParallel:
    """
    Convenience function to wrap a LoRA model with DDP.
    
    This is the recommended way to create a LoRADistributedDataParallel wrapper.
    
    Args:
        model: Model containing LoRA layers.
        config: Transformer configuration.
        ddp_config: DDP configuration. If None, creates default config.
        lora_param_patterns: Patterns to identify LoRA parameters.
        tenant_id: Tenant identifier for multi-tenant scenarios.
        tenant_process_group: Custom process group for this tenant.
        overlap_grad_reduce: Enable communication-computation overlap.
        bucket_size: Size of gradient buckets. None for auto.
        
    Returns:
        LoRADistributedDataParallel wrapper.
        
    Example:
        >>> ddp_model = wrap_model_with_lora_ddp(
        ...     model=lora_model,
        ...     config=transformer_config,
        ...     overlap_grad_reduce=True,
        ... )
    """
    if not MEGATRON_AVAILABLE:
        raise ImportError("Megatron-Core is required for wrap_model_with_lora_ddp")
    
    if ddp_config is None:
        ddp_config = DistributedDataParallelConfig(
            overlap_grad_reduce=overlap_grad_reduce,
            use_distributed_optimizer=False,  # LoRA params are small
            bucket_size=bucket_size,
        )
    
    if lora_param_patterns is None:
        lora_param_patterns = LoRADistributedDataParallel.DEFAULT_LORA_PATTERNS
    
    return LoRADistributedDataParallel(
        config=config,
        ddp_config=ddp_config,
        module=model,
        lora_param_patterns=lora_param_patterns,
        tenant_id=tenant_id,
        tenant_process_group=tenant_process_group,
    )
