# Copyright (c) ModelScope Contributors. All rights reserved.
from typing import Any, Dict, Literal, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn

from twinkle import DeviceMesh, exists


_megatron_available = exists('megatron_core')
_mcore_013 = exists('megatron_core>=0.13')


class MegatronStrategy:

    def __init__(
        self,
        device_mesh: Optional[DeviceMesh] = None,
        expert_tensor_parallel_size: Optional[int] = None,
        sequence_parallel: bool = False,
        use_distributed_optimizer: bool = True,
        mixed_precision: Literal['no', 'fp16', 'bf16'] = 'bf16',
        params_dtype: Optional[str] = None,
        megatron_args: Optional[Dict[str, Any]] = None,
    ):
        self.device_mesh = device_mesh
        self.etp_size = expert_tensor_parallel_size or self.device_mesh.tp_world_size
        self.sequence_parallel = sequence_parallel
        self.use_distributed_optimizer = use_distributed_optimizer
        self.mixed_precision = mixed_precision
        self.params_dtype = params_dtype
        self.megatron_args = megatron_args or {}
        self._initialized = False
        self._parallel_state = None

    def initialize(self, **kwargs) -> None:
        if self._initialized:
            return

        dist.init_process_group(backend='nccl')

        init_kwargs = {
            'tensor_model_parallel_size': self.device_mesh.tp_world_size or 1,
            'pipeline_model_parallel_size': self.device_mesh.pp_world_size or 1,
            'context_parallel_size': self.device_mesh.cp_world_size or 1,
            'virtual_pipeline_model_parallel_size': self.device_mesh.vpp_size or 1,
            'expert_model_parallel_size': self.device_mesh.ep_size or 1,
        }

        if _mcore_013:
            init_kwargs['expert_tensor_parallel_size'] = self.etp_size

        parallel_state.initialize_model_parallel(**init_kwargs)

        self._parallel_state = parallel_state
        self._initialized = True

        # Set CUDA device (may be redundant in Ray mode, but safe)
        local_rank = dist.get_rank() % torch.cuda.device_count()
        torch.cuda.set_device(local_rank)

    def get_params_dtype(self) -> torch.dtype:
        """Get parameter dtype based on configuration.

        Returns:
            PyTorch dtype for model parameters.
        """
        if self.params_dtype is not None:
            dtype_map = {
                'fp32': torch.float32,
                'fp16': torch.float16,
                'bf16': torch.bfloat16,
            }
            return dtype_map.get(self.params_dtype, torch.bfloat16)

        if self.mixed_precision == 'bf16':
            return torch.bfloat16
        elif self.mixed_precision == 'fp16':
            return torch.float16
        return torch.float32

    def _get_transformer_config(self, model: nn.Module):
        """Get TransformerConfig from model, handling PEFT wrappers.

        Args:
            model: The model (may be wrapped with PEFT).

        Returns:
            TransformerConfig if found, None otherwise.
        """
        # Direct config attribute
        config = getattr(model, 'config', None)
        if config is not None and hasattr(config,
                                          'tensor_model_parallel_size'):
            return config

        # PEFT model: model.base_model.model.config
        if hasattr(model, 'base_model'):
            base = model.base_model
            if hasattr(base, 'model'):
                config = getattr(base.model, 'config', None)
                if config is not None and hasattr(
                        config, 'tensor_model_parallel_size'):
                    return config
            # Try base.config
            config = getattr(base, 'config', None)
            if config is not None and hasattr(config,
                                              'tensor_model_parallel_size'):
                return config

        # Wrapped model: model.model.config
        if hasattr(model, 'model'):
            config = getattr(model.model, 'config', None)
            if config is not None and hasattr(config,
                                              'tensor_model_parallel_size'):
                return config

        # Recursive search through modules
        for name, module in model.named_modules():
            config = getattr(module, 'config', None)
            if config is not None and hasattr(config,
                                              'tensor_model_parallel_size'):
                return config

        return None

    def wrap_model(
        self,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        use_distributed_optimizer: bool = True,
    ) -> Tuple[nn.Module, Optional[torch.optim.Optimizer]]:
        """Wrap model with Megatron DDP for data parallelism.

        This method behaves differently based on twinkle's execution mode:

        **Local mode (torchrun)**:
          - Uses Megatron native DDP wrapping
          - All processes are synchronized by torchrun, so collective ops work

        **Ray mode**:
          - Currently skips DDP wrapping to avoid deadlocks
          - Ray's asynchronous actor model makes collective synchronization hard
          - Each DP replica trains independently

        **Transformers/Accelerate comparison**:
          - Accelerate's `prepare()` works in Ray because it's a local operation
          - Megatron DDP's `broadcast_params()` is a collective that needs sync

        Args:
            model: The Megatron model (already has TP/PP via TransformerConfig).
            optimizer: Optional optimizer.
            use_distributed_optimizer: Whether to use distributed optimizer.

        Returns:
            Tuple of (wrapped_model, optimizer).
        """
        if not self._initialized:
            self.initialize()

        # Determine execution mode
        import os
        twinkle_mode = os.environ.get('TWINKLE_MODE', 'local')

        # Check DP world size
        dp_group = self.dp_group
        dp_world_size = 1
        if dp_group is not None:
            dp_world_size = dist.get_world_size(dp_group)

        if dp_world_size <= 1:
            # No DP needed (single GPU or TP-only)
            return model, optimizer

        if twinkle_mode == 'ray':
            # In Ray mode, skip DDP for now due to collective sync issues
            # TODO: Implement Ray-compatible DDP with barrier synchronization
            import warnings
            warnings.warn(
                'Skipping Megatron DDP in Ray mode. Each DP replica trains independently. '
                'For synchronized training, use torchrun (TWINKLE_MODE=local).'
            )
            return model, optimizer

        # Local mode (torchrun): Use Megatron native DDP
        return self._wrap_with_megatron_ddp(model, optimizer,
                                            use_distributed_optimizer)

    def _wrap_with_megatron_ddp(
        self,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer],
        use_distributed_optimizer: bool,
    ) -> Tuple[nn.Module, Optional[torch.optim.Optimizer]]:
        """
        Wrap model with Megatron native DDP (for torchrun mode).
        """
        from megatron.core.distributed import DistributedDataParallelConfig
        from megatron.core.transformer.module import Float16Module

        # Get TransformerConfig from model
        config = self._get_transformer_config(model)
        if config is None:
            import warnings
            warnings.warn(
                'Could not find TransformerConfig. Skipping DDP wrapping. '
                'Gradient sync will need to be done manually.')
            return model, optimizer

        # Ensure model is on GPU
        try:
            model_device = next(model.parameters()).device
            if model_device.type == 'cpu':
                local_rank = dist.get_rank() % torch.cuda.device_count()
                model = model.to(f'cuda:{local_rank}')
        except StopIteration:
            pass  # No parameters

        # Wrap with Float16Module for mixed precision (like Megatron's get_model)
        if (config.fp16
                or config.bf16) and not isinstance(model, Float16Module):
            # Check if the inner model (for PEFT) needs wrapping
            inner_model = model
            if hasattr(model, 'base_model') and hasattr(
                    model.base_model, 'model'):
                inner_model = model.base_model.model

            # Only wrap if not already wrapped
            if not isinstance(inner_model, Float16Module):
                # For PEFT models, we can't easily wrap the inner model
                # Just proceed without Float16Module
                if not hasattr(model, 'base_model'):
                    model = Float16Module(config, model)

        # Create DDP config
        ddp_config = DistributedDataParallelConfig(
            grad_reduce_in_fp32=True,
            overlap_grad_reduce=False,
            use_distributed_optimizer=use_distributed_optimizer,
        )

        # Wrap with MegatronDDP
        # TODO: multi-tenant ddp
        try:
            wrapped_model = MegatronDDP(
                config=config,
                ddp_config=ddp_config,
                module=model,
            )

            # Broadcast params from data parallel src rank
            # In torchrun mode, all ranks enter here simultaneously, so this works
            wrapped_model.broadcast_params()

            return wrapped_model, optimizer

        except Exception as e:
            import warnings
            warnings.warn(
                f'Failed to wrap with Megatron DDP: {e}. Using unwrapped model.'
            )
            return model, optimizer

    def get_model_config(
        self,
        hidden_size: int,
        num_attention_heads: int,
        num_layers: int,
        ffn_hidden_size: Optional[int] = None,
        num_query_groups: Optional[int] = None,
        num_experts: Optional[int] = None,
        moe_router_topk: int = 2,
        **kwargs,
    ):
        from megatron.core.transformer import TransformerConfig

        config = TransformerConfig(
            num_layers=num_layers,
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            num_query_groups=num_query_groups or num_attention_heads,
            ffn_hidden_size=ffn_hidden_size or 4 * hidden_size,
            use_cpu_initialization=True,
            params_dtype=self.get_params_dtype(),
            tensor_model_parallel_size=self.tp_size,
            pipeline_model_parallel_size=self.pp_size,
            context_parallel_size=self.cp_size,
            expert_model_parallel_size=self.ep_size,
            sequence_parallel=self.sequence_parallel,
            num_moe_experts=num_experts,
            moe_router_topk=moe_router_topk,
            **kwargs,
        )

        return config

