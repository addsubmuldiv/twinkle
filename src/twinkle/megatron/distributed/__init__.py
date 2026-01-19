# Copyright (c) twinkle authors. All rights reserved.
"""
[WIP]
Distributed training utilities for multi-tenant Megatron LoRA.

Core components:
- tenant_context: ContextVar-based tenant management
- tenant_manager: Tenant lifecycle (adapters, optimizers)
- multi_tenant_ddp: Per-tenant gradient buffers and sync
"""

from .multi_tenant_ddp import MultiTenantLoRADDP, TenantDDPState
from .tenant_context import (TenantInfo, generate_tenant_id,
                             get_current_tenant, require_tenant,
                             set_current_tenant, tenant_scope)
from .tenant_manager import TenantManager, TenantState
from .clock_cycle_scheduler import (
    ClockCycleScheduler,
    ClockCycleTrainingClient,
    CycleStats,
    RequestType,
    TrainingRequest,
    ModelInterfaceError,
    validate_model_interface,
)

__all__ = [
    # Context
    'get_current_tenant',
    'set_current_tenant',
    'require_tenant',
    'tenant_scope',
    'generate_tenant_id',
    'TenantInfo',
    # Manager
    'TenantManager',
    'TenantState',
    # DDP (Twinkle mode)
    'MultiTenantLoRADDP',
    'TenantDDPState',
    # Clock Cycle Scheduler
    'ClockCycleScheduler',
    'ClockCycleTrainingClient',
    'CycleStats',
    'RequestType',
    'TrainingRequest',
    'ModelInterfaceError',
    'validate_model_interface',
]
