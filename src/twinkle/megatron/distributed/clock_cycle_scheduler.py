# Copyright (c) twinkle authors. All rights reserved.
"""
Clock Cycle Scheduler for multi-tenant training.

This module implements a time-sharing scheduler that batches requests
from multiple tenants into fixed clock cycles.

## Key Concepts

- Clock Cycle: Fixed time interval where all pending requests are processed
- Request Queue: Collects requests between cycles  
- Batched Grad Sync: One communication round for all tenants (efficient)
- Gradient Isolation: Each tenant has separate LoRA params, no gradient overwrite
"""

import logging
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple
from concurrent.futures import Future

import torch
import torch.nn as nn
import torch.distributed as dist

logger = logging.getLogger(__name__)


# ============ Request Types ============

class RequestType(Enum):
    """Type of training request."""
    FORWARD_BACKWARD = "forward_backward"
    OPTIM_STEP = "optim_step"
    ZERO_GRAD = "zero_grad"


@dataclass
class TrainingRequest:
    """A training request from a tenant."""
    tenant_id: str
    request_type: RequestType
    inputs: Any = None
    labels: Any = None
    kwargs: Dict[str, Any] = field(default_factory=dict)
    future: Optional[Future] = None
    submitted_at: float = field(default_factory=time.time)


# ============ Cycle Statistics ============

@dataclass
class CycleStats:
    """Statistics for a clock cycle."""
    cycle_id: int
    start_time: float
    end_time: float
    num_tenants: int
    num_requests: int
    total_samples: int
    forward_time: float
    backward_time: float
    grad_sync_time: float
    optim_step_time: float
    
    @property
    def duration(self) -> float:
        return self.end_time - self.start_time
    
    @property
    def gpu_active_time(self) -> float:
        return self.forward_time + self.backward_time + self.optim_step_time
    
    @property
    def gpu_utilization(self) -> float:
        if self.duration > 0:
            return self.gpu_active_time / self.duration
        return 0.0
    
    @property
    def samples_per_second(self) -> float:
        if self.duration > 0:
            return self.total_samples / self.duration
        return 0.0


# ============ Model Interface Requirements ============

class ModelInterfaceError(Exception):
    """Raised when model doesn't implement required interface."""
    pass


def validate_model_interface(model: nn.Module) -> None:
    """
    Validate that model implements the required interface.
    
    Required methods:
    - scope(tenant_id) -> context manager
    - zero_grad(tenant_id) -> None
    - step(tenant_id) -> None
    - __call__(inputs) -> output (forward)
    
    Optional methods:
    - clip_grad_norm(tenant_id, max_norm) -> None
    - finish_grad_sync(tenant_id) -> None
    - finish_grad_sync_batched(tenant_ids) -> None
    """
    required = ['scope', 'zero_grad', 'step']
    missing = [m for m in required if not hasattr(model, m)]
    
    if missing:
        raise ModelInterfaceError(
            f"Model must implement: {required}. Missing: {missing}"
        )


# ============ Gradient Synchronization ============

class GradientSynchronizer:
    """
    Handles gradient synchronization for multiple tenants.
    
    For distributed training, this batches gradient communication
    to reduce the number of NCCL calls.
    """
    
    def __init__(self, model: nn.Module):
        self.model = model
    
    def sync_individual(self, tenant_id: str) -> float:
        """Synchronize gradients for a single tenant."""
        if not dist.is_initialized():
            return 0.0
        
        t0 = time.time()
        
        if hasattr(self.model, 'finish_grad_sync'):
            self.model.finish_grad_sync(tenant_id)
        
        return time.time() - t0
    
    def sync_batched(self, tenant_ids: List[str]) -> float:
        """
        Synchronize gradients for multiple tenants.
        
        Uses batched sync if model supports it, otherwise falls back
        to individual sync.
        """
        if not tenant_ids:
            return 0.0
        
        t0 = time.time()
        
        if hasattr(self.model, 'finish_grad_sync_batched'):
            # Optimized: one call for all tenants
            self.model.finish_grad_sync_batched(tenant_ids)
        elif dist.is_initialized():
            # Fallback: sync each tenant individually
            for tenant_id in tenant_ids:
                self.sync_individual(tenant_id)
        
        return time.time() - t0


# ============ Clock Cycle Scheduler ============

class ClockCycleScheduler:
    """
    Clock cycle scheduler for multi-tenant training.
    
    Collects requests from multiple tenants and executes them in batched
    clock cycles. While computation is per-tenant serial (due to LoRA
    architecture), communication is batched for efficiency.
    
    ## Benefits
    
    1. **Unified Scheduling**: All tenants processed in fixed cycles
    2. **Batched Communication**: One grad sync round for all tenants  
    3. **Fair Scheduling**: All pending requests processed together
    4. **Predictable Latency**: Fixed cycle interval
    
    ## Usage
    
    ```python
    scheduler = ClockCycleScheduler(model, cycle_interval_ms=100)
    scheduler.start()
    
    # From multiple clients
    future1 = scheduler.submit_forward_backward('tenant_a', inputs_a)
    future2 = scheduler.submit_forward_backward('tenant_b', inputs_b)
    
    result1 = future1.result()
    result2 = future2.result()
    
    scheduler.stop()
    ```
    """
    
    def __init__(
        self,
        model: nn.Module,
        cycle_interval_ms: float = 100.0,
        loss_fn: Optional[Callable] = None,
    ):
        """
        Initialize the scheduler.
        
        Args:
            model: The multi-tenant model. Must implement:
                   - scope(tenant_id) -> context manager
                   - zero_grad(tenant_id) -> None
                   - step(tenant_id) -> None
                   - __call__(inputs) -> output
                   
            cycle_interval_ms: Clock cycle interval in milliseconds.
            loss_fn: Loss function (output, labels) -> loss. 
                     Default: output.mean()
        
        Raises:
            ModelInterfaceError: If model doesn't implement required methods.
        """
        # Validate model interface
        validate_model_interface(model)
        
        self.model = model
        self.cycle_interval = cycle_interval_ms / 1000.0
        self.loss_fn = loss_fn or self._default_loss_fn
        
        # Gradient synchronizer
        self._grad_sync = GradientSynchronizer(model)
        
        # Request queue (thread-safe)
        self._queue_lock = threading.Lock()
        self._request_queue: Dict[str, List[TrainingRequest]] = defaultdict(list)
        
        # Cycle management
        self._running = False
        self._cycle_thread: Optional[threading.Thread] = None
        self._current_cycle_id = 0
        
        # Statistics
        self._stats: List[CycleStats] = []
        self._stats_lock = threading.Lock()
    
    def _default_loss_fn(self, output: torch.Tensor, labels: Any) -> torch.Tensor:
        """Default loss function (mean of output)."""
        if isinstance(output, torch.Tensor):
            return output.mean()
        raise ValueError(f"Cannot compute loss on {type(output)}, provide loss_fn")
    
    def start(self):
        """Start the clock cycle loop."""
        if self._running:
            return
        
        self._running = True
        self._cycle_thread = threading.Thread(
            target=self._cycle_loop,
            name="ClockCycleLoop",
            daemon=True,
        )
        self._cycle_thread.start()
        logger.info(f"Clock cycle scheduler started (interval={self.cycle_interval*1000:.0f}ms)")
    
    def stop(self):
        """Stop the clock cycle loop."""
        self._running = False
        if self._cycle_thread:
            self._cycle_thread.join(timeout=5.0)
            self._cycle_thread = None
        logger.info("Clock cycle scheduler stopped")
    
    def submit_forward_backward(
        self,
        tenant_id: str,
        inputs: Any,
        labels: Any = None,
        **kwargs,
    ) -> Future:
        """
        Submit a forward-backward request.
        
        Returns immediately with a Future containing the result.
        """
        future = Future()
        request = TrainingRequest(
            tenant_id=tenant_id,
            request_type=RequestType.FORWARD_BACKWARD,
            inputs=inputs,
            labels=labels,
            kwargs=kwargs,
            future=future,
        )
        
        with self._queue_lock:
            self._request_queue[tenant_id].append(request)
        
        return future
    
    def submit_optim_step(self, tenant_id: str, **kwargs) -> Future:
        """Submit an optimizer step request."""
        future = Future()
        request = TrainingRequest(
            tenant_id=tenant_id,
            request_type=RequestType.OPTIM_STEP,
            kwargs=kwargs,
            future=future,
        )
        
        with self._queue_lock:
            self._request_queue[tenant_id].append(request)
        
        return future
    
    def submit_zero_grad(self, tenant_id: str) -> Future:
        """Submit a zero_grad request."""
        future = Future()
        request = TrainingRequest(
            tenant_id=tenant_id,
            request_type=RequestType.ZERO_GRAD,
            future=future,
        )
        
        with self._queue_lock:
            self._request_queue[tenant_id].append(request)
        
        return future
    
    def _cycle_loop(self):
        """Main clock cycle loop."""
        while self._running:
            cycle_start = time.time()
            
            # Collect pending requests (deep copy for thread safety)
            with self._queue_lock:
                pending = {
                    tenant_id: list(reqs)
                    for tenant_id, reqs in self._request_queue.items()
                    if reqs
                }
                self._request_queue.clear()
            
            # Execute cycle if there are requests
            if pending:
                self._execute_cycle(pending)
            
            # Wait for next cycle
            elapsed = time.time() - cycle_start
            sleep_time = max(0, self.cycle_interval - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)
    
    def _execute_cycle(self, requests: Dict[str, List[TrainingRequest]]):
        """
        Execute one clock cycle.
        
        Phases:
        1. Forward-backward for each tenant (serial due to LoRA architecture)
        2. Batched gradient synchronization (efficient)
        3. Optimizer step for each tenant
        
        Note: Forward-backward is serial per tenant because each tenant's
        LoRA weights are embedded in every layer. There's no way to "merge"
        computation across tenants for Transformer models.
        
        However, gradient sync is batched, reducing communication overhead.
        """
        cycle_start = time.time()
        self._current_cycle_id += 1
        cycle_id = self._current_cycle_id
        
        # Group requests by type
        fwd_bwd_reqs: Dict[str, TrainingRequest] = {}
        optim_step_reqs: Dict[str, TrainingRequest] = {}
        zero_grad_reqs: Dict[str, TrainingRequest] = {}
        
        for tenant_id, reqs in requests.items():
            for req in reqs:
                if req.request_type == RequestType.FORWARD_BACKWARD:
                    fwd_bwd_reqs[tenant_id] = req
                elif req.request_type == RequestType.OPTIM_STEP:
                    optim_step_reqs[tenant_id] = req
                elif req.request_type == RequestType.ZERO_GRAD:
                    zero_grad_reqs[tenant_id] = req
        
        num_tenants = len(set(fwd_bwd_reqs.keys()) | set(optim_step_reqs.keys()))
        num_requests = sum(len(reqs) for reqs in requests.values())
        
        logger.debug(f"Cycle {cycle_id}: {num_tenants} tenants, {num_requests} requests")
        
        # Tracking
        successful_tenants = []
        failed_tenants = []
        total_samples = 0
        forward_time = 0.0
        backward_time = 0.0
        grad_sync_time = 0.0
        optim_step_time = 0.0
        
        try:
            # ============ PHASE 1: Forward-Backward (per tenant) ============
            # Each tenant's forward-backward is independent because:
            # 1. Each tenant has separate LoRA parameters
            # 2. Gradients accumulate to each tenant's own LoRA params
            # 3. No gradient overwrite between tenants
            
            for tenant_id, req in fwd_bwd_reqs.items():
                try:
                    inputs = req.inputs
                    labels = req.labels
                    
                    # Get batch size for stats
                    if isinstance(inputs, torch.Tensor):
                        batch_size = inputs.size(0)
                    elif isinstance(inputs, dict) and 'input_ids' in inputs:
                        batch_size = inputs['input_ids'].size(0)
                    else:
                        batch_size = 1
                    
                    total_samples += batch_size
                    
                    with self.model.scope(tenant_id):
                        # Forward
                        t0 = time.time()
                        output = self.model(inputs)
                        forward_time += time.time() - t0
                        
                        # Loss
                        loss = self.loss_fn(output, labels)
                        
                        # Backward
                        # Note: Each tenant's LoRA params are separate,
                        # so loss.backward() accumulates gradients to
                        # this tenant's params only - no overwrite
                        t0 = time.time()
                        loss.backward()
                        backward_time += time.time() - t0
                    
                    # Record result
                    result = {
                        'loss': loss.item() if hasattr(loss, 'item') else float(loss),
                        'cycle_id': cycle_id,
                        'batch_size': batch_size,
                    }
                    req.future.set_result(result)
                    successful_tenants.append(tenant_id)
                    
                except Exception as e:
                    logger.error(f"Tenant {tenant_id} forward-backward failed: {e}")
                    failed_tenants.append(tenant_id)
                    req.future.set_exception(e)
                    
                    # Clean up failed tenant's gradient state
                    try:
                        self.model.zero_grad(tenant_id)
                    except Exception:
                        pass
            
            # ============ PHASE 2: Batched Gradient Sync ============
            # This is where we get efficiency: one sync round for all tenants
            if successful_tenants:
                grad_sync_time = self._grad_sync.sync_batched(successful_tenants)
            
            # ============ PHASE 3: Optimizer Step ============
            for tenant_id, req in optim_step_reqs.items():
                try:
                    t0 = time.time()
                    
                    # Clip gradients (optional)
                    if hasattr(self.model, 'clip_grad_norm'):
                        self.model.clip_grad_norm(tenant_id=tenant_id)
                    
                    # Optimizer step
                    self.model.step(tenant_id)
                    
                    # Zero grad after step
                    self.model.zero_grad(tenant_id)
                    
                    optim_step_time += time.time() - t0
                    req.future.set_result({'cycle_id': cycle_id})
                    
                except Exception as e:
                    logger.error(f"Tenant {tenant_id} optimizer step failed: {e}")
                    req.future.set_exception(e)
            
            # ============ PHASE 4: Standalone zero_grad ============
            for tenant_id, req in zero_grad_reqs.items():
                if tenant_id not in optim_step_reqs:
                    try:
                        self.model.zero_grad(tenant_id)
                        req.future.set_result(None)
                    except Exception as e:
                        req.future.set_exception(e)
        
        except Exception as e:
            logger.exception(f"Cycle {cycle_id} failed: {e}")
            for reqs in requests.values():
                for req in reqs:
                    if not req.future.done():
                        req.future.set_exception(e)
        
        # Record stats
        cycle_end = time.time()
        stats = CycleStats(
            cycle_id=cycle_id,
            start_time=cycle_start,
            end_time=cycle_end,
            num_tenants=num_tenants,
            num_requests=num_requests,
            total_samples=total_samples,
            forward_time=forward_time,
            backward_time=backward_time,
            grad_sync_time=grad_sync_time,
            optim_step_time=optim_step_time,
        )
        
        with self._stats_lock:
            self._stats.append(stats)
        
        logger.debug(
            f"Cycle {cycle_id} completed: duration={stats.duration*1000:.1f}ms, "
            f"tenants={num_tenants}, samples={total_samples}"
        )
    
    def get_stats(self) -> List[CycleStats]:
        """Get all cycle statistics."""
        with self._stats_lock:
            return list(self._stats)
    
    def get_summary_stats(self) -> Dict[str, float]:
        """Get summary statistics."""
        with self._stats_lock:
            if not self._stats:
                return {}
            
            total_cycles = len(self._stats)
            total_duration = sum(s.duration for s in self._stats)
            total_samples = sum(s.total_samples for s in self._stats)
            total_forward = sum(s.forward_time for s in self._stats)
            total_backward = sum(s.backward_time for s in self._stats)
            total_sync = sum(s.grad_sync_time for s in self._stats)
            total_optim = sum(s.optim_step_time for s in self._stats)
            total_gpu_time = total_forward + total_backward + total_optim
            
            return {
                'total_cycles': total_cycles,
                'total_duration': total_duration,
                'total_samples': total_samples,
                'total_gpu_time': total_gpu_time,
                'total_comm_time': total_sync,
                'avg_cycle_duration': total_duration / total_cycles,
                'gpu_utilization': total_gpu_time / total_duration if total_duration > 0 else 0,
                'throughput_samples_per_sec': total_samples / total_duration if total_duration > 0 else 0,
            }


# ============ Training Client ============

class ClockCycleTrainingClient:
    """
    Client for clock cycle scheduler.
    
    Provides a simple API for submitting training requests.
    """
    
    def __init__(self, scheduler: ClockCycleScheduler, tenant_id: str):
        self.scheduler = scheduler
        self.tenant_id = tenant_id
    
    def forward_backward(self, inputs: Any, labels: Any = None) -> Future:
        """Submit forward-backward (returns Future)."""
        return self.scheduler.submit_forward_backward(self.tenant_id, inputs, labels)
    
    def optim_step(self) -> Future:
        """Submit optimizer step (returns Future)."""
        return self.scheduler.submit_optim_step(self.tenant_id)
    
    def zero_grad(self) -> Future:
        """Submit zero_grad (returns Future)."""
        return self.scheduler.submit_zero_grad(self.tenant_id)
    
    def train_step(self, inputs: Any, labels: Any = None) -> Dict[str, Any]:
        """
        Execute a complete training step (blocking).
        
        Submits forward_backward and optim_step in the same cycle.
        """
        fwd_future = self.forward_backward(inputs, labels)
        opt_future = self.optim_step()
        
        result = fwd_future.result()
        opt_future.result()
        
        return result
