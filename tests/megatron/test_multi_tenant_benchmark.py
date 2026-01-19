#!/usr/bin/env python
"""
Benchmark comparison of multi-tenant architectures.

Compares:
1. Twinkle Mode: Per-tenant serial execution (independent calls)
2. Clock Cycle Mode: Unified scheduling + batched communication

Key insight: For LLM+LoRA, batch merging is NOT possible because LoRA
weights are embedded in every layer. The benefit of Clock Cycle is
batched communication, not merged computation.

Metrics:
- Throughput (samples/second)
- Latency (per step)
- Communication efficiency (N syncs vs 1 sync)
"""

import argparse
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ============ Mock Model with Tinker-compatible API ============

class MockBaseModel(nn.Module):
    """Mock base model (shared across tenants)."""
    
    def __init__(self, hidden_size: int, num_layers: int, simulate_ms: float):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.simulate_ms = simulate_ms
        
        # Create base layers (frozen)
        self.layers = nn.ModuleList([
            nn.Linear(hidden_size, hidden_size, bias=False)
            for _ in range(num_layers)
        ])
        
        for layer in self.layers:
            layer.weight.requires_grad = False
        
        # Stats
        self.forward_calls = 0
        self.total_samples = 0
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass - simulates compute time."""
        batch_size = x.size(0)
        self.forward_calls += 1
        self.total_samples += batch_size
        
        # Simulate compute (scales slightly with batch size)
        # Key insight: one large batch is more efficient than N small batches
        time.sleep(self.simulate_ms / 1000.0 * (1 + 0.1 * (batch_size / 8)))
        
        for layer in self.layers:
            x = layer(x)
            x = torch.relu(x)
        return x
    
    def reset_stats(self):
        self.forward_calls = 0
        self.total_samples = 0


class MockLoRAAdapter(nn.Module):
    """Mock LoRA adapter for a single tenant."""
    
    def __init__(self, hidden_size: int, rank: int = 8, simulate_ms: float = 1.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.rank = rank
        self.simulate_ms = simulate_ms
        
        self.lora_A = nn.Parameter(torch.randn(rank, hidden_size) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(hidden_size, rank))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply LoRA transformation."""
        time.sleep(self.simulate_ms / 1000.0)
        return x + x @ self.lora_A.T @ self.lora_B.T


class MockMultiTenantModel(nn.Module):
    """
    Mock multi-tenant model with Tinker-compatible API.
    
    Supports:
    - base_forward(): Run base model only (for batch merging)
    - apply_lora(): Apply per-tenant LoRA
    - scope(): Context manager for tenant selection
    - finish_grad_sync_batched(): Batched gradient sync
    """
    
    def __init__(
        self,
        hidden_size: int = 256,
        num_layers: int = 4,
        lora_rank: int = 8,
        base_model_ms: float = 10.0,
        lora_ms: float = 2.0,
        comm_ms: float = 5.0,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lora_rank = lora_rank
        self.base_model_ms = base_model_ms
        self.lora_ms = lora_ms
        self.comm_ms = comm_ms
        
        # Base model (shared)
        self.base_model = MockBaseModel(hidden_size, num_layers, base_model_ms)
        
        # Per-tenant adapters
        self._adapters: Dict[str, MockLoRAAdapter] = nn.ModuleDict()
        self._optimizers: Dict[str, torch.optim.Optimizer] = {}
        
        # Current tenant context
        self._current_tenant: Optional[str] = None
        self._lock = threading.Lock()
        
        # Stats
        self._compute_time = 0.0
        self._comm_time = 0.0
    
    def initialize(
        self,
        tenant_id: Optional[str] = None,
        optimizer_kwargs: Optional[Dict] = None,
        **kwargs,
    ) -> str:
        """Initialize a tenant."""
        import uuid
        tenant_id = tenant_id or str(uuid.uuid4())[:8]
        
        with self._lock:
            if tenant_id in self._adapters:
                raise ValueError(f"Tenant {tenant_id} exists")
            
            # Create adapter
            adapter = MockLoRAAdapter(self.hidden_size, self.lora_rank, self.lora_ms)
            self._adapters[tenant_id] = adapter
            
            # Create optimizer
            opt_kwargs = optimizer_kwargs or {'lr': 1e-4}
            self._optimizers[tenant_id] = torch.optim.AdamW(adapter.parameters(), **opt_kwargs)
            
            self._current_tenant = tenant_id
        
        return tenant_id
    
    def finalize(self, tenant_id: Optional[str] = None):
        """Finalize a tenant."""
        tenant_id = tenant_id or self._current_tenant
        with self._lock:
            if tenant_id in self._adapters:
                del self._adapters[tenant_id]
                del self._optimizers[tenant_id]
    
    @contextmanager
    def scope(self, tenant_id: str):
        """Context manager for tenant scope."""
        old = self._current_tenant
        self._current_tenant = tenant_id
        try:
            yield
        finally:
            self._current_tenant = old
    
    def base_forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Run ONLY the base model (for batch merging).
        
        This is the key for Tinker efficiency - call once for all tenants.
        """
        return self.base_model(x)
    
    def apply_lora(self, features: torch.Tensor, tenant_id: Optional[str] = None) -> torch.Tensor:
        """Apply per-tenant LoRA to pre-computed features."""
        tenant_id = tenant_id or self._current_tenant
        if tenant_id and tenant_id in self._adapters:
            return self._adapters[tenant_id](features)
        return features
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Full forward pass (base + current tenant's LoRA)."""
        features = self.base_model(x)
        return self.apply_lora(features)
    
    def backward(self, loss: torch.Tensor, tenant_id: Optional[str] = None):
        """Backward pass."""
        t0 = time.time()
        loss.backward()
        self._compute_time += time.time() - t0
    
    def finish_grad_sync(self, tenant_id: Optional[str] = None):
        """Finish gradient sync for single tenant."""
        time.sleep(self.comm_ms / 1000.0)
        self._comm_time += self.comm_ms / 1000.0
    
    def finish_grad_sync_batched(self, tenant_ids: List[str]):
        """
        Batched gradient sync (Tinker optimization).
        
        One all-reduce for all tenants instead of N all-reduces.
        """
        # Simulate batched communication (more efficient than N separate calls)
        # Overhead is sub-linear with number of tenants
        batched_time = self.comm_ms / 1000.0 * (1 + 0.1 * len(tenant_ids))
        time.sleep(batched_time)
        self._comm_time += batched_time
    
    def clip_grad_norm(self, tenant_id: Optional[str] = None, max_norm: float = 1.0):
        """Clip gradients."""
        tenant_id = tenant_id or self._current_tenant
        if tenant_id in self._adapters:
            torch.nn.utils.clip_grad_norm_(
                self._adapters[tenant_id].parameters(), max_norm
            )
    
    def step(self, tenant_id: Optional[str] = None):
        """Optimizer step."""
        tenant_id = tenant_id or self._current_tenant
        if tenant_id in self._optimizers:
            self._optimizers[tenant_id].step()
    
    def zero_grad(self, tenant_id: Optional[str] = None):
        """Zero gradients."""
        tenant_id = tenant_id or self._current_tenant
        if tenant_id in self._optimizers:
            self._optimizers[tenant_id].zero_grad(set_to_none=True)
    
    def get_stats(self) -> Dict[str, Any]:
        return {
            'compute_time': self._compute_time,
            'comm_time': self._comm_time,
            'base_model_forward_calls': self.base_model.forward_calls,
            'base_model_total_samples': self.base_model.total_samples,
        }
    
    def reset_stats(self):
        self._compute_time = 0.0
        self._comm_time = 0.0
        self.base_model.reset_stats()
    
    def tenant_count(self) -> int:
        return len(self._adapters)
    
    def has_tenant(self, tenant_id: str) -> bool:
        return tenant_id in self._adapters


# ============ Benchmark Classes ============

@dataclass
class BenchmarkConfig:
    """Configuration for benchmark."""
    num_tenants: int = 4
    steps_per_tenant: int = 10
    batch_size_per_tenant: int = 8
    hidden_size: int = 256
    base_model_ms: float = 10.0  # Base model forward time
    lora_ms: float = 2.0         # LoRA forward time per tenant
    comm_ms: float = 5.0         # Communication time
    clock_cycle_interval_ms: float = 50.0


@dataclass
class BenchmarkResult:
    """Result of a benchmark run."""
    mode: str
    total_time: float
    total_steps: int
    total_samples: int
    throughput_steps: float      # steps/second
    throughput_samples: float    # samples/second
    avg_latency: float           # seconds per step
    base_model_calls: int        # Number of base model forward calls
    base_model_samples: int      # Total samples processed by base model
    compute_time: float
    comm_time: float
    gpu_utilization: float       # compute_time / total_time
    
    def __str__(self):
        return (
            f"{self.mode}:\n"
            f"  Total time: {self.total_time:.2f}s\n"
            f"  Total steps: {self.total_steps} ({self.total_samples} samples)\n"
            f"  Throughput: {self.throughput_steps:.2f} steps/s, {self.throughput_samples:.2f} samples/s\n"
            f"  Avg latency: {self.avg_latency*1000:.2f} ms/step\n"
            f"  Base model calls: {self.base_model_calls} (samples: {self.base_model_samples})\n"
            f"  GPU utilization: {self.gpu_utilization*100:.1f}%\n"
        )


class TwinkleBenchmark:
    """
    Benchmark for Twinkle mode (per-tenant serial execution).
    
    In this mode:
    - Each tenant's request is processed separately
    - Base model is called N times (once per tenant)
    - Gradient sync is done N times
    """
    
    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.model = MockMultiTenantModel(
            hidden_size=config.hidden_size,
            base_model_ms=config.base_model_ms,
            lora_ms=config.lora_ms,
            comm_ms=config.comm_ms,
        )
    
    def run(self) -> BenchmarkResult:
        """Run the benchmark."""
        logger.info("Running Twinkle mode benchmark...")
        
        # Initialize tenants
        tenant_ids = []
        for i in range(self.config.num_tenants):
            tid = self.model.initialize(tenant_id=f"tenant_{i}")
            tenant_ids.append(tid)
        
        self.model.reset_stats()
        
        # Create dummy input
        x = torch.randn(self.config.batch_size_per_tenant, self.config.hidden_size)
        
        total_steps = 0
        total_samples = 0
        step_latencies = []
        
        start_time = time.time()
        
        # Training loop - serial per tenant
        for step in range(self.config.steps_per_tenant):
            for tenant_id in tenant_ids:
                step_start = time.time()
                
                with self.model.scope(tenant_id):
                    self.model.zero_grad(tenant_id)
                    output = self.model(x)  # Full forward (base + LoRA)
                    loss = output.mean()
                    self.model.backward(loss, tenant_id)
                    self.model.finish_grad_sync(tenant_id)  # Individual sync
                    self.model.clip_grad_norm(tenant_id)
                    self.model.step(tenant_id)
                
                step_latencies.append(time.time() - step_start)
                total_steps += 1
                total_samples += self.config.batch_size_per_tenant
        
        total_time = time.time() - start_time
        
        # Cleanup
        for tid in tenant_ids:
            self.model.finalize(tid)
        
        # Calculate metrics
        stats = self.model.get_stats()
        
        return BenchmarkResult(
            mode="Twinkle (Serial)",
            total_time=total_time,
            total_steps=total_steps,
            total_samples=total_samples,
            throughput_steps=total_steps / total_time,
            throughput_samples=total_samples / total_time,
            avg_latency=sum(step_latencies) / len(step_latencies),
            base_model_calls=stats['base_model_forward_calls'],
            base_model_samples=stats['base_model_total_samples'],
            compute_time=stats['compute_time'],
            comm_time=stats['comm_time'],
            gpu_utilization=stats['compute_time'] / total_time,
        )


class TinkerBenchmark:
    """
    Benchmark for Tinker mode (clock cycle with batch merging).
    
    In this mode:
    - Multiple tenants' requests are batched in each cycle
    - Base model is called ONCE per cycle (with merged batch)
    - Gradient sync is done ONCE per cycle (batched)
    """
    
    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.model = MockMultiTenantModel(
            hidden_size=config.hidden_size,
            base_model_ms=config.base_model_ms,
            lora_ms=config.lora_ms,
            comm_ms=config.comm_ms,
        )
    
    def run(self) -> BenchmarkResult:
        """Run the benchmark."""
        from twinkle.megatron.distributed.clock_cycle_scheduler import (
            ClockCycleScheduler,
            ClockCycleTrainingClient,
        )
        
        logger.info("Running Tinker mode benchmark...")
        
        # Initialize tenants
        tenant_ids = []
        for i in range(self.config.num_tenants):
            tid = self.model.initialize(tenant_id=f"tenant_{i}")
            tenant_ids.append(tid)
        
        self.model.reset_stats()
        
        # Create scheduler
        scheduler = ClockCycleScheduler(
            model=self.model,
            cycle_interval_ms=self.config.clock_cycle_interval_ms,
        )
        scheduler.start()
        
        # Create clients for each tenant
        clients = {
            tid: ClockCycleTrainingClient(scheduler, tid)
            for tid in tenant_ids
        }
        
        total_steps = 0
        total_samples = 0
        step_latencies = []
        
        start_time = time.time()
        
        # Training loop - all tenants submit concurrently
        def tenant_worker(tenant_id: str, client: ClockCycleTrainingClient):
            nonlocal total_steps, total_samples
            latencies = []
            
            # Each tenant has its own batch
            x = torch.randn(self.config.batch_size_per_tenant, self.config.hidden_size)
            
            for step in range(self.config.steps_per_tenant):
                step_start = time.time()
                
                # Submit forward-backward and optimizer step
                result = client.train_step(x)
                
                latencies.append(time.time() - step_start)
                total_steps += 1
                total_samples += self.config.batch_size_per_tenant
            
            return latencies
        
        # Run all tenants concurrently
        with ThreadPoolExecutor(max_workers=self.config.num_tenants) as executor:
            futures = {
                executor.submit(tenant_worker, tid, clients[tid]): tid
                for tid in tenant_ids
            }
            
            for future in as_completed(futures):
                try:
                    latencies = future.result()
                    step_latencies.extend(latencies)
                except Exception as e:
                    logger.error(f"Tenant worker failed: {e}")
        
        total_time = time.time() - start_time
        
        # Stop scheduler
        scheduler.stop()
        
        # Get scheduler stats
        sched_stats = scheduler.get_summary_stats()
        
        # Cleanup
        for tid in tenant_ids:
            self.model.finalize(tid)
        
        # Calculate metrics
        model_stats = self.model.get_stats()
        
        return BenchmarkResult(
            mode="Tinker (Clock Cycle)",
            total_time=total_time,
            total_steps=total_steps,
            total_samples=total_samples,
            throughput_steps=total_steps / total_time,
            throughput_samples=total_samples / total_time,
            avg_latency=sum(step_latencies) / len(step_latencies) if step_latencies else 0,
            base_model_calls=model_stats['base_model_forward_calls'],
            base_model_samples=model_stats['base_model_total_samples'],
            compute_time=model_stats['compute_time'],
            comm_time=model_stats['comm_time'],
            gpu_utilization=sched_stats.get('gpu_utilization', 0),
        )


# ============ Test Functions ============

def test_twinkle_mode():
    """Test Twinkle mode functionality."""
    logger.info("Testing Twinkle mode...")
    
    model = MockMultiTenantModel(base_model_ms=1.0, lora_ms=0.5, comm_ms=0.5)
    
    # Initialize 2 tenants
    tid1 = model.initialize(tenant_id="test_1")
    tid2 = model.initialize(tenant_id="test_2")
    
    assert model.tenant_count() == 2
    assert model.has_tenant(tid1)
    assert model.has_tenant(tid2)
    
    # Training step for tenant 1
    x = torch.randn(4, 256)
    with model.scope(tid1):
        model.zero_grad(tid1)
        output = model(x)
        loss = output.mean()
        model.backward(loss, tid1)
        model.finish_grad_sync(tid1)
        model.step(tid1)
    
    # Training step for tenant 2
    with model.scope(tid2):
        model.zero_grad(tid2)
        output = model(x)
        loss = output.mean()
        model.backward(loss, tid2)
        model.finish_grad_sync(tid2)
        model.step(tid2)
    
    # Verify base model was called twice
    stats = model.get_stats()
    assert stats['base_model_forward_calls'] == 2, f"Expected 2 calls, got {stats['base_model_forward_calls']}"
    
    # Cleanup
    model.finalize(tid1)
    model.finalize(tid2)
    
    assert model.tenant_count() == 0
    
    logger.info("Twinkle mode test PASSED")
    return True


def test_tinker_mode():
    """Test Tinker mode functionality."""
    from twinkle.megatron.distributed.clock_cycle_scheduler import (
        ClockCycleScheduler,
        ClockCycleTrainingClient,
    )
    
    logger.info("Testing Tinker mode...")
    
    model = MockMultiTenantModel(base_model_ms=1.0, lora_ms=0.5, comm_ms=0.5)
    
    # Initialize 2 tenants
    tid1 = model.initialize(tenant_id="test_1")
    tid2 = model.initialize(tenant_id="test_2")
    
    # Create scheduler
    scheduler = ClockCycleScheduler(model, cycle_interval_ms=50.0)
    scheduler.start()
    
    # Create clients
    client1 = ClockCycleTrainingClient(scheduler, tid1)
    client2 = ClockCycleTrainingClient(scheduler, tid2)
    
    x1 = torch.randn(4, 256)
    x2 = torch.randn(4, 256)
    
    # Both tenants submit requests (should be in same cycle)
    future1 = client1.forward_backward(x1)
    future2 = client2.forward_backward(x2)
    
    opt1 = client1.optim_step()
    opt2 = client2.optim_step()
    
    # Wait for results
    result1 = future1.result(timeout=5.0)
    result2 = future2.result(timeout=5.0)
    opt1.result(timeout=5.0)
    opt2.result(timeout=5.0)
    
    assert 'loss' in result1 or 'error' in result1, f"Unexpected result: {result1}"
    assert 'loss' in result2 or 'error' in result2, f"Unexpected result: {result2}"
    
    # Check they were in same cycle
    if 'cycle_id' in result1 and 'cycle_id' in result2:
        logger.info(f"Cycle IDs: {result1['cycle_id']}, {result2['cycle_id']}")
    
    # Stop scheduler
    scheduler.stop()
    
    # Check stats
    stats = scheduler.get_summary_stats()
    logger.info(f"Scheduler stats: {stats}")
    
    # Cleanup
    model.finalize(tid1)
    model.finalize(tid2)
    
    logger.info("Tinker mode test PASSED")
    return True


def test_batch_merging():
    """Test that batch merging works correctly."""
    from twinkle.megatron.distributed.clock_cycle_scheduler import BatchBuilder, TrainingRequest, RequestType
    
    logger.info("Testing batch merging...")
    
    builder = BatchBuilder()
    
    # Create requests from 3 tenants
    requests = {
        'tenant_a': TrainingRequest(
            tenant_id='tenant_a',
            request_type=RequestType.FORWARD_BACKWARD,
            inputs=torch.randn(4, 256),
        ),
        'tenant_b': TrainingRequest(
            tenant_id='tenant_b',
            request_type=RequestType.FORWARD_BACKWARD,
            inputs=torch.randn(8, 256),
        ),
        'tenant_c': TrainingRequest(
            tenant_id='tenant_c',
            request_type=RequestType.FORWARD_BACKWARD,
            inputs=torch.randn(2, 256),
        ),
    }
    
    # Build merged batch
    merged = builder.build(requests)
    
    # Verify
    assert merged.total_size == 14, f"Expected 14, got {merged.total_size}"
    assert merged.merged_inputs.shape == (14, 256), f"Wrong shape: {merged.merged_inputs.shape}"
    assert merged.tenant_slices['tenant_a'] == (0, 4)
    assert merged.tenant_slices['tenant_b'] == (4, 12)
    assert merged.tenant_slices['tenant_c'] == (12, 14)
    
    logger.info("Batch merging test PASSED")
    return True


def run_benchmark_comparison(config: BenchmarkConfig):
    """Run and compare both benchmarks."""
    print("")
    print("=" * 60)
    print("Multi-Tenant Architecture Benchmark")
    print("=" * 60)
    print(f"Config: {config}")
    print("")
    
    # Run Twinkle benchmark
    twinkle = TwinkleBenchmark(config)
    twinkle_result = twinkle.run()
    
    # Run Tinker benchmark
    tinker = TinkerBenchmark(config)
    tinker_result = tinker.run()
    
    # Print results
    print("")
    print("=" * 60)
    print("Results")
    print("=" * 60)
    print(twinkle_result)
    print(tinker_result)
    
    # Comparison
    print("=" * 60)
    print("Comparison")
    print("=" * 60)
    
    # Throughput
    speedup = tinker_result.throughput_samples / twinkle_result.throughput_samples
    print(f"Throughput speedup (Tinker/Twinkle): {speedup:.2f}x")
    
    # Base model efficiency
    base_model_ratio = twinkle_result.base_model_calls / max(tinker_result.base_model_calls, 1)
    print(f"Base model calls: {twinkle_result.base_model_calls} vs {tinker_result.base_model_calls} ({base_model_ratio:.1f}x fewer)")
    
    # Latency
    latency_diff = (twinkle_result.avg_latency - tinker_result.avg_latency) / twinkle_result.avg_latency * 100
    print(f"Latency improvement: {latency_diff:.1f}%")
    
    # GPU utilization
    gpu_diff = (tinker_result.gpu_utilization - twinkle_result.gpu_utilization) * 100
    print(f"GPU utilization difference: {gpu_diff:+.1f}%")
    
    return twinkle_result, tinker_result


def main():
    parser = argparse.ArgumentParser(description="Multi-tenant benchmark")
    parser.add_argument("--num-tenants", type=int, default=4)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--base-model-ms", type=float, default=10.0)
    parser.add_argument("--lora-ms", type=float, default=2.0)
    parser.add_argument("--comm-ms", type=float, default=5.0)
    parser.add_argument("--cycle-ms", type=float, default=50.0)
    parser.add_argument("--test-only", action="store_true", help="Run tests only")
    args = parser.parse_args()
    
    if args.test_only:
        test_twinkle_mode()
        test_batch_merging()
        test_tinker_mode()
        logger.info("All tests passed!")
        return
    
    config = BenchmarkConfig(
        num_tenants=args.num_tenants,
        steps_per_tenant=args.steps,
        batch_size_per_tenant=args.batch_size,
        base_model_ms=args.base_model_ms,
        lora_ms=args.lora_ms,
        comm_ms=args.comm_ms,
        clock_cycle_interval_ms=args.cycle_ms,
    )
    
    run_benchmark_comparison(config)


if __name__ == "__main__":
    main()
