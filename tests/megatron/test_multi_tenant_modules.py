#!/usr/bin/env python
"""
Unit tests for multi-tenant modules.

Tests:
1. tenant_context.py - ContextVar-based tenant management
2. tenant_manager.py - Tenant lifecycle management (WIP, requires PEFT)
3. multi_tenant_ddp.py - Multi-tenant DDP (requires Megatron)
4. clock_cycle_scheduler.py - Clock cycle scheduling

Each module is tested independently with mock dependencies where needed.
"""

import logging
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import torch
import torch.nn as nn

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ============================================================================
# Test 1: tenant_context.py
# ============================================================================

class TestTenantContext(unittest.TestCase):
    """Tests for tenant_context module."""
    
    def setUp(self):
        """Reset tenant context before each test."""
        from twinkle.megatron.distributed.tenant_context import set_current_tenant
        set_current_tenant(None)
    
    def test_get_set_tenant(self):
        """Test basic get/set tenant."""
        from twinkle.megatron.distributed.tenant_context import (
            get_current_tenant, set_current_tenant
        )
        
        # Initially None
        self.assertIsNone(get_current_tenant())
        
        # Set tenant
        set_current_tenant('tenant_a')
        self.assertEqual(get_current_tenant(), 'tenant_a')
        
        # Change tenant
        set_current_tenant('tenant_b')
        self.assertEqual(get_current_tenant(), 'tenant_b')
        
        # Clear tenant
        set_current_tenant(None)
        self.assertIsNone(get_current_tenant())
    
    def test_require_tenant(self):
        """Test require_tenant raises error when not set."""
        from twinkle.megatron.distributed.tenant_context import (
            require_tenant, set_current_tenant
        )
        
        # Should raise when not set
        with self.assertRaises(RuntimeError):
            require_tenant()
        
        # Should return when set
        set_current_tenant('tenant_a')
        self.assertEqual(require_tenant(), 'tenant_a')
    
    def test_tenant_scope(self):
        """Test tenant_scope context manager."""
        from twinkle.megatron.distributed.tenant_context import (
            get_current_tenant, tenant_scope, set_current_tenant
        )
        
        set_current_tenant('outer')
        
        with tenant_scope('inner'):
            self.assertEqual(get_current_tenant(), 'inner')
        
        # Should restore after context
        self.assertEqual(get_current_tenant(), 'outer')
    
    def test_nested_scopes(self):
        """Test nested tenant scopes."""
        from twinkle.megatron.distributed.tenant_context import (
            get_current_tenant, tenant_scope
        )
        
        with tenant_scope('a'):
            self.assertEqual(get_current_tenant(), 'a')
            
            with tenant_scope('b'):
                self.assertEqual(get_current_tenant(), 'b')
                
                with tenant_scope('c'):
                    self.assertEqual(get_current_tenant(), 'c')
                
                self.assertEqual(get_current_tenant(), 'b')
            
            self.assertEqual(get_current_tenant(), 'a')
    
    def test_thread_isolation(self):
        """Test that tenant context is isolated between threads."""
        from twinkle.megatron.distributed.tenant_context import (
            get_current_tenant, set_current_tenant
        )
        
        results = {}
        
        def thread_func(tenant_id: str, delay: float):
            set_current_tenant(tenant_id)
            time.sleep(delay)
            results[tenant_id] = get_current_tenant()
        
        # Run multiple threads
        threads = [
            threading.Thread(target=thread_func, args=('thread_a', 0.1)),
            threading.Thread(target=thread_func, args=('thread_b', 0.05)),
            threading.Thread(target=thread_func, args=('thread_c', 0.15)),
        ]
        
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        # Each thread should have its own context
        self.assertEqual(results['thread_a'], 'thread_a')
        self.assertEqual(results['thread_b'], 'thread_b')
        self.assertEqual(results['thread_c'], 'thread_c')
    
    def test_generate_tenant_id(self):
        """Test tenant ID generation."""
        from twinkle.megatron.distributed.tenant_context import generate_tenant_id
        
        id1 = generate_tenant_id()
        id2 = generate_tenant_id()
        
        # Should be unique
        self.assertNotEqual(id1, id2)
        
        # Should be 8 chars
        self.assertEqual(len(id1), 8)
        self.assertEqual(len(id2), 8)
    
    def test_with_tenant_context_decorator(self):
        """Test @with_tenant_context decorator."""
        from twinkle.megatron.distributed.tenant_context import (
            with_tenant_context, tenant_scope
        )
        
        @with_tenant_context
        def example_func(tenant_id: Optional[str] = None):
            return tenant_id
        
        # Should use context when tenant_id not provided
        with tenant_scope('context_tenant'):
            result = example_func()
            self.assertEqual(result, 'context_tenant')
        
        # Should use explicit tenant_id when provided
        with tenant_scope('context_tenant'):
            result = example_func(tenant_id='explicit_tenant')
            self.assertEqual(result, 'explicit_tenant')


# ============================================================================
# Test 2: clock_cycle_scheduler.py
# ============================================================================

class MockMultiTenantModel(nn.Module):
    """Mock model that implements the required interface for ClockCycleScheduler."""
    
    def __init__(self, hidden_size: int = 64, simulate_ms: float = 1.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.simulate_ms = simulate_ms
        
        # Base layer (frozen)
        self.base = nn.Linear(hidden_size, hidden_size)
        self.base.weight.requires_grad = False
        
        # Per-tenant adapters
        self._adapters: Dict[str, nn.Module] = {}
        self._optimizers: Dict[str, torch.optim.Optimizer] = {}
        self._current_tenant: Optional[str] = None
    
    def add_tenant(self, tenant_id: str) -> None:
        """Add a tenant with LoRA adapter."""
        adapter = nn.Linear(self.hidden_size, self.hidden_size)
        self._adapters[tenant_id] = adapter
        self._optimizers[tenant_id] = torch.optim.SGD(adapter.parameters(), lr=0.01)
    
    def remove_tenant(self, tenant_id: str) -> None:
        """Remove a tenant."""
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
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass using current tenant's adapter."""
        time.sleep(self.simulate_ms / 1000.0)
        
        out = self.base(x)
        if self._current_tenant and self._current_tenant in self._adapters:
            out = out + self._adapters[self._current_tenant](x)
        return out
    
    def zero_grad(self, tenant_id: str) -> None:
        """Zero gradients for tenant."""
        if tenant_id in self._optimizers:
            self._optimizers[tenant_id].zero_grad(set_to_none=True)
    
    def step(self, tenant_id: str) -> None:
        """Optimizer step for tenant."""
        if tenant_id in self._optimizers:
            self._optimizers[tenant_id].step()
    
    def clip_grad_norm(self, tenant_id: str, max_norm: float = 1.0) -> None:
        """Clip gradients for tenant."""
        if tenant_id in self._adapters:
            torch.nn.utils.clip_grad_norm_(
                self._adapters[tenant_id].parameters(), max_norm
            )
    
    def finish_grad_sync(self, tenant_id: str) -> None:
        """Gradient sync for single tenant (no-op in non-distributed)."""
        pass
    
    def finish_grad_sync_batched(self, tenant_ids: List[str]) -> None:
        """Batched gradient sync (no-op in non-distributed)."""
        pass


class TestClockCycleScheduler(unittest.TestCase):
    """Tests for clock_cycle_scheduler module."""
    
    def test_model_interface_validation(self):
        """Test that scheduler validates model interface."""
        from twinkle.megatron.distributed.clock_cycle_scheduler import (
            ClockCycleScheduler, ModelInterfaceError
        )
        
        # Model without required methods should fail
        bad_model = nn.Linear(10, 10)
        with self.assertRaises(ModelInterfaceError):
            ClockCycleScheduler(bad_model)
        
        # Good model should work
        good_model = MockMultiTenantModel()
        scheduler = ClockCycleScheduler(good_model, cycle_interval_ms=10)
        self.assertIsNotNone(scheduler)
    
    def test_basic_training_step(self):
        """Test basic training step through scheduler."""
        from twinkle.megatron.distributed.clock_cycle_scheduler import (
            ClockCycleScheduler, ClockCycleTrainingClient
        )
        
        model = MockMultiTenantModel(simulate_ms=0.5)
        model.add_tenant('tenant_a')
        
        scheduler = ClockCycleScheduler(model, cycle_interval_ms=20)
        scheduler.start()
        
        try:
            client = ClockCycleTrainingClient(scheduler, 'tenant_a')
            
            x = torch.randn(4, 64)
            result = client.train_step(x)
            
            self.assertIn('loss', result)
            self.assertIn('cycle_id', result)
            self.assertEqual(result['batch_size'], 4)
            
        finally:
            scheduler.stop()
            model.remove_tenant('tenant_a')
    
    def test_multi_tenant_concurrent(self):
        """Test multiple tenants submitting concurrently."""
        from twinkle.megatron.distributed.clock_cycle_scheduler import (
            ClockCycleScheduler, ClockCycleTrainingClient
        )
        
        model = MockMultiTenantModel(simulate_ms=0.5)
        model.add_tenant('tenant_a')
        model.add_tenant('tenant_b')
        model.add_tenant('tenant_c')
        
        scheduler = ClockCycleScheduler(model, cycle_interval_ms=50)
        scheduler.start()
        
        try:
            clients = {
                tid: ClockCycleTrainingClient(scheduler, tid)
                for tid in ['tenant_a', 'tenant_b', 'tenant_c']
            }
            
            # Submit from multiple threads
            results = {}
            
            def worker(tenant_id: str, client: ClockCycleTrainingClient):
                x = torch.randn(4, 64)
                return client.train_step(x)
            
            with ThreadPoolExecutor(max_workers=3) as executor:
                futures = {
                    executor.submit(worker, tid, clients[tid]): tid
                    for tid in clients
                }
                for future in as_completed(futures):
                    tid = futures[future]
                    results[tid] = future.result()
            
            # All should succeed
            for tid, result in results.items():
                self.assertIn('loss', result)
                self.assertIn('cycle_id', result)
            
            # Check stats
            stats = scheduler.get_summary_stats()
            self.assertGreater(stats['total_cycles'], 0)
            self.assertEqual(stats['total_samples'], 12)  # 3 tenants * 4 samples
            
        finally:
            scheduler.stop()
            for tid in ['tenant_a', 'tenant_b', 'tenant_c']:
                model.remove_tenant(tid)
    
    def test_gradient_isolation(self):
        """Test that gradients are isolated between tenants."""
        from twinkle.megatron.distributed.clock_cycle_scheduler import (
            ClockCycleScheduler, ClockCycleTrainingClient
        )
        
        model = MockMultiTenantModel(simulate_ms=0.5)
        model.add_tenant('tenant_a')
        model.add_tenant('tenant_b')
        
        # Get initial weights
        weight_a_before = model._adapters['tenant_a'].weight.clone()
        weight_b_before = model._adapters['tenant_b'].weight.clone()
        
        scheduler = ClockCycleScheduler(model, cycle_interval_ms=20)
        scheduler.start()
        
        try:
            # Only tenant_a trains
            client_a = ClockCycleTrainingClient(scheduler, 'tenant_a')
            x = torch.randn(4, 64)
            client_a.train_step(x)
            
            # tenant_a weights should change
            weight_a_after = model._adapters['tenant_a'].weight
            self.assertFalse(torch.allclose(weight_a_before, weight_a_after))
            
            # tenant_b weights should NOT change
            weight_b_after = model._adapters['tenant_b'].weight
            self.assertTrue(torch.allclose(weight_b_before, weight_b_after))
            
        finally:
            scheduler.stop()
            model.remove_tenant('tenant_a')
            model.remove_tenant('tenant_b')
    
    def test_error_handling(self):
        """Test error handling for failed requests."""
        from twinkle.megatron.distributed.clock_cycle_scheduler import (
            ClockCycleScheduler
        )
        
        # Create a model that raises error on forward for unknown tenant
        class FailingModel(MockMultiTenantModel):
            def forward(self, x):
                if self._current_tenant not in self._adapters:
                    raise KeyError(f"Tenant '{self._current_tenant}' not found")
                return super().forward(x)
        
        model = FailingModel()
        # Don't add any tenants - requests should fail
        
        scheduler = ClockCycleScheduler(model, cycle_interval_ms=20)
        scheduler.start()
        
        try:
            # Submit request for non-existent tenant
            future = scheduler.submit_forward_backward('nonexistent', torch.randn(4, 64))
            
            # Should raise exception
            with self.assertRaises(Exception):
                future.result(timeout=5.0)
                
        finally:
            scheduler.stop()
    
    def test_cycle_stats(self):
        """Test cycle statistics collection."""
        from twinkle.megatron.distributed.clock_cycle_scheduler import (
            ClockCycleScheduler, ClockCycleTrainingClient
        )
        
        model = MockMultiTenantModel(simulate_ms=1.0)
        model.add_tenant('tenant_a')
        
        scheduler = ClockCycleScheduler(model, cycle_interval_ms=50)
        scheduler.start()
        
        try:
            client = ClockCycleTrainingClient(scheduler, 'tenant_a')
            
            # Run multiple steps
            for _ in range(3):
                x = torch.randn(4, 64)
                client.train_step(x)
            
            # Get stats
            stats_list = scheduler.get_stats()
            summary = scheduler.get_summary_stats()
            
            self.assertEqual(len(stats_list), 3)
            self.assertEqual(summary['total_cycles'], 3)
            self.assertEqual(summary['total_samples'], 12)
            
            # Check individual stats
            for stat in stats_list:
                self.assertGreater(stat.forward_time, 0)
                self.assertGreater(stat.duration, 0)
                
        finally:
            scheduler.stop()
            model.remove_tenant('tenant_a')


# ============================================================================
# Test 3: multi_tenant_ddp.py (Mock test - requires Megatron)
# ============================================================================

class TestMultiTenantDDP(unittest.TestCase):
    """Tests for multi_tenant_ddp module (mocked)."""
    
    def test_tenant_ddp_state_dataclass(self):
        """Test TenantDDPState dataclass."""
        from twinkle.megatron.distributed.multi_tenant_ddp import TenantDDPState
        
        state = TenantDDPState(tenant_id='test_tenant')
        
        self.assertEqual(state.tenant_id, 'test_tenant')
        self.assertEqual(state.params, [])
        self.assertEqual(state.buffers, [])
        self.assertEqual(state.bucket_groups, [])
        self.assertIsNone(state.process_group)
    
    @unittest.skipUnless(
        False,  # Skip by default - requires Megatron
        "Requires Megatron-Core"
    )
    def test_multi_tenant_lora_ddp_creation(self):
        """Test MultiTenantLoRADDP creation (requires Megatron)."""
        pass
    
    def test_requires_megatron(self):
        """Test that MultiTenantLoRADDP requires Megatron."""
        from unittest.mock import MagicMock, patch
        
        with patch('twinkle.megatron.distributed.multi_tenant_ddp.MEGATRON_AVAILABLE', False):
            from twinkle.megatron.distributed.multi_tenant_ddp import MultiTenantLoRADDP
            
            with self.assertRaises(ImportError):
                MultiTenantLoRADDP(
                    config=MagicMock(),
                    ddp_config=MagicMock(),
                    module=nn.Linear(10, 10),
                )


# ============================================================================
# Test 4: MegatronMultiAdapter
# ============================================================================

class TestMegatronMultiAdapter(unittest.TestCase):
    """Tests for MegatronMultiAdapter."""
    
    def test_adapter_context_var(self):
        """Test adapter name ContextVar management."""
        from twinkle.megatron.model.multi_tenant_megatron import MegatronMultiAdapter
        
        # Reset state
        MegatronMultiAdapter._patched = False
        
        # Test get/set
        self.assertIsNone(MegatronMultiAdapter.get_current_adapter_name())
        MegatronMultiAdapter.set_current_adapter_name("adapter_a")
        self.assertEqual(MegatronMultiAdapter.get_current_adapter_name(), "adapter_a")
        MegatronMultiAdapter.set_current_adapter_name(None)
        self.assertIsNone(MegatronMultiAdapter.get_current_adapter_name())


# ============================================================================
# Test 5: Integration test - tenant_context + clock_cycle_scheduler
# ============================================================================

class TestIntegration(unittest.TestCase):
    """Integration tests combining multiple modules."""
    
    def test_context_with_scheduler(self):
        """Test that tenant_context works with scheduler."""
        from twinkle.megatron.distributed.tenant_context import (
            get_current_tenant, tenant_scope, set_current_tenant
        )
        from twinkle.megatron.distributed.clock_cycle_scheduler import (
            ClockCycleScheduler, ClockCycleTrainingClient
        )
        
        model = MockMultiTenantModel(simulate_ms=0.5)
        model.add_tenant('tenant_a')
        model.add_tenant('tenant_b')
        
        scheduler = ClockCycleScheduler(model, cycle_interval_ms=30)
        scheduler.start()
        
        try:
            # Test that context propagates correctly
            with tenant_scope('tenant_a'):
                self.assertEqual(get_current_tenant(), 'tenant_a')
                
                client = ClockCycleTrainingClient(scheduler, 'tenant_a')
                x = torch.randn(4, 64)
                result = client.train_step(x)
                
                self.assertIn('loss', result)
            
            # Context should be cleared outside
            set_current_tenant(None)
            self.assertIsNone(get_current_tenant())
            
        finally:
            scheduler.stop()
            model.remove_tenant('tenant_a')
            model.remove_tenant('tenant_b')
    
    def test_multi_threaded_with_context(self):
        """Test multi-threaded training with tenant context."""
        from twinkle.megatron.distributed.tenant_context import (
            get_current_tenant, tenant_scope
        )
        from twinkle.megatron.distributed.clock_cycle_scheduler import (
            ClockCycleScheduler, ClockCycleTrainingClient
        )
        
        model = MockMultiTenantModel(simulate_ms=0.5)
        for i in range(4):
            model.add_tenant(f'tenant_{i}')
        
        scheduler = ClockCycleScheduler(model, cycle_interval_ms=50)
        scheduler.start()
        
        results = {}
        errors = []
        
        def worker(tenant_id: str):
            try:
                with tenant_scope(tenant_id):
                    # Verify context is correct
                    if get_current_tenant() != tenant_id:
                        errors.append(f"Context mismatch for {tenant_id}")
                        return
                    
                    client = ClockCycleTrainingClient(scheduler, tenant_id)
                    x = torch.randn(4, 64)
                    result = client.train_step(x)
                    results[tenant_id] = result
            except Exception as e:
                errors.append(str(e))
        
        try:
            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = [
                    executor.submit(worker, f'tenant_{i}')
                    for i in range(4)
                ]
                for f in futures:
                    f.result()
            
            self.assertEqual(len(errors), 0, f"Errors: {errors}")
            self.assertEqual(len(results), 4)
            
            for tid, result in results.items():
                self.assertIn('loss', result)
                
        finally:
            scheduler.stop()
            for i in range(4):
                model.remove_tenant(f'tenant_{i}')


# ============================================================================
# Main
# ============================================================================

def run_tests():
    """Run all tests."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test cases
    suite.addTests(loader.loadTestsFromTestCase(TestTenantContext))
    suite.addTests(loader.loadTestsFromTestCase(TestClockCycleScheduler))
    suite.addTests(loader.loadTestsFromTestCase(TestMultiTenantDDP))
    suite.addTests(loader.loadTestsFromTestCase(TestMegatronMultiAdapter))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegration))
    
    # Run
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    exit(0 if success else 1)
