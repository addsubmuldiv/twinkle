import sys
import types
import unittest

import torch

from twinkle.kernel import kernelize_model
from twinkle.kernel.base import is_kernels_available
from twinkle.kernel.function import register_function_kernel, apply_function_kernel
from twinkle.kernel.registry import get_global_function_registry


def _ensure_test_packages() -> None:
    if "tests" not in sys.modules:
        tests_pkg = types.ModuleType("tests")
        tests_pkg.__path__ = []
        sys.modules["tests"] = tests_pkg
    if "tests.kernel" not in sys.modules:
        kernel_pkg = types.ModuleType("tests.kernel")
        kernel_pkg.__path__ = []
        sys.modules["tests.kernel"] = kernel_pkg


class TestMpsCustomKernel(unittest.TestCase):
    def setUp(self):
        get_global_function_registry()._clear()

    def tearDown(self):
        get_global_function_registry()._clear()

    def test_custom_mps_function_kernel(self):
        if not torch.backends.mps.is_available():
            self.skipTest("MPS not available in this environment.")

        _ensure_test_packages()
        module_name = "tests.kernel._tmp_mps_module"
        temp_module = types.ModuleType(module_name)

        def original(x):
            return x + 1

        def mps_impl(x):
            return x + 10

        temp_module.target = original
        temp_module.__path__ = []
        sys.modules[module_name] = temp_module

        try:
            register_function_kernel(
                func_name="target",
                target_module=module_name,
                func_impl=mps_impl,
                device="mps",
                mode="inference",
            )

            apply_function_kernel(
                target_module=module_name,
                device="mps",
                mode="inference",
            )

            x = torch.ones(4, device="mps")
            y = temp_module.target(x)
            self.assertTrue(torch.allclose(y, x + 10))
            self.assertFalse(torch.allclose(y, x + 1))
        finally:
            sys.modules.pop(module_name, None)

    def test_custom_mps_function_kernel_device_filter(self):
        if not torch.backends.mps.is_available():
            self.skipTest("MPS not available in this environment.")

        _ensure_test_packages()
        module_name = "tests.kernel._tmp_mps_module_device"
        temp_module = types.ModuleType(module_name)

        def original(x):
            return x + 1

        def mps_impl(x):
            return x + 10

        temp_module.target = original
        temp_module.__path__ = []
        sys.modules[module_name] = temp_module

        try:
            register_function_kernel(
                func_name="target",
                target_module=module_name,
                func_impl=mps_impl,
                device="mps",
                mode="inference",
            )

            apply_function_kernel(
                target_module=module_name,
                device="cpu",
                mode="inference",
            )

            x = torch.ones(4, device="mps")
            y = temp_module.target(x)
            self.assertTrue(torch.allclose(y, x + 1))
            self.assertFalse(torch.allclose(y, x + 10))
        finally:
            sys.modules.pop(module_name, None)

    def test_custom_mps_function_kernel_mode_filter(self):
        if not torch.backends.mps.is_available():
            self.skipTest("MPS not available in this environment.")

        _ensure_test_packages()
        module_name = "tests.kernel._tmp_mps_module_mode"
        temp_module = types.ModuleType(module_name)

        def original(x):
            return x + 1

        def mps_impl(x):
            return x + 10

        temp_module.target = original
        temp_module.__path__ = []
        sys.modules[module_name] = temp_module

        try:
            register_function_kernel(
                func_name="target",
                target_module=module_name,
                func_impl=mps_impl,
                device="mps",
                mode="train",
            )

            apply_function_kernel(
                target_module=module_name,
                device="mps",
                mode="inference",
            )

            x = torch.ones(4, device="mps")
            y = temp_module.target(x)
            self.assertTrue(torch.allclose(y, x + 1))
            self.assertFalse(torch.allclose(y, x + 10))
        finally:
            sys.modules.pop(module_name, None)

    def test_custom_mps_function_kernel_training_mode(self):
        if not torch.backends.mps.is_available():
            self.skipTest("MPS not available in this environment.")

        _ensure_test_packages()
        module_name = "tests.kernel._tmp_mps_module_train"
        temp_module = types.ModuleType(module_name)

        def original(x):
            return x + 1

        def mps_impl(x):
            return x + 10

        temp_module.target = original
        temp_module.__path__ = []
        sys.modules[module_name] = temp_module

        try:
            register_function_kernel(
                func_name="target",
                target_module=module_name,
                func_impl=mps_impl,
                device="mps",
                mode="train",
            )

            apply_function_kernel(
                target_module=module_name,
                device="mps",
                mode="train",
            )

            x = torch.ones(4, device="mps")
            y = temp_module.target(x)
            self.assertTrue(torch.allclose(y, x + 10))
            self.assertFalse(torch.allclose(y, x + 1))
        finally:
            sys.modules.pop(module_name, None)

    def test_custom_mps_function_kernel_compile_mode(self):
        if not torch.backends.mps.is_available():
            self.skipTest("MPS not available in this environment.")

        _ensure_test_packages()
        module_name = "tests.kernel._tmp_mps_module_compile"
        temp_module = types.ModuleType(module_name)

        def original(x):
            return x + 1

        class KernelImpl:
            can_torch_compile = True
            has_backward = True

            def __call__(self, x):
                return x + 10

        temp_module.target = original
        temp_module.__path__ = []
        sys.modules[module_name] = temp_module

        try:
            register_function_kernel(
                func_name="target",
                target_module=module_name,
                func_impl=KernelImpl(),
                device="mps",
                mode="inference",
            )

            apply_function_kernel(
                target_module=module_name,
                device="mps",
                mode="inference",
            )

            x = torch.ones(4, device="mps")
            y = temp_module.target(x)
            self.assertTrue(torch.allclose(y, x + 10))
            self.assertFalse(torch.allclose(y, x + 1))
        finally:
            sys.modules.pop(module_name, None)

    def test_custom_mps_kernelize_model(self):
        if not torch.backends.mps.is_available():
            self.skipTest("MPS not available in this environment.")
        if not is_kernels_available():
            self.skipTest("kernels package not available in this environment.")

        _ensure_test_packages()
        module_name = "tests.kernel._tmp_mps_module_kernelize"
        temp_module = types.ModuleType(module_name)

        def original(x):
            return x + 1

        def mps_impl(x):
            return x + 10

        temp_module.target = original
        temp_module.__path__ = []
        sys.modules[module_name] = temp_module

        try:
            register_function_kernel(
                func_name="target",
                target_module=module_name,
                func_impl=mps_impl,
                device="mps",
                mode="inference",
            )

            model = torch.nn.Identity().to("mps")
            kernelize_model(model=model, mode="inference", device="mps", use_fallback=True)

            x = torch.ones(4, device="mps")
            y = temp_module.target(x)
            self.assertTrue(torch.allclose(y, x + 10))
            self.assertFalse(torch.allclose(y, x + 1))
        finally:
            sys.modules.pop(module_name, None)

    def test_custom_mps_kernelize_model_mode_filter(self):
        if not torch.backends.mps.is_available():
            self.skipTest("MPS not available in this environment.")
        if not is_kernels_available():
            self.skipTest("kernels package not available in this environment.")

        _ensure_test_packages()
        module_name = "tests.kernel._tmp_mps_module_kernelize_mode"
        temp_module = types.ModuleType(module_name)

        def original(x):
            return x + 1

        def mps_impl(x):
            return x + 10

        temp_module.target = original
        temp_module.__path__ = []
        sys.modules[module_name] = temp_module

        try:
            register_function_kernel(
                func_name="target",
                target_module=module_name,
                func_impl=mps_impl,
                device="mps",
                mode="train",
            )

            model = torch.nn.Identity()
            kernelize_model(model=model, mode="inference", device="mps", use_fallback=True)

            x = torch.ones(4, device="mps")
            y = temp_module.target(x)
            self.assertTrue(torch.allclose(y, x + 1))
            self.assertFalse(torch.allclose(y, x + 10))
        finally:
            sys.modules.pop(module_name, None)

    def test_custom_mps_kernelize_model_device_filter(self):
        if not torch.backends.mps.is_available():
            self.skipTest("MPS not available in this environment.")
        if not is_kernels_available():
            self.skipTest("kernels package not available in this environment.")

        _ensure_test_packages()
        module_name = "tests.kernel._tmp_mps_module_kernelize_device"
        temp_module = types.ModuleType(module_name)

        def original(x):
            return x + 1

        def mps_impl(x):
            return x + 10

        temp_module.target = original
        temp_module.__path__ = []
        sys.modules[module_name] = temp_module

        try:
            register_function_kernel(
                func_name="target",
                target_module=module_name,
                func_impl=mps_impl,
                device="mps",
                mode="inference",
            )

            model = torch.nn.Identity()
            kernelize_model(model=model, mode="inference", device="cpu", use_fallback=True)

            x = torch.ones(4, device="mps")
            y = temp_module.target(x)
            print(y)
            self.assertTrue(torch.allclose(y, x + 1))
            self.assertFalse(torch.allclose(y, x + 10))
        finally:
            sys.modules.pop(module_name, None)


if __name__ == "__main__":
    unittest.main()
