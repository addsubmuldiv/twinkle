# Copyright (c) ModelScope Contributors. All rights reserved.
"""
Kernel module unit tests
"""
import os
import unittest
from unittest.mock import Mock, MagicMock, patch

from twinkle.kernel import (
    kernelize_model,
    register_layer_kernel,
    register_external_layer,
    register_kernels,
)
from twinkle.kernel.base import (
    get_device_type,
    is_kernels_available,
    is_kernels_enabled,
    to_kernels_mode,
)
from twinkle.kernel.registry import (
    LayerRegistry,
    ExternalLayerRegistry,
    register_layer,
    get_layer_spec,
    get_global_layer_registry,
    get_global_external_layer_registry,
)


class TestBase(unittest.TestCase):
    """测试基础类和环境变量"""

    def test_get_device_type_no_torch(self):
        """测试无 torch 时的设备检测"""
        with patch("twinkle.kernel.base.exists", return_value=False):
            result = get_device_type()
            self.assertIsNone(result)

    def test_is_kernels_available(self):
        """测试 kernels 可用性检测"""
        result = is_kernels_available()
        self.assertIsInstance(result, bool)

    def test_kernels_enabled_env_var(self):
        """测试环境变量控制"""
        original = os.environ.get("TWINKLE_USE_KERNELS")
        try:
            os.environ["TWINKLE_USE_KERNELS"] = "YES"
            from twinkle.kernel.base import _kernels_enabled
            self.assertTrue(_kernels_enabled())

            os.environ["TWINKLE_USE_KERNELS"] = "NO"
            import importlib
            import twinkle.kernel.base
            importlib.reload(twinkle.kernel.base)
            from twinkle.kernel.base import _kernels_enabled
            self.assertFalse(_kernels_enabled())
        finally:
            if original is not None:
                os.environ["TWINKLE_USE_KERNELS"] = original
            else:
                os.environ.pop("TWINKLE_USE_KERNELS", None)

    def test_to_kernels_mode(self):
        """测试 mode 转换"""
        if not is_kernels_available():
            self.skipTest("kernels package not available")

        self.assertEqual(to_kernels_mode("train").name, "TRAINING")
        self.assertEqual(to_kernels_mode("inference").name, "INFERENCE")
        self.assertEqual(to_kernels_mode("compile").name, "TORCH_COMPILE")


class TestLayerRegistry(unittest.TestCase):
    """测试层注册表"""

    def setUp(self):
        self.registry = LayerRegistry()

    def test_register_and_get(self):
        """测试注册和获取"""
        mock_spec = Mock()
        self.registry.register("TestLayer", mock_spec, "cuda")

        result = self.registry.get("TestLayer", "cuda")
        self.assertEqual(result, mock_spec)

        result = self.registry.get("NonExistent", "cuda")
        self.assertIsNone(result)

    def test_register_multiple_devices(self):
        """测试多设备注册"""
        mock_cuda = Mock()
        mock_npu = Mock()

        self.registry.register("TestLayer", mock_cuda, "cuda")
        self.registry.register("TestLayer", mock_npu, "npu")

        self.assertEqual(self.registry.get("TestLayer", "cuda"), mock_cuda)
        self.assertEqual(self.registry.get("TestLayer", "npu"), mock_npu)

    def test_get_without_device(self):
        """测试不指定设备时获取第一个"""
        mock_spec = Mock()
        self.registry.register("TestLayer", mock_spec, "cuda")

        result = self.registry.get("TestLayer")
        self.assertEqual(result, mock_spec)

    def test_has(self):
        """测试是否存在检查"""
        mock_spec = Mock()
        self.assertFalse(self.registry.has("TestLayer"))

        self.registry.register("TestLayer", mock_spec, "cuda")
        self.assertTrue(self.registry.has("TestLayer"))
        self.assertTrue(self.registry.has("TestLayer", "cuda"))
        self.assertFalse(self.registry.has("TestLayer", "npu"))

    def test_list_kernel_names(self):
        """测试列出 kernel 名称"""
        mock_spec = Mock()
        self.registry.register("Layer1", mock_spec, "cuda")
        self.registry.register("Layer2", mock_spec, "cuda")

        names = self.registry.list_kernel_names()
        self.assertCountEqual(names, ["Layer1", "Layer2"])


class TestExternalLayerRegistry(unittest.TestCase):
    """测试外部层注册表"""

    def setUp(self):
        self.registry = ExternalLayerRegistry()

    def test_register_and_get(self):
        """测试注册和获取"""
        mock_class = Mock
        self.registry.register(mock_class, "LlamaAttention")

        result = self.registry.get(mock_class)
        self.assertEqual(result, "LlamaAttention")

    def test_has(self):
        """测试是否存在检查"""
        mock_class = Mock
        self.assertFalse(self.registry.has(mock_class))

        self.registry.register(mock_class, "LlamaAttention")
        self.assertTrue(self.registry.has(mock_class))

    def test_list_mappings(self):
        """测试列出所有映射"""
        class MockClass1:
            pass

        class MockClass2:
            pass

        self.registry.register(MockClass1, "LlamaAttention")
        self.registry.register(MockClass2, "LlamaMLP")

        mappings = self.registry.list_mappings()
        self.assertEqual(len(mappings), 2)


class TestRegisterLayer(unittest.TestCase):
    """测试全局注册函数"""

    def setUp(self):
        get_global_layer_registry()._clear()

    def test_register_and_get_spec(self):
        """测试全局注册和获取"""
        mock_spec = Mock()
        register_layer("TestLayer", mock_spec, "cuda")

        result = get_layer_spec("TestLayer", "cuda")
        self.assertEqual(result, mock_spec)


class TestRegisterLayerKernel(unittest.TestCase):
    """测试 register_layer_kernel 函数"""

    def setUp(self):
        get_global_layer_registry()._clear()

    def test_register_without_kernels_package(self):
        """测试无 kernels 包时的注册"""
        with patch("twinkle.kernel.layer.is_kernels_available", return_value=False):
            register_layer_kernel("TestLayer", repo_id="test/repo")
            self.assertIsNone(get_layer_spec("TestLayer"))

    def test_register_with_kernels_package(self):
        """测试有 kernels 包时的注册"""
        if not is_kernels_available():
            self.skipTest("kernels package not available")

        register_layer_kernel(
            kernel_name="TestLayer",
            repo_id="kernels-community/test",
        )

        self.assertIsNotNone(get_layer_spec("TestLayer"))


class TestKernelizeModel(unittest.TestCase):
    """测试 kernelize_model 函数"""

    def test_kernelize_without_kernels_enabled(self):
        """测试 kernels 未启用时返回原模型"""
        with patch("twinkle.kernel.layer.is_kernels_enabled", return_value=False):
            mock_model = Mock()
            result = kernelize_model(mock_model)
            self.assertEqual(result, mock_model)

    @patch("twinkle.kernel.layer.is_kernels_enabled", return_value=True)
    @patch("twinkle.kernel.layer.is_kernels_available", return_value=False)
    def test_kernelize_without_kernels_available(self, mock_available, mock_enabled):
        """测试 kernels 不可用时返回原模型"""
        mock_model = Mock()
        result = kernelize_model(mock_model)
        self.assertEqual(result, mock_model)


class TestRegisterExternalLayer(unittest.TestCase):
    """测试 register_external_layer 函数"""

    def setUp(self):
        get_global_external_layer_registry()._clear()

    def test_register_external_layer(self):
        """测试注册外部层"""
        mock_class = Mock

        register_external_layer(mock_class, "LlamaAttention")

        result = get_global_external_layer_registry().get(mock_class)
        self.assertEqual(result, "LlamaAttention")

    def test_register_external_qwen_layer(self):
        """测试注册 Qwen2 外部层映射"""
        try:
            from transformers.models.qwen2.modeling_qwen2 import Qwen2Attention
        except ImportError:
            self.skipTest("transformers package not available")

        register_external_layer(Qwen2Attention, "LlamaAttention")

        registry = get_global_external_layer_registry()
        self.assertTrue(registry.has(Qwen2Attention))
        self.assertEqual(registry.get(Qwen2Attention), "LlamaAttention")

    def test_register_external_layer_adds_kernel_layer_name(self):
        """测试 register_external_layer 添加 kernel_layer_name 属性"""
        if not is_kernels_available():
            self.skipTest("kernels package not available")

        class TestLayer:
            pass

        register_external_layer(TestLayer, "TestKernel")

        self.assertTrue(hasattr(TestLayer, "kernel_layer_name"))
        self.assertEqual(TestLayer.kernel_layer_name, "TestKernel")


class TestRegisterKernels(unittest.TestCase):
    """测试 register_kernels 批量注册函数"""

    def setUp(self):
        get_global_layer_registry()._clear()

    @patch("twinkle.kernel.layer.is_kernels_available", return_value=False)
    def test_register_layers_without_kernels(self, mock_available):
        """测试无 kernels 包时批量注册"""
        config = {
            "layers": {
                "LlamaAttention": {"repo_id": "kernels-community/llama-attention"},
                "LlamaMLP": {"repo_id": "kernels-community/llama-mlp"},
            }
        }

        register_kernels(config)

        self.assertIsNone(get_layer_spec("LlamaAttention"))
        self.assertIsNone(get_layer_spec("LlamaMLP"))

    def test_register_functions_not_implemented(self):
        """测试函数级别注册尚未实现"""
        config = {
            "functions": {
                "apply_rotary_pos_emb": {"func_impl": Mock, "target_module": "test"}
            }
        }

        register_kernels(config)


class TestModeSupport(unittest.TestCase):
    """测试 mode 参数支持"""

    def setUp(self):
        get_global_layer_registry()._clear()

    @patch("twinkle.kernel.layer.is_kernels_available", return_value=False)
    def test_register_with_mode_fallback(self, mock_available):
        """测试注册时 mode=None 使用 FALLBACK"""
        from twinkle.kernel.layer import register_layer_kernel, _to_hf_mode
        from kernels import Mode

        result = _to_hf_mode(None)
        self.assertEqual(result, Mode.FALLBACK)

    def test_to_hf_mode_conversion(self):
        """测试 Twinkle mode 到 HF kernels Mode 的转换"""
        if not is_kernels_available():
            self.skipTest("kernels package not available")

        from twinkle.kernel.layer import _to_hf_mode
        from kernels import Mode

        self.assertEqual(_to_hf_mode("train"), Mode.TRAINING)
        self.assertEqual(_to_hf_mode("inference"), Mode.INFERENCE)
        self.assertEqual(_to_hf_mode("compile"), Mode.TORCH_COMPILE)

    @patch("twinkle.kernel.layer.is_kernels_available", return_value=False)
    def test_register_multiple_modes(self, mock_available):
        """测试为同一层注册不同 mode 的 kernel"""
        registry = get_global_layer_registry()

        class MockRepo:
            pass

        repo_inference = MockRepo()
        repo_training = MockRepo()

        from kernels import Mode

        registry.register("TestLayer", repo_inference, "cuda", Mode.INFERENCE)
        registry.register("TestLayer", repo_training, "cuda", Mode.TRAINING)

        self.assertTrue(registry.has("TestLayer", "cuda", Mode.INFERENCE))
        self.assertTrue(registry.has("TestLayer", "cuda", Mode.TRAINING))

        result = registry.get("TestLayer", "cuda", Mode.INFERENCE)
        self.assertEqual(result, repo_inference)


if __name__ == "__main__":
    unittest.main()
