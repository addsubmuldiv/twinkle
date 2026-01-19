import importlib.util
import sys
import types
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _install_kernel_stubs():
    if "kernels" in sys.modules:
        return

    kernels_mod = types.ModuleType("kernels")
    kernels_mod.__path__ = []
    layer_mod = types.ModuleType("kernels.layer")
    layer_mod.__path__ = []
    func_mod = types.ModuleType("kernels.layer.func")

    class DummyProtocol:
        pass

    func_mod.FuncRepositoryProtocol = DummyProtocol

    versions_mod = types.ModuleType("kernels._versions")

    def select_revision_or_version(repo_id, revision, version):
        return revision or version

    versions_mod.select_revision_or_version = select_revision_or_version

    utils_mod = types.ModuleType("kernels.utils")

    def get_kernel(repo_id, revision=None):
        raise RuntimeError("Stub get_kernel called unexpectedly.")

    utils_mod.get_kernel = get_kernel

    sys.modules["kernels"] = kernels_mod
    sys.modules["kernels.layer"] = layer_mod
    sys.modules["kernels.layer.func"] = func_mod
    sys.modules["kernels._versions"] = versions_mod
    sys.modules["kernels.utils"] = utils_mod


def _load_function_kernel():
    _install_kernel_stubs()
    path = ROOT / "twinkle" / "kernel" / "function.py"
    spec = importlib.util.spec_from_file_location(
        "function_kernel_under_test",
        path,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


function_kernel = _load_function_kernel()


class TestFunctionKernel(unittest.TestCase):
    def setUp(self):
        function_kernel._FUNCTION_REGISTRY.clear()

    def tearDown(self):
        function_kernel._FUNCTION_REGISTRY.clear()

    def test_apply_function_kernel_replaces_target(self):
        module_name = "tests.kernel._tmp_module"
        temp_module = types.ModuleType(module_name)

        def original(x):
            return x + 1

        def replacement(x):
            return x + 10

        temp_module.target = original
        sys.modules[module_name] = temp_module

        function_kernel.register_function_kernel(
            func_name="target",
            target_module=module_name,
            func_impl=replacement,
        )

        applied = function_kernel.apply_function_kernel(target_module=module_name)

        self.assertIs(temp_module.target, replacement)
        self.assertEqual(applied, [f"{module_name}.target"])
        sys.modules.pop(module_name, None)

    def test_register_function_kernel_requires_single_source(self):
        with self.assertRaises(ValueError):
            function_kernel.register_function_kernel(
                func_name="f",
                target_module="m",
            )

        with self.assertRaises(ValueError):
            function_kernel.register_function_kernel(
                func_name="f",
                target_module="m",
                func_impl=lambda x: x,
                repo_id="repo",
            )

    def test_register_function_kernel_revision_version_exclusive(self):
        with self.assertRaises(ValueError):
            function_kernel.register_function_kernel(
                func_name="f",
                target_module="m",
                func_impl=lambda x: x,
                revision="r1",
                version="v1",
            )

    def test_register_function_kernel_appends_spec(self):
        function_kernel.register_function_kernel(
            func_name="f",
            target_module="m",
            func_impl=lambda x: x,
            device="cpu",
        )
        self.assertEqual(len(function_kernel._FUNCTION_REGISTRY), 1)
        spec = function_kernel._FUNCTION_REGISTRY[0]
        self.assertEqual(spec.func_name, "f")
        self.assertEqual(spec.target_module, "m")
        self.assertEqual(spec.device, "cpu")

    def test_load_from_hub_with_repo(self):
        class DummyModule:
            def __init__(self):
                self.calls = []

            def __call__(self, x):
                self.calls.append(x)
                return x + 2

        class DummyRepo:
            def load(self):
                return DummyModule

        impl = function_kernel._load_from_hub(
            repo=DummyRepo(),
            repo_id=None,
            revision=None,
            version=None,
            func_name="unused",
        )
        self.assertEqual(impl(3), 5)

    def test_load_from_hub_with_repo_id(self):
        original_select = function_kernel.select_revision_or_version
        original_get_kernel = function_kernel.get_kernel

        def fake_select(repo_id, revision, version):
            self.assertEqual(repo_id, "repo")
            self.assertEqual(revision, "rev")
            self.assertIsNone(version)
            return "resolved"

        class DummyKernel:
            def target(self, x):
                return x + 4

        def fake_get_kernel(repo_id, revision):
            self.assertEqual(repo_id, "repo")
            self.assertEqual(revision, "resolved")
            return DummyKernel()

        function_kernel.select_revision_or_version = fake_select
        function_kernel.get_kernel = fake_get_kernel
        try:
            impl = function_kernel._load_from_hub(
                repo=None,
                repo_id="repo",
                revision="rev",
                version=None,
                func_name="target",
            )
            self.assertEqual(impl(1), 5)
        finally:
            function_kernel.select_revision_or_version = original_select
            function_kernel.get_kernel = original_get_kernel

    def test_apply_function_kernel_uses_repo_id_impl(self):
        module_name = "tests.kernel._tmp_module_repo"
        temp_module = types.ModuleType(module_name)
        temp_module.target = lambda x: x
        sys.modules[module_name] = temp_module

        original_select = function_kernel.select_revision_or_version
        original_get_kernel = function_kernel.get_kernel

        def fake_select(repo_id, revision, version):
            return "resolved"

        class DummyKernel:
            def target(self, x):
                return x + 7

        def fake_get_kernel(repo_id, revision):
            return DummyKernel()

        function_kernel.select_revision_or_version = fake_select
        function_kernel.get_kernel = fake_get_kernel
        try:
            function_kernel.register_function_kernel(
                func_name="target",
                target_module=module_name,
                repo_id="repo",
                revision="rev",
            )
            applied = function_kernel.apply_function_kernel(target_module=module_name)
            self.assertEqual(temp_module.target(1), 8)
            self.assertEqual(applied, [f"{module_name}.target"])
        finally:
            function_kernel.select_revision_or_version = original_select
            function_kernel.get_kernel = original_get_kernel
            sys.modules.pop(module_name, None)

    def test_apply_function_kernel_device_filter(self):
        module_name = "tests.kernel._tmp_module_device"
        temp_module = types.ModuleType(module_name)
        temp_module.target = lambda x: x
        sys.modules[module_name] = temp_module

        function_kernel.register_function_kernel(
            func_name="target",
            target_module=module_name,
            func_impl=lambda x: x + 1,
            device="cuda",
        )

        applied = function_kernel.apply_function_kernel(
            target_module=module_name,
            device="cpu",
        )
        self.assertEqual(applied, [])
        sys.modules.pop(module_name, None)


if __name__ == "__main__":
    unittest.main()
