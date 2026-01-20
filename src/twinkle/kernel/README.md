```python

from twinkle.kernel.function import register_function_kernel

# 1) 直接传实现
def fast_add(x, y):
    return x + y + 1  # 示例

register_function_kernel(
    func_name="add",
    target_module="my_pkg.math_ops",
    func_impl=fast_add,
    device="cuda",
)



```

```python
# 2) 传 repo 对象（实现 FuncRepositoryProtocol）
class MyFuncRepo:
    def load(self):
        return MyKernelFunc  # 返回可调用类

class MyKernelFunc:
    def __call__(self, x, y):
        return x + y + 2

register_function_kernel(
    func_name="add",
    target_module="my_pkg.math_ops",
    repo=MyFuncRepo(),
    device="cuda",
)
```


```python
# 3) 通过 Hub repo_id
register_function_kernel(
    func_name="add",
    target_module="my_pkg.math_ops",
    repo_id="kernels-community/activation",
    revision="main",  # 或 version="0.1.0"，两者只能选一个
    device="cuda",
)
```

```python
from twinkle.kernel.function import apply_function_kernel

apply_function_kernel(target_module="my_pkg.math_ops", device="cuda")
```



```python
config = {
    "layers": {
        "linear": {
            "repo_id": "kernels-community/linear",
            "layer_name": "Linear",
            "version": "0.1.0",
            "device": "cuda",
            "mode": "inference",
        },
        "conv2d": {
            "repo_path": "/path/to/local/repo",
            "package_name": "my_kernels",
            "layer_name": "Conv2d",
            "device": "cuda",
        },
    },
    "functions": {
        "add": {
            "target_module": "my_pkg.math_ops",
            "func_impl": lambda x, y: x + y + 1,
            "device": "cuda",
        },
        "relu": {
            "target_module": "my_pkg.activations",
            "repo_id": "kernels-community/activation",
            "revision": "main",
            "device": "cuda",
        },
    },
}

register_kernels(config)

config = {
    "functions": [
        {
            "func_name": "add",
            "target_module": "my_pkg.math_ops",
            "func_impl": lambda x, y: x + y + 1,
            "device": "cuda",
        }
    ]
}

```