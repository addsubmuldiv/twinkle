from typing import Union, Callable, Any, List, Optional, Literal
from ..utils import torch as torch_util, framework as framework_util
from ..utils import exists


def apply_kernel(module: Any,
                 mode: Literal['train', 'inference', 'compile', None] = 'train',
                 kernel: Optional[Union[str, Callable[[*Any], Any]]]=None,
                 target_modules: Union[str, List[str]]=None,
                 device: Optional[Union[str, Any]] = None,
                ) -> Any:
    if framework_util.get_framework(module) == 'torch':
        if torch_util.get_library(module) == 'transformers':
            if exists('kernels'):
                from kernels import kernelize, Mode
                kernel_mode = Mode.TRAINING
                if mode == 'inference':
                    kernel_mode = Mode.INFERENCE
                elif mode == 'compile':
                    kernel_mode = Mode.TORCH_COMPILE
                from kernels import kernelize
                return kernelize(module, mode=kernel_mode, device=device)

        assert target_modules is not None and kernel is not None


    else:
        raise NotImplementedError(f'Unsupported applying kernels for: {module.__class__}')


def apply_kernel_torch(module: Any,
                             mode: Literal['train', 'inference', 'compile', None] = 'train',
                             kernel: Optional[Union[str, Callable[[*Any], Any]]]=None,
                             target_modules: Union[str, List[str]]=None,
                             device: Optional[Union[str, Any]] = None,):


