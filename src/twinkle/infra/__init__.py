from typing import Literal, List, Optional
from typing import Type, TypeVar, Tuple, overload

from .device_group import DeviceGroup

_mode: Optional[Literal['local', 'ray', 'remote']] = None

_library: Optional[Literal['transformers', 'megatron', 'other']] = None

_device_group: Optional[List[DeviceGroup]] = None

_inited = False


def initialize(mode: Literal['local', 'ray', 'remote'],
               library: Literal['transformers', 'megatron', 'other'],
               groups: Optional[List[DeviceGroup]] = None,):
    global _mode, _library, _device_group, _inited
    assert mode in ('local', 'ray', 'remote')
    assert library in ('transformers', 'megatron', 'other')
    _mode = mode
    _library = library
    _device_group = groups
    _inited = True


T1 = TypeVar('T1', bound=object)
T2 = TypeVar('T2', bound=object)
T3 = TypeVar('T3', bound=object)
T4 = TypeVar('T4', bound=object)
T5 = TypeVar('T5', bound=object)

@overload
def prepare(__c1: Type[T1], /) -> Tuple[Type[T1]]: ...

@overload
def prepare(__c1: Type[T1], __c2: Type[T2], /) -> Tuple[Type[T1], Type[T2]]: ...

@overload
def prepare(__c1: Type[T1], __c2: Type[T2], __c3: Type[T3], /) -> Tuple[Type[T1], Type[T2], Type[T3]]: ...

@overload
def prepare(__c1: Type[T1], __c2: Type[T2], __c3: Type[T3], __c4: Type[T4], /) -> Tuple[Type[T1], Type[T2], Type[T3], Type[T4]]: ...

@overload
def prepare(__c1: Type[T1], __c2: Type[T2], __c3: Type[T3], __c4: Type[T4], __c5: Type[T5], /) -> Tuple[Type[T1], Type[T2], Type[T3], Type[T4], Type[T5]]: ...

def prepare(*components):
    if not _inited:
        raise AssertionError("initialize() must be called before prepare()")
    _output = []
    for component in components:
        _output.append(prepare_one(component))
    return _output


def prepare_one(component: Type[T1]) -> Type[T1]:
    if not _inited:
        raise AssertionError("initialize() must be called before prepare()")
    if _mode == 'local':
        return component
    elif _mode == 'ray':
        from .ray import RayHelper
        return RayHelper.wrap(component)
    elif _mode == 'remote':
        raise ValueError(f'Remote mode is not supported with twinkle, use `twinkle-client instead.`')
    else:
        raise ValueError(f'Unknown mode "{_mode}"')