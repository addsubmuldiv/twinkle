import functools
from typing import Literal, List, Optional
from typing import Type, TypeVar, Tuple, overload

from .device_group import DeviceGroup
from .. import requires

_mode: Optional[Literal['local', 'ray', 'remote']] = 'local'

_device_group: Optional[List[DeviceGroup]] = None

_remote_components: dict = {}


def initialize(mode: Literal['local', 'ray', 'remote'],
               groups: Optional[List[DeviceGroup]] = None,):
    global _mode, _device_group
    assert mode in ('local', 'ray', 'remote')
    _mode = mode
    if _mode == 'ray':
        requires('ray')
    _device_group = groups


def _get_remote_component(component):
    if component not in _remote_components:
        import ray
        _remote_components[component] = ray.remote(component)
    return _remote_components[component]


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

def prepare(*components, group_name: Optional[str]=None):
    _output = []
    for component in components:
        _output.append(prepare_one(component, group_name=group_name))
    return tuple(_output)


def prepare_one(component: Type[T1], group_name: Optional[str]=None) -> Type[T1]:

    class WrappedComponent:

        def __init__(self, *args, **kwargs):
            if _mode == 'local':
                self._actor = component(*args, **kwargs)
            elif _mode == 'ray':
                import ray
                from .ray import RayHelper
                self._actor = RayHelper.create_workers(self._actor, group_name, *args, **kwargs)
            else:
                raise NotImplementedError(f'Unsupported mode {_mode}')

        def __getattr__(self, name):
            attr = getattr(self._actor, name)

            if callable(attr):
                @functools.wraps(attr)
                def wrapper(*args, **kwargs):
                    if _mode == 'local':
                        return attr(*args, **kwargs)
                    elif _mode == 'ray':
                        import ray
                        return ray.get(attr.remote(*args, **kwargs))
                    else:
                        raise NotImplementedError(f'Unsupported mode {_mode}')

                return wrapper
            return attr

        @property
        def actor(self):
            return self._actor

    return WrappedComponent
