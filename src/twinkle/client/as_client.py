import functools
from typing import Callable, TypeVar

T = TypeVar('T')


def as_client():

    def decorator(func: Callable[..., T]) -> Callable[..., T]:

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        return wrapper

    return decorator