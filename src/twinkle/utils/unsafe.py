import os
from collections.abc import Mapping
from typing import Iterable, Callable


def any_callable(args):
    if isinstance(args, Mapping):
        return any(any_callable(arg) for arg in args.values())
    elif isinstance(args, Iterable):
        return any(any_callable(arg) for arg in args)
    elif isinstance(args, Callable):
        return True
    elif isinstance(args, type):
        return True
    else:
        return False


def check_unsafe(*args, **kwargs):
    if os.environ.get("TRUST_REMOTE_CODE", "0") == "0":
        if any_callable(*args) or any_callable(**kwargs):
            raise ValueError(
                "Twinkle does not support Callable or Type inputs."
            )