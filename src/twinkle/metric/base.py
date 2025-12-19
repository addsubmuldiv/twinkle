from typing import Dict, Any


class Metric:

    def __call__(self, *args, **kwargs) -> Dict[str, Any]:
        ...