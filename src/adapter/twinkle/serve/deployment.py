from typing import Any, Tuple
import ray


class Deployment:

    _registry: 'WorkerRegistry' = None

    @staticmethod
    def init_registry():
        if Deployment._registry is not None:
            return

        @ray.remote
        class WorkerRegistry:

            def __init__(self):
                self.config = {}

            def add_config(self, key: str, value: Any):
                self.config[key] = value

            def add_or_get(self, key: str, value: Any) -> Tuple[bool, Any]:
                if key in self.config:
                    return self.config[key]
                self.config[key] = value
                return value

            def get_config(self, key: str):
                return self.config.get(key)

            def clear(self):
                self.config.clear()

        try:
            Deployment._registry = ray.get_actor('twinkle_resource_registry')
        except ValueError:
            try:
                Deployment._registry = WorkerRegistry.options(
                    name='twinkle_resource_registry',
                    lifetime='detached',
                ).remote()
            except ValueError:
                Deployment._registry = ray.get_actor('twinkle_resource_registry')
        assert Deployment._registry is not None

