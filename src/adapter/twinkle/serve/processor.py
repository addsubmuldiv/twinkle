from typing import Dict, Any

from fastapi import FastAPI
from ray import serve
import twinkle
from twinkle import DeviceGroup, DeviceMesh
from twinkle.model import TransformersModel


def build_processor_app(model_id: str,
                      device_group: Dict[str, Any],
                      device_mesh: Dict[str, Any],
                      **kwargs):
    app = FastAPI()
    device_group = DeviceGroup(**device_group)
    twinkle.initialize(mode='ray',
                       groups=[device_group],
                       lazy_collect=False)

    device_mesh = DeviceMesh(**device_mesh)

    @serve.deployment(name="ProcessorManagement")
    @serve.ingress(app)
    class ProcessorManagement:
        def __init__(self):
            self.sampler = TransformersModel(model_id=model_id, device_mesh=device_mesh, **kwargs)

    return ProcessorManagement.bind()
