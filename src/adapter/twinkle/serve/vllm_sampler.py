from typing import Dict, Any, List

from fastapi import FastAPI
from ray import serve
import twinkle
from twinkle import DeviceGroup, DeviceMesh
from twinkle.data_format import Trajectory
from twinkle.sampler import VLLMSampler, Sampler


def build_sampler_app(model_id: str,
                      device_group: Dict[str, Any],
                      device_mesh: Dict[str, Any],
                      **kwargs):
    app = FastAPI()
    device_group = DeviceGroup(**device_group)
    twinkle.initialize(mode='ray',
                       groups=[device_group],
                       lazy_collect=False)

    device_mesh = DeviceMesh(**device_mesh)

    @serve.deployment(name="SamplerManagement")
    @serve.ingress(app)
    class SamplerManagement(Sampler):

        def __init__(self):
            self.sampler = VLLMSampler(model_id=model_id, device_mesh=device_mesh, **kwargs)

        def sample(self, trajectories: List[Trajectory], adapter_name = '')-> List[Trajectory]:
            return self.sampler.sample(trajectories, adapter_name)

        def add_adapter_to_sampler(self, adapter_name: str, config):
            return self.sampler.add_adapter_to_sampler(adapter_name, config)

        def sync_weights(self, state_dict: Dict[str, Any], adapter_name=''):
            return self.sampler.sync_weights(state_dict, adapter_name)

    return SamplerManagement.bind()
