from typing import Union, List, Optional

from twinkle import Platform, DeviceMesh
from twinkle.data_format import InputFeature, to_transformers_dict


class InputProcessor:

    def __init__(self, device_mesh: Optional[DeviceMesh] = None, **kwargs):
        self.device_mesh = device_mesh

    def __call__(self, inputs: Union[InputFeature, List[InputFeature]]):
        if isinstance(inputs, list):
            inputs = self.collate_fn(inputs)
        return self.prepare_inputs(inputs)

    def prepare_inputs(self, inputs: InputFeature) -> InputFeature:
        import torch
        for key in list(inputs.keys()):
            if isinstance(inputs[key], torch.Tensor):
                inputs[key] = inputs[key].to(Platform.get_local_device())
        return inputs

    def collate_fn(self, inputs: List[InputFeature]) -> InputFeature:
        from torch.utils.data import default_collate
        batch_encoded = default_collate([to_transformers_dict(_input) for _input in inputs])
        return InputFeature(**batch_encoded)
