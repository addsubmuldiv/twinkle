from typing import Optional, Union
from twinkle import DeviceMesh
from twinkle.template import Template


class InputProcessor:

    def __init__(self, device_mesh: Optional[DeviceMesh] = None, template: Optional[Union[Template, str]] = None):
        self.device_mesh = device_mesh
        self.template = template()

    def __call__(self, inputs):
        inputs = self.dispatch(inputs)
        if self.template is not None:
            inputs = self.template.encode(inputs)
        return self.prepare_inputs(inputs)

    def prepare_inputs(self, inputs):
        ...

    def collate_fn(self, inputs):
        ...
