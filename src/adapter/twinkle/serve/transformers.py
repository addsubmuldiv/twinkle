import uuid
from typing import Dict, Any, Union, Type, List

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from ray import serve

import twinkle
from adapter.twinkle.serve.deployment import Deployment
from adapter.twinkle.serve.validation import is_token_valid
from twinkle import DeviceGroup, DeviceMesh
from twinkle.data_format import InputFeature, Trajectory
from twinkle.loss import Loss
from twinkle.model import TransformersModel
from twinkle.model.base import TwinkleModel


def build_model_app(model_id: str,
                    device_group: Dict[str, Any],
                    device_mesh: Dict[str, Any],
                    **kwargs):
    app = FastAPI()
    device_group = DeviceGroup(**device_group)
    twinkle.initialize(mode='ray', groups=[device_group], lazy_collect=False)

    device_mesh = DeviceMesh(**device_mesh)

    resource_prefix = 'model_' + model_id.replace('/', '_') + '_'
    resource_uuid = resource_prefix + str(uuid.uuid4().hex)

    @app.middleware("http")
    async def verify_token(request: Request, call_next):
        authorization = request.headers.get("Authorization")
        if not authorization:
            return JSONResponse(status_code=401, content={"detail": "Missing token"})

        token = authorization[7:] if authorization.startswith("Bearer ") else authorization
        if not is_token_valid(token):
            return JSONResponse(status_code=403, content={"detail": "Invalid token"})

        request.state.token = token
        response = await call_next(request)
        return response

    @serve.deployment(name="ModelManagement")
    @serve.ingress(app)
    class ModelManagement(TwinkleModel, Deployment):
        def __init__(self):
            self.init_registry()
            model = TransformersModel(model_id=model_id, device_mesh=device_mesh, **kwargs)
            self._registry.add_config(resource_uuid, model)

        @app.post("/forward")
        def forward(self, *, inputs: Union[InputFeature, List[InputFeature], List[Trajectory]], **kwargs):
            return self.model.forward(inputs=inputs, **kwargs)

        @app.post("/forward_only")
        def forward_only(self, *, inputs: Union[InputFeature, List[InputFeature], List[Trajectory]], **kwargs):
            return self.model.forward_only(inputs=inputs, **kwargs)

        @app.post("/calculate_loss")
        def calculate_loss(self, **kwargs):
            return self.model.calculate_loss(**kwargs)

        @app.post("/backward")
        def backward(self, **kwargs):
            return self.model.backward(**kwargs)

        @app.post("/forward_backward")
        def forward_backward(self, *, inputs: Union[InputFeature, List[InputFeature], List[Trajectory]], **kwargs):
            return self.model.forward_backward(inputs=inputs, **kwargs)

        @app.post("/step")
        def step(self, **kwargs):
            return self.model.step(**kwargs)

        @app.post("/zero_grad")
        def zero_grad(self, **kwargs):
            return self.model.zero_grad(**kwargs)

        @app.post("/lr_step")
        def lr_step(self, **kwargs):
            return self.model.lr_step(**kwargs)

        @app.post("/set_loss")
        def set_loss(self, loss_cls: Union[Type[Loss], str], **kwargs):
            return self.model.set_loss(loss_cls, **kwargs)

        @app.post("/set_optimizer")
        def set_optimizer(self, optimizer_cls: str, **kwargs):
            return self.model.set_optimizer(optimizer_cls, **kwargs)

        @app.post("/set_lr_scheduler")
        def set_lr_scheduler(self, scheduler_cls: str, **kwargs):
            return self.model.set_lr_scheduler(scheduler_cls, **kwargs)

        @app.post("/save")
        def save(self, output_dir: str, **kwargs):
            return self.model.save(output_dir, **kwargs)

        @app.post("/add_adapter")
        def add_adapter_to_model(self, adapter_name: str, config: Dict[str, Any]):
            return self.model.add_adapter_to_model(adapter_name, config)

        @app.post("/set_template")
        def set_template(self, template_cls: str, **kwargs):
            return self.model.set_template(template_cls, **kwargs)

        @app.post("/set_processor")
        def set_processor(self, processor_cls: str, **kwargs):
            return self.model.set_processor(processor_cls, **kwargs)

    return ModelManagement.bind()