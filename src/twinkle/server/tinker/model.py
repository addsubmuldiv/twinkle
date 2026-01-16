# Copyright (c) ModelScope Contributors. All rights reserved.
from datetime import datetime
import os
import threading
import time
from typing import Dict, Any, Optional
import uuid

from fastapi import FastAPI, Request
from peft import LoraConfig
from pydantic import BaseModel
from ray import serve
from tinker import types

import twinkle
from twinkle import DeviceGroup, DeviceMesh
from twinkle.model import TwinkleModel, MultiLoraTransformersModel
from twinkle.model.base import TwinkleModel
from twinkle.data_format import InputFeature, Trajectory
from twinkle.server.twinkle.validation import verify_request_token, init_config_registry, ConfigRegistryProxy
from .state import get_server_state

def build_model_app(model_id: str,
                    nproc_per_node: int,
                    device_group: Dict[str, Any],
                    device_mesh: Dict[str, Any],
                    deploy_options: Dict[str, Any],
                    **kwargs):
    app = FastAPI()

    @app.middleware("http")
    async def verify_token(request: Request, call_next):
        return await verify_request_token(request=request, call_next=call_next)

    @serve.deployment(name="ModelManagement")
    @serve.ingress(app)
    class ModelManagement():

        COUNT_DOWN = 60 * 30

        def __init__(self, model_id: str, nproc_per_node: int, device_group: Dict[str, Any], device_mesh: Dict[str, Any], **kwargs):
            self.device_group = DeviceGroup(**device_group)
            twinkle.initialize(mode='ray', nproc_per_node=nproc_per_node, groups=[self.device_group], lazy_collect=False)
            self.device_mesh = DeviceMesh(**device_mesh)
            self.model_id = model_id
            self.model: TwinkleModel = None
            self.kwargs = kwargs
            self.adapter_records: Dict[str, int] = {}
            self.hb_thread = threading.Thread(target=self.countdown, daemon=True)
            self.hb_thread.start()
            self.adapter_lock = threading.Lock()
            self.config_registry: ConfigRegistryProxy = init_config_registry()
            self.state = get_server_state()
            self.per_token_model_limit = int(os.environ.get("TWINKLE_PER_USER_MODEL_LIMIT", 3))
            self.key_token_dict = {}

        def countdown(self):
            while True:
                time.sleep(1)
                for key in list(self.adapter_records.keys()):
                    self.adapter_records[key] += 1
                    if self.adapter_records[key] > self.COUNT_DOWN:
                        with self.adapter_lock:
                            self.model.remove_adapter(key)
                        self.adapter_records.pop(key, None)
                        token = self.key_token_dict.pop(key, None)
                        if token:
                            self.handle_adapter_count(token, False)

        def handle_adapter_count(self, token: str, add: bool):
            user_key = token + '_' + 'model_adapter'
            cur_count = self.config_registry.get_config(user_key) or 0
            if add:
                if cur_count < self.per_token_model_limit:
                    self.config_registry.add_config(user_key, cur_count + 1)
                else:
                    raise RuntimeError(f'Model adapter count limitation reached: {self.per_token_model_limit}')
            else:
                if cur_count > 0:
                    cur_count -= 1
                    self.config_registry.add_config(user_key, cur_count)
                if cur_count <= 0:
                    self.config_registry.pop(user_key)
        
        @staticmethod
        def _default_forward_output() -> types.ForwardBackwardOutput:
            tensor = types.TensorData(data=[0.0], dtype="float32", shape=[1])
            loss_output: types.LossFnOutput = {"loss": tensor, "logprobs": tensor}
            return types.ForwardBackwardOutput(
                loss_fn_output_type="TensorData",
                loss_fn_outputs=[loss_output],
                metrics={"loss:avg": 0.0, "tokens_per_second:avg": 0.0},
            )

        @staticmethod
        def _new_request_id() -> str:
            return f"req_{uuid.uuid4().hex}"

        async def _store_future(self, payload: Any, model_id: Optional[str] = None) -> types.UntypedAPIFuture:
            request_id = self._new_request_id()
            await self.state.store_future(request_id, payload, model_id)
            return types.UntypedAPIFuture(request_id=request_id, model_id=model_id)

        @app.post("/create_model")
        async def create_model(self, request: types.CreateModelRequest) -> types.UntypedAPIFuture:
            model_id = await self.state.register_model(request)
            result = types.CreateModelResponse(model_id=model_id)
            model = MultiLoraTransformersModel(
                model_id=self.model_id,
                device_mesh=self.device_mesh,
                remote_group=self.device_group.name,
                **self.kwargs
            )
            return await self._store_future(result, model_id=model_id)

        @app.post("/get_info")
        async def get_info(self, request: types.GetInfoRequest) -> types.GetInfoResponse:
            metadata = await self.state.get_model_metadata(str(request.model_id))
            model_name = metadata.get("base_model") if metadata else str(request.model_id)
            lora_rank = None
            is_lora = False
            if metadata and metadata.get("lora_config"):
                lora_rank = metadata["lora_config"].get("rank")
                is_lora = True
            return types.GetInfoResponse(
                model_data=types.ModelData(model_name=model_name),
                model_id=request.model_id,
                is_lora=is_lora,
                lora_rank=lora_rank,
            )

        @app.post("/unload_model")
        async def unload_model(self, request: types.UnloadModelRequest) -> types.UntypedAPIFuture:
            await self.state.unload_model(request.model_id)
            result = types.UnloadModelResponse(model_id=request.model_id)
            return await self._store_future(result, model_id=request.model_id)

        @app.post("/forward")
        async def forward(self, request: types.ForwardRequest) -> types.UntypedAPIFuture:
            return await self._store_future(self._default_forward_output(), model_id=request.model_id)

        @app.post("/forward_backward")
        async def forward_backward(self, request: types.ForwardBackwardRequest) -> types.UntypedAPIFuture:
            return await self._store_future(self._default_forward_output(), model_id=request.model_id)

        @app.post("/optim_step")
        async def optim_step(self, request: types.OptimStepRequest) -> types.UntypedAPIFuture:
            metrics = types.OptimStepResponse(metrics={
                "grad_norm": 0.0,
                "weight_norm": 0.0,
                "update_norm": 0.0,
            })
            return await self._store_future(metrics, model_id=request.model_id)

        @app.post("/save_weights")
        async def save_weights(self, request: types.SaveWeightsRequest) -> types.UntypedAPIFuture:
            suffix = request.path or f"checkpoint-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
            path = f"tinker://{request.model_id}/{suffix}"
            result = types.SaveWeightsResponse(path=path)
            return await self._store_future(result, model_id=request.model_id)

        @app.post("/load_weights")
        async def load_weights(self, request: types.LoadWeightsRequest) -> types.UntypedAPIFuture:
            result = types.LoadWeightsResponse(path=request.path)
            return await self._store_future(result, model_id=request.model_id)

    return ModelManagement.options(**deploy_options).bind(model_id, nproc_per_node, device_group, device_mesh, **kwargs)