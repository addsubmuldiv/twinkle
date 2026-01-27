import re
from contextlib import contextmanager
from dataclasses import dataclass, field
from types import MethodType
from typing import Optional, List, Dict

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from peft.tuners.lora import LoraLayer, Linear, Embedding

from twinkle import torch_util
from twinkle.patch.base import Patch


@dataclass
class LoraTenant:

    index: int
    adapter_name: str
    config: LoraConfig
    tenant_adapter_name: Optional[str] = None
    tenant_config: Optional[LoraConfig] = None
    lora_A_weights: Dict[str, torch.Tensor] = field(default_factory=lambda: {})


class MultiLora(Patch):

    def __init__(self, max_loras=5, max_r=32):
        self.max_loras = max_loras
        self.max_r = max_r
        self.loras: List[LoraTenant] = []
        self.module = None
        self._active_adapters = []

    def _get_available_lora(self) -> Optional[LoraTenant]:
        for _lora in self.loras:
            if _lora.tenant_adapter_name is None:
                return _lora
        return None

    @contextmanager
    def active_adapters(self, adapter_names: List[str]):
        self._active_adapters = adapter_names
        yield
        self._active_adapters = []

    def acquire_lora(self, tenant_adapter_name: str, config: LoraConfig) -> LoraTenant:
        if self.has_lora(tenant_adapter_name):
            raise ValueError(f'Lora {tenant_adapter_name} already exists')
        _available_lora = self._get_available_lora()
        if _available_lora is None:
            raise RuntimeError(f"No lora available for tenant {tenant_adapter_name}")
        if config.r > self.max_r:
            raise RuntimeError(f"Too big rank for lora: {config.r}")
        _available_lora.tenant_config = config
        _available_lora.tenant_adapter_name = tenant_adapter_name
        return _available_lora

    def release_lora(self, tenant_adapter_name: str) -> Optional[str]:
        for _lora in self.loras:
            if _lora.tenant_adapter_name == tenant_adapter_name:
                _lora.tenant_config = None
                _lora.tenant_adapter_name = None
                self._load_initial_weights(_lora.adapter_name)
                return _lora.adapter_name
        else:
            raise ValueError(f'No lora found for tenant {tenant_adapter_name}')

    def has_lora(self, adapter_name: str) -> bool:
        return len([_lora for _lora in self.loras if _lora.tenant_adapter_name == adapter_name]) > 0

    def find_lora(self, adapter_name):
        return [_lora for _lora in self.loras if _lora.tenant_adapter_name == adapter_name][0]

    def _find_loras(self, hidden_states, adapter_names: List[str]):
        _loras = []
        _hidden_states = []
        start_idx = 0
        current_adapter = adapter_names[0]

        for i in range(1, len(adapter_names) + 1):
            if i == len(adapter_names) or adapter_names[i] != current_adapter:
                end_idx = i
                _hidden_states.append(hidden_states[start_idx:end_idx])
                _loras.append(self.find_lora(current_adapter))
                if i < len(adapter_names):
                    start_idx = i
                    current_adapter = adapter_names[i]

        return _hidden_states, _loras

    def _patch_lora_forward(_self, base_layer: LoraLayer):

        if isinstance(base_layer, Linear):
            def _linear_forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
                self._check_forward_args(x, *args, **kwargs)
                result = self.base_layer(x, *args, **kwargs)
                torch_result_dtype = result.dtype

                assert x.shape[0] == len(_self._active_adapters)
                hidden_states, loras = self._find_loras(x, _self._active_adapters)
                results = []
                for i, _hidden_state, _lora in enumerate(zip(hidden_states, loras)):
                    result = self.base_layer(_hidden_state, *args, **kwargs)
                    lora_A = self.lora_A[_lora.adapter_name]
                    lora_B = self.lora_B[_lora.adapter_name]

                    dropout = self.lora_dropout[_lora.adapter_name]
                    scaling = _lora.tenant_config.lora_alpha / _lora.tenant_config.r
                    x = self._cast_input_dtype(x, lora_A.weight.dtype)
                    dropout_x = dropout(x)
                    lora_A_out = torch.nn.functional.linear(dropout_x, lora_A.weight[:_lora.tenant_config.r, :], bias=None)
                    lora_B_out = torch.nn.functional.linear(lora_A_out, lora_B.weight[:, :_lora.tenant_config.r], bias=None)
                    result = result + lora_B_out * scaling

                    result = result.to(torch_result_dtype)
                    results.append(result)

                return torch.cat(results, dim=0)

            base_layer.forward = MethodType(_linear_forward, base_layer)
        elif isinstance(base_layer, Embedding):

            def _embedding_forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
                self._check_forward_args(x, *args, **kwargs)
                result = self.base_layer(x, *args, **kwargs)
                torch_result_dtype = result.dtype

                assert x.shape[0] == len(self._active_adapters)
                hidden_states, loras = self._find_loras(x, self._active_adapters)
                results = []
                for _hidden_state, _lora in zip(hidden_states, loras):
                    sub_result = self.base_layer(_hidden_state, *args, **kwargs)

                    embedding_A = self.lora_embedding_A[_lora.adapter_name].T
                    embedding_B = self.lora_embedding_B[_lora.adapter_name].T
                    scaling = _lora.tenant_config.lora_alpha / _lora.tenant_config.r

                    embedding_A = embedding_A[:, :_lora.tenant_config.r]
                    embedding_B = embedding_B[:_lora.tenant_config.r, :]

                    after_A = self._embed(_hidden_state, embedding_A)
                    sub_result = sub_result + (after_A @ embedding_B) * scaling

                    sub_result = sub_result.to(torch_result_dtype)
                    results.append(sub_result)

                return torch.cat(results, dim=0)

            base_layer.forward = MethodType(_embedding_forward, base_layer)

    def patch(self, module: torch.nn.Module, *args, **kwargs):
        self.module = module
        for i in range(self.max_loras):
            config = LoraConfig(
                r=self.max_r,
                target_modules='all-linear',
                lora_alpha=32,
            )
            lora_tenant = LoraTenant(index=i, adapter_name=f'lora_{i}', config=config)
            self.loras.append(lora_tenant)
            if isinstance(module, PeftModel):
                module.add_adapter(lora_tenant.adapter_name, config)
            else:
                module = get_peft_model(module, config, lora_tenant.adapter_name)
        self.module = module
        return module

    def save_initial_weights(self):
        for i in range(self.max_loras):
            lora_tenant = self.loras[i]
            pattern = re.compile(rf'\.lora_(?:A|embedding_A)\.{re.escape(lora_tenant.adapter_name)}\.')
            for name, parameter in self.module.named_parameters():
                if pattern.search(name):
                    lora_tenant.lora_A_weights[name] = parameter.data.clone().to('cpu')
    
    def get_state_dict(self, tenant_adapter_name):
        state_dict = {}
        for i in range(self.max_loras):
            if self.loras[i].tenant_adapter_name == tenant_adapter_name:
                pattern = re.compile(rf'\.lora_\w+\.{re.escape(self.loras[i].adapter_name)}\.')
                for name, parameter in self.module.named_parameters():
                    if pattern.search(name):
                        _param = torch_util.to_local_tensor(parameter)
                        if 'embedding_A' in name:
                            _param = _param[:, :self.loras[i].tenant_config.r]
                        elif 'embedding_B' in name:
                            _param = _param[:self.loras[i].tenant_config.r, :]
                        elif '_A' in name:
                            _param = _param[:self.loras[i].tenant_config.r, :]
                        elif '_B' in name:
                            _param = _param[:, :self.loras[i].tenant_config.r]
                        state_dict[name] = _param
                break
        else:
            raise ValueError(f'Adapter {tenant_adapter_name} not found')
        return state_dict

    def _load_initial_weights(self, origin_adapter_name):
        for i in range(self.max_loras):
            if self.loras[i].adapter_name == origin_adapter_name:
                lora_tenant = self.loras[i]
                pattern_A = re.compile(rf'\.lora_(?:A|embedding_A)\.{origin_adapter_name}\.')
                pattern_B = re.compile(rf'\.lora_(?:B|embedding_B)\.{origin_adapter_name}\.')
                for name, parameter in self.module.named_parameters():
                    if pattern_A.search(name):
                        parameter.data.copy_(lora_tenant.lora_A_weights[name])
                    if pattern_B.search(name):
                        parameter.data.copy_(torch.zeros_like(parameter.data).to(parameter.data.dtype).to('cpu'))
                break
