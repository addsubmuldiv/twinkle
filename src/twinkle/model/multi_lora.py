import re
from dataclasses import dataclass, field
from types import MethodType
from typing import Optional, List, Dict

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from peft.tuners.lora import LoraLayer, Linear, Embedding

from twinkle import Platform
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

    def _get_available_lora(self) -> Optional[LoraTenant]:
        for _lora in self.loras:
            if _lora.tenant_adapter_name is None:
                return _lora
        return None

    def acquire_lora(self, tenant_adapter_name: str, config: LoraConfig) -> LoraTenant:
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

    def _patch_lora_forward(self, base_layer: LoraLayer):

        if isinstance(base_layer, Linear):
            def _linear_forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
                self._check_forward_args(x, *args, **kwargs)
                VARIANT_KWARG_KEYS = ["alora_offsets"]
                adapter_names = kwargs.pop("adapter_names", None)
                variant_kwargs = {k: kwargs.pop(k, None) for k in VARIANT_KWARG_KEYS}  # don't pass these to base_layer

                if self.disable_adapters:
                    if self.merged:
                        self.unmerge()
                    result = self.base_layer(x, *args, **kwargs)
                elif adapter_names is not None:
                    result = self._mixed_batch_forward(x, *args, adapter_names=adapter_names, **variant_kwargs,
                                                       **kwargs)
                elif self.merged:
                    result = self.base_layer(x, *args, **kwargs)
                else:
                    result = self.base_layer(x, *args, **kwargs)
                    torch_result_dtype = result.dtype

                    lora_A_keys = self.lora_A.keys()
                    for active_adapter in self.active_adapters:
                        if active_adapter not in lora_A_keys:
                            continue

                        lora_A = self.lora_A[active_adapter]
                        lora_B = self.lora_B[active_adapter]

                        dropout = self.lora_dropout[active_adapter]
                        scaling = self.scaling[active_adapter]

                        x = self._cast_input_dtype(x, lora_A.weight.dtype)
                        if active_adapter not in self.lora_variant:
                            if tenant_rank:
                                dropout_x = dropout(x)
                                lora_A_out = torch.nn.functional.linear(dropout_x, lora_A.weight[:tenant_rank, :], bias=None)
                                lora_B_out = torch.nn.functional.linear(lora_A_out, lora_B.weight[:, :tenant_rank], bias=None)
                                result = result + lora_B_out * scaling
                            else:
                                result = result + lora_B(lora_A(dropout(x))) * scaling
                        else:
                            raise NotImplementedError

                    result = result.to(torch_result_dtype)

                return result

            base_layer.forward = MethodType(_linear_forward, base_layer)
        elif isinstance(base_layer, Embedding):

            def _embedding_forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
                self._check_forward_args(x, *args, **kwargs)
                adapter_names = kwargs.pop("adapter_names", None)

                if self.disable_adapters:
                    if self.merged:
                        self.unmerge()
                    result = self.base_layer(x, *args, **kwargs)
                elif adapter_names is not None:
                    result = self._mixed_batch_forward(x, *args, adapter_names=adapter_names, **kwargs)
                elif self.merged:
                    result = self.base_layer(x, *args, **kwargs)
                else:
                    result = self.base_layer(x, *args, **kwargs)
                    torch_result_dtype = result.dtype
                    for active_adapter in self.active_adapters:
                        if active_adapter not in self.lora_embedding_A:
                            continue

                        if active_adapter not in self.lora_variant:
                            embedding_A = self.lora_embedding_A[active_adapter].T
                            embedding_B = self.lora_embedding_B[active_adapter].T
                            scaling = self.scaling[active_adapter]
                            if tenant_rank:
                                embedding_A = embedding_A[:, :tenant_rank]
                                embedding_B = embedding_B[:tenant_rank, :]
                            after_A = self._embed(x, embedding_A)
                            result = result + (after_A @ embedding_B) * scaling
                        else:
                            raise NotImplementedError
                    result = result.to(torch_result_dtype)

                return result

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
