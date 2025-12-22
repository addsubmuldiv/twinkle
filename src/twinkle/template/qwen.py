# Copyright (c) Alibaba, Inc. and its affiliates.
import os
from contextlib import contextmanager
from dataclasses import dataclass, field
from functools import partial
from typing import Any, Dict, List, Literal, Optional

import torch
import torch.nn.functional as F
import transformers
from packaging import version
from PIL import Image
from torch import nn
from transformers.integrations import is_deepspeed_zero3_enabled

from swift.llm import get_packed_seq_params, to_float_dtype
from swift.utils import get_env_args, is_deepspeed_enabled
from twinkle.template.base import Template
from twinkle.template.constant import LLMTemplateType, MLLMTemplateType
from twinkle.template.register import register_template
from twinkle.template.template_inputs import StdTemplateInputs
from twinkle.template.template_meta import TemplateMeta
from twinkle.template.utils import Context, Word, findall
from twinkle.template.vision_utils import load_audio, load_batch, load_video_ovis2, load_video_ovis2_5
from .llama import Llama3TemplateMeta
from .utils import DEFAULT_SYSTEM, ChatmlTemplateMeta, ThinkingTemplate


@dataclass
class QwenTemplateMeta(ChatmlTemplateMeta):
    default_system: Optional[str] = DEFAULT_SYSTEM
    auto_add_bos: bool = False
    stop_words: List[Word] = field(default_factory=lambda: ['<|endoftext|>'])
    agent_template: str = 'hermes'


class Qwen3Template(ThinkingTemplate):
    no_think_prefix = '<think>\n\n</think>\n\n'


register_template(QwenTemplateMeta(LLMTemplateType.qwen3, default_system=None, template_cls=Qwen3Template))


class Qwen3VLTemplate(Qwen2VLTemplate):
    version = 'v3'

    def _encode(self, inputs: StdTemplateInputs) -> Dict[str, Any]:
        encoded = Template._encode(self, inputs)
        processor = self.processor
        input_ids = encoded['input_ids']
        labels = encoded['labels']
        loss_scale = encoded.get('loss_scale', None)
        for media_type in ['images', 'videos']:
            mm_data = getattr(inputs, media_type)
            if mm_data:
                if media_type == 'images':
                    media_token = self.image_token_id
                    media_inputs = processor.image_processor(images=mm_data, return_tensors='pt', do_resize=False)
                    media_grid_thw = media_inputs['image_grid_thw']
                else:
                    split_token = self._tokenize('\n')[0]
                    media_inputs = processor(
                        text=['\n'.join(['<|vision_start|><|video_pad|><|vision_end|>'] * len(mm_data))],
                        videos=mm_data,
                        return_tensors='pt',
                        do_resize=False,
                        **inputs.mm_processor_kwargs)
                    splited_tokens = self._split_list(media_inputs['input_ids'][0].tolist(), split_token)
                    media_grid_thw = media_inputs['video_grid_thw']
                    media_inputs.pop('input_ids', None)
                    media_inputs.pop('attention_mask', None)
                    media_token = self.video_token_id
                idx_list = findall(input_ids, media_token)
                merge_length = processor.image_processor.merge_size**2

                def _get_new_tokens(i):
                    if media_type == 'images':
                        token_len = (media_grid_thw[i].prod() // merge_length)
                        return [media_token] * token_len
                    else:
                        return splited_tokens[i]

                input_ids, labels, loss_scale = self._extend_tokens(input_ids, labels, loss_scale, idx_list,
                                                                    _get_new_tokens)
                encoded.update(media_inputs)

        encoded['input_ids'] = input_ids
        encoded['labels'] = labels
        encoded['loss_scale'] = loss_scale
        return encoded

    def _post_encode(self, model, inputs: Dict[str, Any]) -> Dict[str, Any]:
        return inputs


register_template(QwenTemplateMeta(MLLMTemplateType.qwen3_vl, template_cls=Qwen3VLTemplate, default_system=None))

