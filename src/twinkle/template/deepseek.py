# Copyright (c) Alibaba, Inc. and its affiliates.
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageOps
from transformers.dynamic_module_utils import get_class_from_dynamic_module

from swift.utils import get_env_args
from twinkle.template.base import Template
from twinkle.template.constant import LLMTemplateType, MLLMTemplateType
from twinkle.template.register import TemplateMeta, register_template
from twinkle.template.template_inputs import StdTemplateInputs
from twinkle.template.utils import Prompt, findall
from .utils import ThinkingTemplate


@dataclass
class DeepseekTemplateMeta(TemplateMeta):
    prefix: Prompt = field(default_factory=lambda: [['bos_token_id']])
    prompt: Prompt = field(default_factory=lambda: ['User: {{QUERY}}\n\nAssistant:'])
    chat_sep: Optional[Prompt] = field(default_factory=lambda: [['eos_token_id']])
    suffix: Prompt = field(default_factory=lambda: [['eos_token_id']])
    system_prefix: Optional[Prompt] = field(default_factory=lambda: [['bos_token_id'], '{{SYSTEM}}\n\n'])


class DeepseekV3_1Template(ThinkingTemplate):
    no_think_prefix = '</think>'
    history_think_prefix = '</think>'
    add_no_think_prefix_after_tool = False


register_template(
    DeepseekV2_5TemplateMeta(LLMTemplateType.deepseek_r1, template_cls=ThinkingTemplate, response_prefix='<think>\n'))

# enable thinking: response_prefix='<think>'
register_template(
    DeepseekV2_5TemplateMeta(
        LLMTemplateType.deepseek_v3_1,
        template_cls=DeepseekV3_1Template,
        response_prefix='</think>',
        agent_template='deepseek_v3_1'))
