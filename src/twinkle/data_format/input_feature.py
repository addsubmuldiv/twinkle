from typing import List, Union, Optional, TypedDict
import numpy as np

InputType = Union[List[List[int]], List[int], np.ndarray, 'torch.Tensor']


class InputFeature(TypedDict, total=False):
    input_ids: InputType
    attention_mask: InputType
    position_ids: InputType
    labels: InputType
    completion_mask: InputType
    logits_to_keep: Optional[int]
    num_items_in_batch: Optional[int]


def to_transformers_dict(feature: InputFeature) -> dict:
    import torch
    output = {
        'input_ids': feature.get('input_ids'),
        'attention_mask': feature.get('attention_mask'),
        'position_ids': feature.get('position_ids'),
    }
    for key in list(output.keys()):
        if not isinstance(output[key], torch.Tensor):
            output[key] = np.array(output[key])
    return output


def to_megatron_dict(feature: InputFeature) -> dict:
    raise NotImplementedError()