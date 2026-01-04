from typing import List, Optional, Dict, Any
from transformers import AutoTokenizer, PreTrainedTokenizer
import numpy as np
from twinkle.hub import HubOperation
from twinkle.data_format import Trajectory, InputFeature


class Template:

    def __init__(self, model_id: str, **kwargs):
        model_id = HubOperation.download_model(model_id)
        self.tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(model_id)

    def encode(self, trajectory: Trajectory) -> InputFeature:
        conversation = [dict(message) for message in trajectory['messages']]
        tools = [dict(tool) for tool in trajectory['tools']]
        templated = self.tokenizer.apply_chat_template(conversation=conversation, tools=tools)
        encoded = self.tokenizer(templated, return_tensors="np")
        return InputFeature(
            input_ids=encoded['input_ids'],
            attention_mask=np.ones_like(encoded['input_ids']),
            position_ids=np.arange(0, len(encoded['input_ids'])),
        )

    @staticmethod
    def map_col_to_row(trajectories: Dict[str, Any]):
        rows = []
        total_count = trajectories[next(iter(trajectories.keys()))]
        for i in range(total_count):
            row = {}
            for key in trajectories:
                row[key] = trajectories[key][i]
            rows.append(row)
        return rows


    def batch_encode(self, trajectories: Dict[str, Any]) -> List[InputFeature]:
        output = []
        trajectories = self.map_col_to_row(trajectories)
        for trajectory in trajectories:
            output.append(self.encode(trajectory))
        return output

    def check(self, trajectory: Trajectory) -> Optional[Trajectory]:
        encoded = None
        try:
            encoded = self.encode(trajectory)
            if not encoded:
                return None
            else:
                return trajectory
        except Exception as e: # noqa
            return None
        finally:
            if encoded:
                del encoded

    def batch_check(self, trajectories: List[Trajectory]) -> List[Optional[Trajectory]]:
        output = []
        for trajectory in trajectories:
            output.append(self.check(trajectory))
        return output

    def decode(self, token_ids: List[int], **kwargs) -> str:
        return self.tokenizer.decode(token_ids, **kwargs)

    def batch_decode(self, token_ids: List[List[int]], **kwargs) -> List[str]:
        return [self.tokenizer.decode(_ids, **kwargs) for _ids in token_ids]