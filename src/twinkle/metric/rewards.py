# Copyright (c) ModelScope Contributors. All rights reserved.
from typing import List, Union

from .base import Metric
from ..data_format import InputFeature, ModelOutput


class Rewards(Metric):

    def __init__(self, device_mesh, process_group, **kwargs):
        super().__init__(device_mesh, process_group, **kwargs)
        self.total_correct = 0
        self.total_count = 0
        self.generate_time: float = 0.0
        self.weight_sync_time: float = 0.0
        self.rewards: List[float] = []
        self.format_rewards: List[float] = []
        self.accuracy_rewards: List[float] = []
        self.completion_lengths: List[int] = []

    def reset(self):
        self.generate_time = 0.0
        self.weight_sync_time = 0.0
        self.rewards = []
        self.format_rewards = []
        self.accuracy_rewards = []
        self.completion_lengths = []

    def accumulate(self, inputs: Union[InputFeature, List[InputFeature]], outputs: ModelOutput, **kwargs):


    def to_log_dict(self, step: int) -> Dict[str, float]:
        log_dict = {
            'step': step,
            'profiling/Time taken: GRPOTrainer._move_model_to_vllm': self.weight_sync_time,
            'profiling/Time taken: GRPOTrainer.generate': self.generate_time,
            'train/loss': self.loss,
            'train/grad_norm': self.grad_norm,
        }
        if self.rewards:
            log_dict['train/reward'] = sum(self.rewards) / len(self.rewards)
            log_dict['train/reward_std'] = torch.tensor(self.rewards).std().item() if len(self.rewards) > 1 else 0.0
        if self.format_rewards:
            log_dict['train/rewards/Format/mean'] = sum(self.format_rewards) / len(self.format_rewards)
        if self.accuracy_rewards:
            log_dict['train/rewards/CountdownORM/mean'] = sum(self.accuracy_rewards) / len(self.accuracy_rewards)
        if self.completion_lengths:
            log_dict['train/completions/mean_length'] = sum(self.completion_lengths) / len(self.completion_lengths)
        return log_dict
