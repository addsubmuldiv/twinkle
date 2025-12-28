from typing import Dict, Any, Union, List
import torch

from twinkle.processor.base import InputProcessor
from twinkle.data_format import InputFeature


class GRPOLossProcessor(InputProcessor):
    """
    Processor for preparing inputs required by GRPOLoss.
    
    This processor computes intermediate variables from raw data (input_ids, labels):
    - completion_mask: Boolean mask for completion tokens
    - logits_to_keep: Number of completion tokens to keep
    - num_items_in_batch: Total completion tokens (for DAPO/CISPO normalization)
    
    The processor infers completion positions from labels:
    - labels == -100: prompt tokens (ignored in loss)
    - labels != -100: completion tokens (used in loss)
    """
    
    def __init__(self, ignore_index: int = -100):
        self.ignore_index = ignore_index
    
    def __call__(
        self, 
        inputs: Union[InputFeature, List[InputFeature], Dict[str, Any]],
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Process raw inputs and add GRPO-specific fields.
        
        Args:
            inputs: InputFeature, list of InputFeature, or dict containing:
                - input_ids: Token ids, shape [batch_size, seq_len]
                - labels: Labels with -100 for prompt tokens, shape [batch_size, seq_len]
                
        Returns:
            inputs: Dict with additional fields:
                - completion_mask: Mask for completion tokens, shape [batch_size, logits_to_keep]
                - logits_to_keep: Number of completion tokens (int)
                - num_items_in_batch: Total completion tokens across batch (int)
        """
        # Handle different input types and convert to dict
        if isinstance(inputs, list):
            inputs = self.collate_fn(inputs)
        
        if isinstance(inputs, InputFeature):
            inputs = inputs.to_transformers_dict()
        
        # Ensure inputs is a dict
        if not isinstance(inputs, dict):
            raise TypeError(f"Expected dict, InputFeature or list, got {type(inputs)}")
        
        return self._prepare_grpo_fields(inputs)
    
    def _prepare_grpo_fields(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Compute GRPO-specific fields from labels."""
        labels = inputs['labels']
        
        # Compute logits_to_keep: maximum completion length across batch
        non_ignored = (labels != self.ignore_index).int()
        first_non_ignored = non_ignored.argmax(dim=-1)
        seq_len = labels.shape[-1]
        logits_to_keep = (seq_len - first_non_ignored).max().item()
        
        # Create completion mask for the last logits_to_keep positions
        completion_mask = (labels[:, -logits_to_keep:] != self.ignore_index).float()
        
        # Compute num_items_in_batch: total completion tokens
        num_items_in_batch = completion_mask.sum().int().item()
        
        # Update inputs with GRPO-specific fields
        inputs['completion_mask'] = completion_mask
        inputs['logits_to_keep'] = logits_to_keep
        inputs['num_items_in_batch'] = num_items_in_batch
        
        return inputs
