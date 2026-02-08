# Advantage

Advantage is a component in reinforcement learning for calculating the advantage value of actions relative to the average level. In RLHF training, advantage functions guide policy optimization.

## Basic Interface

```python
class Advantage:
    
    def __call__(self,
                 rewards: Union['torch.Tensor', List[float]],
                 num_generations: int = 1,
                 scale: Literal['group', 'batch', 'none'] = 'group',
                 **kwargs) -> 'torch.Tensor':
        """
        Calculate advantage values
        
        Args:
            rewards: List or tensor of reward values
            num_generations: Number of samples generated per prompt
            scale: Normalization method
                - 'group': Normalize per group (GRPO)
                - 'batch': Normalize across entire batch
                - 'none': No normalization
        
        Returns:
            Advantage tensor
        """
        ...
```

## GRPOAdvantage

GRPO (Group Relative Policy Optimization) advantage function calculates advantages by subtracting the group mean.

```python
from twinkle.advantage import GRPOAdvantage

advantage_fn = GRPOAdvantage()

# Assume 2 prompts, 4 samples each
rewards = [0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0]  # 8 reward values
advantages = advantage_fn(rewards, num_generations=4, scale='group')

# advantages will be each group minus group mean:
# Group 1: [0.0-0.5, 1.0-0.5, 0.0-0.5, 1.0-0.5] = [-0.5, 0.5, -0.5, 0.5]
# Group 2: [1.0-0.25, 0.0-0.25, 0.0-0.25, 0.0-0.25] = [0.75, -0.25, -0.25, -0.25]
```

### How It Works

GRPO groups samples (each group corresponds to multiple generations from one prompt), then within each group:
1. Calculate the group mean reward
2. Each sample's advantage = that sample's reward - group mean
3. Optionally normalize advantage values

This method can:
- Reduce variance and improve training stability
- Perform relative comparisons within groups, better matching the relative nature of human preferences
- Avoid the impact of reward scale

## RLOOAdvantage

RLOO (Reinforcement Learning with Leave-One-Out) advantage function uses leave-one-out method to calculate baseline.

```python
from twinkle.advantage import RLOOAdvantage

advantage_fn = RLOOAdvantage()

rewards = [0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0]
advantages = advantage_fn(rewards, num_generations=4)

# For each sample, baseline is the mean of other samples
# First sample in first group: 0.0 - mean([1.0, 0.0, 1.0]) = 0.0 - 0.667 = -0.667
# ...
```

### How It Works

RLOO for each sample:
1. Calculate the mean reward of other samples in the group (leave-one-out baseline)
2. Advantage = sample reward - leave-one-out baseline
3. Optionally normalize

RLOO advantages:
- Avoids using sample's own information as baseline, reducing bias
- More accurately estimates counterfactual baseline
- Works better when number of samples is larger

## Usage Example

Using advantage function in GRPO training:

```python
from twinkle.advantage import GRPOAdvantage
from twinkle.model import TransformersModel
from twinkle.sampler import vLLMSampler
from twinkle.reward import MathReward

# Create components
actor = TransformersModel(model_id='Qwen/Qwen2.5-7B-Instruct')
sampler = vLLMSampler(model_id='Qwen/Qwen2.5-7B-Instruct')
reward_fn = MathReward()
advantage_fn = GRPOAdvantage()

# Training loop
for batch in dataloader:
    # 1. Sample generation
    response = sampler.sample(batch, num_samples=4)
    
    # 2. Calculate rewards
    rewards = reward_fn(response.trajectories, batch.ground_truths)
    
    # 3. Calculate advantages
    advantages = advantage_fn(rewards, num_generations=4)
    
    # 4. Policy optimization
    loss = actor.forward_backward(
        inputs=response.inputs,
        advantages=advantages
    )
    actor.clip_grad_and_step()
```

> The choice of advantage function has an important impact on RLHF training effectiveness. GRPO is simpler and more efficient, while RLOO is theoretically superior but requires more samples.
