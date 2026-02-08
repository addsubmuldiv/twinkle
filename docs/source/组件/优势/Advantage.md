# Advantage

Advantage (优势函数) 是强化学习中用于计算动作相对于平均水平的优势值的组件。在 RLHF 训练中,优势函数用于指导策略优化。

## 基本接口

```python
class Advantage:
    
    def __call__(self,
                 rewards: Union['torch.Tensor', List[float]],
                 num_generations: int = 1,
                 scale: Literal['group', 'batch', 'none'] = 'group',
                 **kwargs) -> 'torch.Tensor':
        """
        计算优势值
        
        Args:
            rewards: 奖励值列表或张量
            num_generations: 每个 prompt 生成的样本数量
            scale: 归一化方式
                - 'group': 对每组样本进行归一化 (GRPO)
                - 'batch': 对整个 batch 进行归一化
                - 'none': 不进行归一化
        
        Returns:
            优势值张量
        """
        ...
```

## GRPOAdvantage

GRPO (Group Relative Policy Optimization) 优势函数通过减去组内均值来计算优势。

```python
from twinkle.advantage import GRPOAdvantage

advantage_fn = GRPOAdvantage()

# 假设有 2 个 prompt,每个生成 4 个样本
rewards = [0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0]  # 8 个奖励值
advantages = advantage_fn(rewards, num_generations=4, scale='group')

# advantages 会是每组减去组内均值:
# 第一组: [0.0-0.5, 1.0-0.5, 0.0-0.5, 1.0-0.5] = [-0.5, 0.5, -0.5, 0.5]
# 第二组: [1.0-0.25, 0.0-0.25, 0.0-0.25, 0.0-0.25] = [0.75, -0.25, -0.25, -0.25]
```

### 工作原理

GRPO 将样本分组(每组对应一个 prompt 的多个生成),然后在组内:
1. 计算组内奖励均值
2. 每个样本的优势 = 该样本的奖励 - 组内均值
3. 可选地对优势值进行归一化

这种方法能够:
- 减少方差,提高训练稳定性
- 在组内进行相对比较,更符合人类偏好的相对性
- 避免奖励尺度的影响

## RLOOAdvantage

RLOO (Reinforcement Learning with Leave-One-Out) 优势函数使用留一法计算基线。

```python
from twinkle.advantage import RLOOAdvantage

advantage_fn = RLOOAdvantage()

rewards = [0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0]
advantages = advantage_fn(rewards, num_generations=4)

# 对于每个样本,基线是除了它以外的其他样本的均值
# 第一组第一个样本: 0.0 - mean([1.0, 0.0, 1.0]) = 0.0 - 0.667 = -0.667
# ...
```

### 工作原理

RLOO 对每个样本:
1. 计算除该样本外组内其他样本的奖励均值 (留一基线)
2. 优势 = 该样本奖励 - 留一基线
3. 可选地进行归一化

RLOO 的优势:
- 避免使用样本自身信息作为基线,减少偏差
- 更准确地估计反事实基线
- 在样本数量较多时效果更好

## 使用示例

在 GRPO 训练中使用优势函数:

```python
from twinkle.advantage import GRPOAdvantage
from twinkle.model import TransformersModel
from twinkle.sampler import vLLMSampler
from twinkle.reward import MathReward

# 创建组件
actor = TransformersModel(model_id='Qwen/Qwen2.5-7B-Instruct')
sampler = vLLMSampler(model_id='Qwen/Qwen2.5-7B-Instruct')
reward_fn = MathReward()
advantage_fn = GRPOAdvantage()

# 训练循环
for batch in dataloader:
    # 1. 采样生成
    response = sampler.sample(batch, num_samples=4)
    
    # 2. 计算奖励
    rewards = reward_fn(response.trajectories, batch.ground_truths)
    
    # 3. 计算优势
    advantages = advantage_fn(rewards, num_generations=4)
    
    # 4. 策略优化
    loss = actor.forward_backward(
        inputs=response.inputs,
        advantages=advantages
    )
    actor.clip_grad_and_step()
```

> 优势函数的选择对 RLHF 训练效果有重要影响。GRPO 更简单高效,RLOO 在理论上更优但需要更多样本。
