# CheckpointEngine

CheckpointEngine (检查点引擎) 是用于在训练器和推理进程之间同步模型权重的组件,主要用于 RLHF 训练中 Actor 模型和 Rollout 采样器之间的权重同步。

## 基本接口

```python
class CheckpointEngine(ABC):
    """检查点引擎基类
    
    检查点引擎处理训练器和推理进程之间的权重同步。
    """
    
    @abstractmethod
    def prepare(self) -> dict[str, Any]:
        """准备权重同步前的准备工作"""
        ...
    
    @abstractmethod
    def init_process_group(self, rank: int, world_size: int, **kwargs):
        """初始化进程组"""
        ...
    
    @abstractmethod
    async def send_weights(self, weight_generator):
        """发送权重(在训练器进程中调用)"""
        ...
    
    @abstractmethod
    def receive_weights(self) -> AsyncGenerator:
        """接收权重(在推理进程中调用)"""
        ...
    
    @abstractmethod
    def finalize(self):
        """清理资源"""
        ...
```

## NCCLCheckpointEngine

使用 NCCL 进行 GPU 间高速权重传输的检查点引擎。

```python
from twinkle.checkpoint_engine import NCCLCheckpointEngine

# 在训练进程 (rank 0)
engine = NCCLCheckpointEngine(bucket_size=512<<20)  # 512MB bucket
engine.is_master = True
engine.prepare()
engine.init_process_group(rank=0, world_size=5)

# 发送权重
await engine.send_weights(model.named_parameters())
engine.finalize()

# 在推理进程 (rank 1-4)
engine = NCCLCheckpointEngine(bucket_size=512<<20)
engine.prepare()
engine.init_process_group(rank=1, world_size=5, master_metadata=metadata)

# 接收权重
async for name, tensor in engine.receive_weights():
    model.load_state_dict({name: tensor}, strict=False)
engine.finalize()
```

### 特性

- **高速传输**: 使用 NCCL 实现 GPU 间点对点高速传输
- **零拷贝**: 直接在 GPU 内存间传输,无需经过 CPU
- **分桶传输**: 支持大模型的分桶传输

## HCCLCheckpointEngine  

使用 HCCL 进行昇腾 NPU 间权重传输的检查点引擎。

```python
from twinkle.checkpoint_engine import HCCLCheckpointEngine

engine = HCCLCheckpointEngine(bucket_size=512<<20)
# 使用方式与 NCCLCheckpointEngine 相同
```

## 使用场景

在 RLHF 训练中的典型使用:

```python
import asyncio
from twinkle.checkpoint_engine import NCCLCheckpointEngine
from twinkle.model import TransformersModel
from twinkle.sampler import vLLMSampler

# 训练器进程
async def trainer_process():
    actor = TransformersModel(model_id='Qwen/Qwen2.5-7B-Instruct')
    engine = NCCLCheckpointEngine()
    engine.is_master = True
    engine.prepare()
    engine.init_process_group(rank=0, world_size=2)
    
    for step in range(num_steps):
        # 训练一步
        loss = actor.forward_backward(...)
        actor.clip_grad_and_step()
        
        # 定期同步权重到采样器
        if step % sync_interval == 0:
            await engine.send_weights(
                actor.model.named_parameters()
            )
    
    engine.finalize()

# 采样器进程
async def sampler_process():
    sampler = vLLMSampler(model_id='Qwen/Qwen2.5-7B-Instruct')
    engine = NCCLCheckpointEngine()
    engine.prepare()
    engine.init_process_group(rank=1, world_size=2)
    
    # 接收并加载新权重
    async for name, tensor in engine.receive_weights():
        sampler.engine.update_weight(name, tensor)
    
    engine.finalize()

# 运行
asyncio.run(trainer_process())
asyncio.run(sampler_process())
```

## 配置参数

- **bucket_size**: 权重桶大小,控制每次传输的数据量。较大的桶可以提高传输效率,但会占用更多内存
- **compression**: 是否压缩权重数据(未来支持)
- **timeout**: 传输超时时间

> 检查点引擎是 RLHF 训练基础设施的关键组件,确保训练器和采样器使用一致的模型权重。
