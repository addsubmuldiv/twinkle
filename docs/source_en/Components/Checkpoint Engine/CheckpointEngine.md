# CheckpointEngine

CheckpointEngine is a component used to synchronize model weights between trainer and inference processes, primarily used in RLHF training to synchronize weights between Actor models and Rollout samplers.

## Basic Interface

```python
class CheckpointEngine(ABC):
    """Checkpoint engine base class
    
    The checkpoint engine handles weight synchronization between trainer and inference processes.
    """
    
    @abstractmethod
    def prepare(self) -> dict[str, Any]:
        """Prepare for weight synchronization"""
        ...
    
    @abstractmethod
    def init_process_group(self, rank: int, world_size: int, **kwargs):
        """Initialize process group"""
        ...
    
    @abstractmethod
    async def send_weights(self, weight_generator):
        """Send weights (called in trainer process)"""
        ...
    
    @abstractmethod
    def receive_weights(self) -> AsyncGenerator:
        """Receive weights (called in inference process)"""
        ...
    
    @abstractmethod
    def finalize(self):
        """Clean up resources"""
        ...
```

## NCCLCheckpointEngine

A checkpoint engine that uses NCCL for high-speed weight transfer between GPUs.

```python
from twinkle.checkpoint_engine import NCCLCheckpointEngine

# In training process (rank 0)
engine = NCCLCheckpointEngine(bucket_size=512<<20)  # 512MB bucket
engine.is_master = True
engine.prepare()
engine.init_process_group(rank=0, world_size=5)

# Send weights
await engine.send_weights(model.named_parameters())
engine.finalize()

# In inference process (rank 1-4)
engine = NCCLCheckpointEngine(bucket_size=512<<20)
engine.prepare()
engine.init_process_group(rank=1, world_size=5, master_metadata=metadata)

# Receive weights
async for name, tensor in engine.receive_weights():
    model.load_state_dict({name: tensor}, strict=False)
engine.finalize()
```

### Features

- **High-speed transfer**: Uses NCCL for GPU-to-GPU point-to-point high-speed transfer
- **Zero-copy**: Direct transfer between GPU memories without going through CPU
- **Bucketed transfer**: Supports bucketed transfer for large models

## HCCLCheckpointEngine  

A checkpoint engine that uses HCCL for weight transfer between Ascend NPUs.

```python
from twinkle.checkpoint_engine import HCCLCheckpointEngine

engine = HCCLCheckpointEngine(bucket_size=512<<20)
# Usage is the same as NCCLCheckpointEngine
```

## Usage Scenarios

Typical usage in RLHF training:

```python
import asyncio
from twinkle.checkpoint_engine import NCCLCheckpointEngine
from twinkle.model import TransformersModel
from twinkle.sampler import vLLMSampler

# Trainer process
async def trainer_process():
    actor = TransformersModel(model_id='Qwen/Qwen2.5-7B-Instruct')
    engine = NCCLCheckpointEngine()
    engine.is_master = True
    engine.prepare()
    engine.init_process_group(rank=0, world_size=2)
    
    for step in range(num_steps):
        # Train one step
        loss = actor.forward_backward(...)
        actor.clip_grad_and_step()
        
        # Periodically sync weights to sampler
        if step % sync_interval == 0:
            await engine.send_weights(
                actor.model.named_parameters()
            )
    
    engine.finalize()

# Sampler process
async def sampler_process():
    sampler = vLLMSampler(model_id='Qwen/Qwen2.5-7B-Instruct')
    engine = NCCLCheckpointEngine()
    engine.prepare()
    engine.init_process_group(rank=1, world_size=2)
    
    # Receive and load new weights
    async for name, tensor in engine.receive_weights():
        sampler.engine.update_weight(name, tensor)
    
    engine.finalize()

# Run
asyncio.run(trainer_process())
asyncio.run(sampler_process())
```

## Configuration Parameters

- **bucket_size**: Weight bucket size, controls the amount of data transferred each time. Larger buckets can improve transfer efficiency but consume more memory
- **compression**: Whether to compress weight data (future support)
- **timeout**: Transfer timeout duration

> Checkpoint engine is a key component of RLHF training infrastructure, ensuring that trainers and samplers use consistent model weights.
