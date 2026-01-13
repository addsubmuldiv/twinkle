# Megatron + Ray 模式开发状态

## 最新更新

已修改 `TwinkleBridgeInitializer._initialize_megatron()` 和 `MegatronStrategy.initialize()`，
使其能够自动检测 `TWINKLE_MODE` 环境变量，并在 Ray 模式下正确初始化分布式环境。

**关键修改**：
1. `bridge.py`: `_initialize_megatron()` 检测 Ray 模式并使用正确的环境变量初始化
2. `megatron.py` (strategy): `initialize()` 同样支持 Ray 模式
3. `lora.py`: 统一的 demo，通过 `--mode ray` 切换

## Twinkle Ray 架构解析

### RayHelper 环境变量传递

在 `ray_helper.py` 第 247-268 行，`RayHelper.create_workers` **已经正确设置**了所有必要的环境变量：

```python
env_vars.update({
    'WORLD_SIZE': str(world_size),
    'RANK': str(rank),
    'LOCAL_RANK': str(0),
    'MASTER_ADDR': ip,
    'MASTER_PORT': str(port),
    'TWINKLE_MODE': 'ray',
    ...
})
runtime_env = RuntimeEnv(env_vars=env_vars)
worker = worker_cls.options(runtime_env=runtime_env, ...).remote(*args, **kwargs)
```

**每个 Ray actor 都有正确隔离的环境变量**。

### 为什么之前没有工作？

问题不在环境变量传递，而在于 **`TwinkleBridgeInitializer` 没有读取这些环境变量**。

之前的代码：
```python
if not dist.is_initialized():
    dist.init_process_group(backend='nccl')  # 只用默认值！
```

修复后：
```python
if twinkle_mode == 'ray':
    rank = int(os.environ.get('RANK', '0'))
    world_size = int(os.environ.get('WORLD_SIZE', '1'))
    master_addr = os.environ.get('MASTER_ADDR', 'localhost')
    master_port = os.environ.get('MASTER_PORT', '29500')
    
    dist.init_process_group(
        backend='nccl',
        init_method=f'tcp://{master_addr}:{master_port}',
        rank=rank,
        world_size=world_size,
    )
```

## 当前实现总结

### MegatronWorker 类（临时方案）

之前实现了独立的 `MegatronWorkerGroup` 和 `MegatronWorker` 类作为临时方案。
现在已修改核心代码，理论上不再需要这个类。

### PP=1 梯度问题（仍存在）

**问题描述**：
- PEFT/LoRA 包装的模型在 Megatron 的 `forward_backward_no_pipelining` 下
- 模型输出 `logits.requires_grad=False`
- 导致 `backward_step` 被跳过

**根本原因**：
- Megatron 的 `backward_step` 在第 464 行检查：`if output_tensor[0].requires_grad:`
- 如果为 False，backward 被跳过
- PEFT 模型的 forward 可能破坏了梯度追踪

**临时解决方案**：使用 PP > 1

### 为什么 Ray demo 与 Local demo 分开？

当前有两个独立的 demo 文件：
- `lora.py`: Local 模式，使用 torchrun 启动
- `lora_ray.py`: Ray 模式，使用 MegatronWorkerGroup

**分开的原因**：

1. **启动方式不同**
   - Local: `torchrun --nproc_per_node=N` 启动多个进程
   - Ray: `python` 启动单个 driver，创建 Ray actors

2. **环境变量来源不同**
   - Local: torchrun 自动设置 `LOCAL_RANK`, `WORLD_SIZE` 等
   - Ray: 需要在 actor 内部手动设置

3. **分布式初始化不同**
   - Local: 在脚本开头调用 `twinkle.initialize(mode='local')`
   - Ray: 每个 worker 需要单独初始化 `torch.distributed` 和 `mpu`

4. **代码入口不同**
   - Local: `lora.py` 直接在 main 中调用 train()
   - Ray: `lora_ray.py` 创建 worker group，然后调用 worker 方法

---

## TODO 列表

### 高优先级

1. **[ ] 修复 PP=1 时的梯度问题**
   - 问题：PEFT 模型 forward 输出 `requires_grad=False`
   - 可能的方案：
     - a) 在 `MegatronModel.forward_backward` 中使用手动 backward
     - b) 修改 PEFT 集成方式，确保梯度正确流动
     - c) 找出 Megatron/TE 哪里断开了梯度

2. **[ ] 统一 local 和 ray 模式的 demo**
   - 目标：单一 `lora.py` 同时支持两种模式
   - 需要修改 `MegatronModel` 或 `MegatronStrategy` 来处理 Ray 环境下的初始化
   - 参考 `twinkle/cookbook/sft/lora.py` 的统一设计

3. **[ ] 将 MegatronWorker 逻辑集成到 Twinkle 核心架构**
   - 选项 A：修改 `MegatronStrategy` 添加 Ray 模式支持
   - 选项 B：修改 `MegatronModel` 在 `@remote_class` 下正确初始化
   - 选项 C：创建 `MegatronRayStrategy` 专门处理 Ray + Megatron

### 中优先级

4. **[ ] 支持 DP > 1 的 Ray 模式**
   - 当前只测试了 TP+PP 组合
   - DP 需要正确的梯度同步（`finalize_model_grads`）

5. **[ ] 支持 CP > 1 的 Ray 模式**
   - Context Parallel 需要正确的序列分割和 loss 聚合

6. **[ ] 移除 MegatronWorkerGroup 的 hardcode**
   - 当前 worker.py 中有很多硬编码逻辑
   - 应该复用 `TwinkleBridgeInitializer` 的配置

### 低优先级

7. **[ ] 添加 checkpoint 保存/加载支持**
   - 当前 Ray 模式没有实现 `model.save()`

8. **[ ] 性能优化**
   - 减少 Ray object 传输开销
   - 优化 batch 分发

9. **[ ] 错误处理和恢复**
   - Worker 失败时的处理
   - 分布式 barrier 超时处理

---

## 统一 demo 的设计方案

要实现 `lora.py` 同时支持 local 和 ray 模式，需要：

```python
# 方案：在 MegatronModel/MegatronStrategy 中检测模式并初始化

# 1. 修改 twinkle.initialize 添加 Megatron 专用参数
twinkle.initialize(
    mode='ray',  # 或 'local'
    megatron_config={
        'tp_size': 2,
        'pp_size': 2,
        # ...
    }
)

# 2. MegatronModel 在 __init__ 中检测模式
class MegatronModel:
    def __init__(self, ...):
        if twinkle.get_mode() == 'ray':
            # 在 actor 内部初始化分布式
            self._init_ray_distributed()
        else:
            # 使用 torchrun 已设置的分布式
            self._init_local_distributed()

# 3. RayHelper 在创建 actors 时设置环境变量
# 类似当前 MegatronWorkerGroup 的做法
```

**当前阻碍**：
1. `@remote_class` 不支持在 actor 创建时传递自定义参数（如 rank, world_size）
2. `MegatronModel.__init__` 在 driver 和 worker 中都会被调用，需要区分

---

## 测试状态

| 配置 | GPUs | Local Mode | Ray Mode |
|------|------|------------|----------|
| TP=2, PP=2 | 4 | ✅ | ✅ |
| TP=1, PP=4 | 4 | ✅ | ✅ |
| TP=2, PP=1 | 2 | ✅ | ❌ (梯度问题) |
| TP=1, PP=2 | 2 | ✅ | 未测试 |
| DP=2, TP=2, PP=2 | 8 | ⚠️ | 未测试 |
| CP > 1 | - | ⚠️ | 未测试 |

---

## 文件清单

- `twinkle/src/twinkle/megatron/worker.py`: MegatronWorker 和 MegatronWorkerGroup 实现
- `twinkle/cookbook/megatron/lora.py`: Local 模式 demo
- `twinkle/cookbook/megatron/lora_ray.py`: Ray 模式 demo（暂时保留）
- `twinkle/src/twinkle/model/megatron.py`: MegatronModel 核心实现
- `twinkle/src/twinkle/megatron/model/bridge.py`: Megatron 模型初始化 bridge
