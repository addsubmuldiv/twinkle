# Twinkle Megatron 组件架构

## 整体代码结构

```
twinkle/src/twinkle/
├── model/
│   ├── megatron.py           # MegatronModel 主类（对外接口）
│   └── strategy/
│       └── megatron.py       # MegatronStrategy 策略类
└── megatron/                  # Megatron-Core 集成模块
    ├── __init__.py            # 公共 API 导出
    ├── utils.py               # 工具函数和配置映射
    ├── worker.py              # [已废弃] Ray Worker 类
    ├── tuners/
    │   ├── __init__.py
    │   └── lora.py            # LoRA 并行线性层实现
    └── model/
        ├── __init__.py
        ├── bridge.py          # HF ↔ Megatron 权重转换桥
        ├── initializer.py     # 模型初始化器
        └── qwen3.py           # Qwen3 模型支持
```

## 核心组件详解

### 1. MegatronModel (`model/megatron.py`)

**作用**：对外暴露的主要接口类，封装了 Megatron 模型的完整训练流程。

**关键特性**：
- 使用 `@remote_class(execute='all')` 装饰器，支持 Ray 分布式
- 提供与 `TransformersModel` 类似的 API
- 支持 TP/PP/CP/EP 多种并行策略
- 集成 PEFT/LoRA 微调

**核心方法**：
```python
# 初始化
model = MegatronModel(
    pretrained_model_name_or_path='Qwen/Qwen2.5-7B-Instruct',
    tensor_model_parallel_size=2,
    pipeline_model_parallel_size=2,
)

# 添加 LoRA
model.add_adapter_to_model('lora', LoraConfig(...))

# 训练循环
output = model.forward_backward(inputs=batch, adapter_name='lora')
model.step(adapter_name='lora')
```

### 2. MegatronStrategy (`model/strategy/megatron.py`)

**作用**：管理 Megatron 分布式并行状态的策略类。

**关键特性**：
- 封装 `mpu.initialize_model_parallel()` 调用
- 支持 local (torchrun) 和 Ray 两种执行模式
- 自动检测 `TWINKLE_MODE` 环境变量
- 提供 TP/PP/DP/CP/EP 进程组访问

**初始化流程**：
```python
# Local 模式（torchrun）
# - 环境变量由 torchrun 设置
# - dist.init_process_group 使用默认值

# Ray 模式
# - 读取 RayHelper 设置的 RANK, WORLD_SIZE, MASTER_ADDR 等
# - 显式传递给 dist.init_process_group
```

### 3. TwinkleBridgeInitializer (`megatron/model/bridge.py`)

**作用**：HuggingFace 到 Megatron 的模型初始化和权重转换。

**关键特性**：
- 自动转换 HF config 到 Megatron TransformerConfig
- 支持流式加载大模型权重（避免 OOM）
- 处理 TP/PP 权重分片
- 支持 MoE 模型

**核心流程**：
```
HuggingFace Model
    ↓ (AutoConfig.from_pretrained)
HF Config
    ↓ (convert_hf_config)
Megatron TransformerConfig
    ↓ (GPTModel)
Megatron Model
    ↓ (TwinkleBridgeAdapter.load_weights)
Loaded Megatron Model
```

### 4. MegatronModelInitializer (`megatron/model/initializer.py`)

**作用**：替代方案的模型初始化器（非 bridge 模式）。

**与 Bridge 的区别**：
- Bridge：使用 `initialize_megatron` 初始化 Megatron 环境
- Initializer：假设 Megatron 已经初始化，只负责模型创建

**推荐使用 Bridge 模式**（`use_megatron_bridge=True`，默认）。

### 5. LoraParallelLinear (`megatron/tuners/lora.py`)

**作用**：为 Megatron 的并行线性层提供 LoRA 支持。

**关键特性**：
- 适配 TransformerEngine 的 TELinear、TEColumnParallelLinear 等
- 保持 TP 兼容性
- 支持 `dispatch_megatron` 注册到 PEFT

### 6. 工具函数 (`megatron/utils.py`)

**关键函数**：
- `convert_hf_config`: HF config → Megatron config
- `find_all_linears`: 查找模型中所有线性层
- `set_linear_is_expert`: 标记 MoE expert 层
- `prepare_lora_model`: 准备 LoRA 模型
- `TenantProcessGroupManager`: 多租户进程组管理

## 数据流

### Local 模式 (torchrun)

```
torchrun --nproc_per_node=4 lora.py
    ↓
每个进程独立执行
    ↓
MegatronModel.__init__
    ↓ (use_megatron_bridge=True)
TwinkleBridgeInitializer._initialize_megatron
    ↓ (检测 TWINKLE_MODE='local')
dist.init_process_group(backend='nccl')  # 使用 torchrun 环境变量
    ↓
mpu.initialize_model_parallel(tp_size, pp_size, ...)
    ↓
模型创建和权重加载
```

### Ray 模式

```
python lora.py --mode ray
    ↓
twinkle.initialize(mode='ray')
    ↓
RayHelper.create_workers()  # 设置 RANK, WORLD_SIZE, MASTER_ADDR 等环境变量
    ↓
MegatronModel.__init__ (在 Ray actor 内)
    ↓ (use_megatron_bridge=True)
TwinkleBridgeInitializer._initialize_megatron
    ↓ (检测 TWINKLE_MODE='ray')
读取环境变量 RANK, WORLD_SIZE, MASTER_ADDR, MASTER_PORT
    ↓
dist.init_process_group(
    backend='nccl',
    init_method='tcp://...',
    rank=rank,
    world_size=world_size
)
    ↓
mpu.initialize_model_parallel(tp_size, pp_size, ...)
    ↓
模型创建和权重加载
```

## 废弃组件

### worker.py (MegatronWorker / MegatronWorkerGroup)

**状态**：已废弃

**原因**：
这是之前为了解决 Ray + Megatron 集成问题的临时方案。现在已经在核心代码中正确处理了分布式初始化，所以不再需要这个独立的 Worker 类。

**替代方案**：
直接使用 `MegatronModel` + `@remote_class` + `remote_group` 参数。
