# Sampler

Sampler (采样器) 是 Twinkle 中用于生成模型输出的组件,主要用于 RLHF 训练中的样本生成。采样器支持多种推理引擎,包括 vLLM 和原生 PyTorch。

## 基本接口

```python
class Sampler(ABC):
    
    @abstractmethod
    def sample(
        self,
        inputs: Union[InputFeature, List[InputFeature], Trajectory, List[Trajectory]],
        sampling_params: Optional[SamplingParams] = None,
        adapter_name: str = '',
        *,
        num_samples: int = 1,
    ) -> SampleResponse:
        """对给定输入进行采样"""
        ...
    
    def add_adapter_to_model(self, adapter_name: str, config_or_dir, **kwargs):
        """添加 LoRA 适配器"""
        ...
    
    def set_template(self, template_cls: Union[Template, Type[Template], str], **kwargs):
        """设置模板"""
        ...
```

采样器的核心方法是 `sample`,它接受输入数据并返回生成的样本。

## vLLMSampler

vLLMSampler 使用 vLLM 引擎进行高效推理,支持高吞吐量的批量采样。

```python
from twinkle.sampler import vLLMSampler
from twinkle.data_format import SamplingParams
from twinkle import DeviceMesh

# 创建采样器
sampler = vLLMSampler(
    model_id='ms://Qwen/Qwen2.5-7B-Instruct',
    device_mesh=DeviceMesh.from_sizes(dp_size=2, tp_size=2),
    remote_group='sampler_group'
)

# 添加 LoRA
sampler.add_adapter_to_model('my_lora', 'path/to/lora')

# 设置采样参数
params = SamplingParams(
    max_tokens=512,
    temperature=0.7,
    top_p=0.9,
    top_k=50
)

# 进行采样
response = sampler.sample(
    trajectories,
    sampling_params=params,
    adapter_name='my_lora',
    num_samples=4  # 每个 prompt 生成 4 个样本
)
```

### 特性

- **高性能**: 使用 PagedAttention 和连续批处理实现高吞吐量
- **LoRA 支持**: 支持动态加载和切换 LoRA 适配器
- **多样本生成**: 可以为每个 prompt 生成多个样本
- **Tensor Parallel**: 支持张量并行加速大模型推理

## TorchSampler

TorchSampler 使用原生 PyTorch 和 transformers 进行推理,适合小规模采样或调试。

```python
from twinkle.sampler import TorchSampler
from twinkle import DeviceMesh

sampler = TorchSampler(
    model_id='ms://Qwen/Qwen2.5-7B-Instruct',
    device_mesh=DeviceMesh.from_sizes(dp_size=1),
)

response = sampler.sample(trajectories, sampling_params=params)
```

### 特性

- **简单易用**: 基于 transformers 的标准接口
- **灵活性高**: 容易定制和扩展
- **内存占用小**: 适合小规模采样

## 远程执行

采样器支持 `@remote_class` 装饰器,可以在 Ray 集群中运行:

```python
import twinkle
from twinkle import DeviceGroup, DeviceMesh
from twinkle.sampler import VLLMSampler

# 初始化 Ray 集群
device_groups = [
    DeviceGroup(name='sampler', ranks=4, device_type='cuda')
]
twinkle.initialize('ray', groups=device_groups)

# 创建远程采样器
sampler = VLLMSampler(
    model_id='ms://Qwen/Qwen2.5-7B-Instruct',
    device_mesh=DeviceMesh.from_sizes(dp_size=4),
    remote_group='sampler'
)

# sample 方法会在 remote worker 中执行
response = sampler.sample(trajectories, sampling_params=params)
```

> 采样器在 RLHF 训练中通常与 Actor 模型分离,使用不同的硬件资源,避免推理和训练相互干扰。
