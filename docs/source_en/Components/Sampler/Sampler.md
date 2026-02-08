# Sampler

Sampler is a component in Twinkle for generating model outputs, primarily used for sample generation in RLHF training. The sampler supports multiple inference engines, including vLLM and native PyTorch.

## Basic Interface

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
        """Sample from given inputs"""
        ...
    
    def add_adapter_to_model(self, adapter_name: str, config_or_dir, **kwargs):
        """Add LoRA adapter"""
        ...
    
    def set_template(self, template_cls: Union[Template, Type[Template], str], **kwargs):
        """Set template"""
        ...
```

The core method of the sampler is `sample`, which accepts input data and returns generated samples.

## vLLMSampler

vLLMSampler uses the vLLM engine for efficient inference, supporting high-throughput batch sampling.

```python
from twinkle.sampler import vLLMSampler
from twinkle.data_format import SamplingParams
from twinkle import DeviceMesh

# Create sampler
sampler = vLLMSampler(
    model_id='ms://Qwen/Qwen2.5-7B-Instruct',
    device_mesh=DeviceMesh.from_sizes(dp_size=2, tp_size=2),
    remote_group='sampler_group'
)

# Add LoRA
sampler.add_adapter_to_model('my_lora', 'path/to/lora')

# Set sampling parameters
params = SamplingParams(
    max_tokens=512,
    temperature=0.7,
    top_p=0.9,
    top_k=50
)

# Perform sampling
response = sampler.sample(
    trajectories,
    sampling_params=params,
    adapter_name='my_lora',
    num_samples=4  # Generate 4 samples per prompt
)
```

### Features

- **High Performance**: Achieves high throughput using PagedAttention and continuous batching
- **LoRA Support**: Supports dynamic loading and switching of LoRA adapters
- **Multi-Sample Generation**: Can generate multiple samples for each prompt
- **Tensor Parallel**: Supports tensor parallelism to accelerate large model inference

## TorchSampler

TorchSampler uses native PyTorch and transformers for inference, suitable for small-scale sampling or debugging.

```python
from twinkle.sampler import TorchSampler

sampler = TorchSampler(
    model_id='ms://Qwen/Qwen2.5-7B-Instruct',
    device_mesh=DeviceMesh.from_sizes(dp_size=1),
)

response = sampler.sample(trajectories, sampling_params=params)
```

### Features

- **Easy to Use**: Based on transformers' standard interface
- **High Flexibility**: Easy to customize and extend
- **Low Memory Usage**: Suitable for small-scale sampling

## Remote Execution

Samplers support the `@remote_class` decorator and can run in Ray clusters:

```python
import twinkle
from twinkle import DeviceGroup

# Initialize Ray cluster
device_groups = [
    DeviceGroup(name='sampler', ranks=4, device_type='cuda')
]
twinkle.initialize('ray', groups=device_groups)

# Create remote sampler
sampler = vLLMSampler(
    model_id='ms://Qwen/Qwen2.5-7B-Instruct',
    device_mesh=DeviceMesh.from_sizes(dp_size=4),
    remote_group='sampler'
)

# sample method executes in remote worker
response = sampler.sample(trajectories, sampling_params=params)
```

> Samplers in RLHF training are typically separated from the Actor model, using different hardware resources to avoid interference between inference and training.
