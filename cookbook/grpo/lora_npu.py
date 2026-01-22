import numpy as np
import copy
from peft import LoraConfig
import os
import twinkle
from twinkle import DeviceMesh, get_device_placement
from twinkle.dataloader import DataLoader
from twinkle.dataset import Dataset, DatasetMeta
from twinkle.infra import DeviceGroup, remote_function, remote_class
from twinkle.model import TransformersModel
from twinkle.reward import MathReward
from twinkle.sampler import VLLMSampler
from twinkle.weight_loader import NativeLoader

# import debugpy
# debugpy.listen(5678)
# debugpy.wait_for_client()
os.environ.setdefault('TRUST_REMOTE_CODE', '1')
os.environ.setdefault('TWINKLE_SEED', '42')
os.environ.setdefault('TWINKLE_FULL_DETERMINISM', '1')
use_ref_model = os.environ.get('TWINKLE_USE_REF_MODEL', '1') != '0'
num_generations = 2
kl_beta = 0.0
max_length = int(os.environ.get('TWINKLE_MAX_LENGTH', '4096'))
truncation_strategy = 'right'

def build_template_kwargs(include_model_id: bool = False):
    kwargs = {}
    if include_model_id:
        kwargs['model_id'] = os.environ.get('TWINKLE_MODEL_ID', '/home/zyh/model/Qwen3-0.6B')
    if max_length > 0:
        kwargs['max_length'] = max_length
        kwargs['truncation_strategy'] = truncation_strategy
    return kwargs

visible_devices_env = os.environ.get('ASCEND_RT_VISIBLE_DEVICES') or os.environ.get('CUDA_VISIBLE_DEVICES')
if visible_devices_env:
    visible_devices = [d for d in visible_devices_env.split(',') if d.strip()]
    nproc_per_node = len(visible_devices)
else:
    nproc_per_node = 8

def _parse_ranks_env(name: str):
    raw = os.environ.get(name)
    if not raw:
        return None
    ranks = [int(v.strip()) for v in raw.split(',') if v.strip()]
    return ranks or None

actor_ranks = _parse_ranks_env('TWINKLE_ACTOR_RANKS')
ref_ranks = _parse_ranks_env('TWINKLE_REF_RANKS')
if actor_ranks is None:
    actor_size = int(os.environ.get('TWINKLE_ACTOR_SIZE', '6'))
    actor_ranks = list(range(actor_size))
if ref_ranks is None and use_ref_model:
    ref_size = int(os.environ.get('TWINKLE_REF_SIZE', '2'))
    ref_start = (max(actor_ranks) + 1) if actor_ranks else 0
    ref_ranks = list(range(ref_start, ref_start + ref_size))


device_groups = [
    DeviceGroup(
        name='actor',
        ranks=actor_ranks,
        device_type='npu',
    ),
]
if use_ref_model:
    device_groups.append(
        DeviceGroup(
            name='ref',
            ranks=ref_ranks,
            device_type='npu',
        )
    )
actor_device_mesh = DeviceMesh(
    device_type='npu',
    mesh=np.array([len(actor_ranks)]),
    mesh_dim_names=('dp',),
)
ref_device_mesh = DeviceMesh(
    device_type='npu',
    mesh=np.array([len(ref_ranks) if ref_ranks is not None else 0]),
    mesh_dim_names=('dp',),
)

twinkle.initialize(mode='ray', groups=device_groups, nproc_per_node=nproc_per_node)


@remote_class()
class ActorGroup:

    def __init__(self, engine_args, lora_config=None, adapter_name=None, **kwargs):
        self.sampler = VLLMSampler(
            '/home/zyh/model/Qwen3-0.6B',
            engine_args,
            device_mesh=actor_device_mesh,
        )
        self.sampler.add_adapter_to_sampler(adapter_name, lora_config)
        self.sampler.set_template('Qwen3Template', adapter_name=adapter_name, **build_template_kwargs(include_model_id=True))

        self.model = TransformersModel(
            model_id='/home/zyh/model/Qwen3-0.6B', 
            remote_group='actor', 
            device_mesh=actor_device_mesh
        )
        self.model.add_adapter_to_model(adapter_name, lora_config)

        self.model.set_loss(
            'GRPOLoss',
            loss_type='grpo',
            epsilon=0.2,
            beta=kl_beta,
            num_generations=num_generations,
            scale_rewards='group',
        )
        self.model.set_optimizer('AdamW', lr=1e-6)
        self.model.set_lr_scheduler('LinearLR')
        self.model.set_template('Qwen3Template', **build_template_kwargs(include_model_id=True))
        self.model.set_processor('GRPOLossProcessor')
        self.weight_loader = NativeLoader()
        self.adapter_name = adapter_name
        self.lora_config = lora_config

    @remote_function(collect='flatten')
    def sample(self, batch):
        print(f"[debug] ActorGroup.sample called with batch type={type(batch)}", flush=True)
        print(f"[debug] ActorGroup.sample called with batch={batch}", flush=True)
        return self.sampler.sample(batch, adapter_name=self.adapter_name)

    @remote_function()
    def forward(self, inputs, **kwargs):
        outputs = self.model.forward(inputs=inputs, **kwargs)
        return outputs['logits']

    @remote_function()
    def forward_only(self, inputs, **kwargs):
        outputs = self.model.forward_only(inputs=inputs, **kwargs)
        return outputs['logits']

    @remote_function()
    def forward_backward(self, inputs, trajectories, ref_logits=None, old_logits=None, **kwargs):
        if old_logits is None:
            old_logits = self.model.forward_only(inputs=inputs, **kwargs)['logits']
        return self.model.forward_backward(
            inputs=inputs,
            trajectories=trajectories,
            ref_logits=ref_logits,
            old_logits=old_logits,
            **kwargs,
        )

    @remote_function()
    def step(self):
        return self.model.step()

    @remote_function()
    def zero_grad(self):
        return self.model.zero_grad()

    @remote_function()
    def lr_step(self):
        return self.model.lr_step()

    @remote_function()
    def sync_weights(self):
        self.weight_loader(self.model, self.sampler, self.adapter_name)

def create_dataset():
    dataset = Dataset(DatasetMeta('ms://modelscope/competition_math'))
    dataset.set_template('Qwen3Template', **build_template_kwargs(include_model_id=True))
    dataset.map('CompetitionMathGRPOProcessor')
    dataset.check(batched=True)
    return dataset


def train():
    dataloader = DataLoader(
        create_dataset, 
        remote_group='actor', 
        device_mesh=actor_device_mesh
    )
    
    engine_args = {
        "model": "/home/zyh/model/Qwen3-0.6B",
        "enable_lora": True,
        "max_loras": 1,
        "max_lora_rank": 64,
        "max_model_len": max_length,
        "gpu_memory_utilization": float(os.environ.get("TWINKLE_VLLM_GPU_MEMORY_UTILIZATION", "0.9")),
    }

    lora_config = LoraConfig(
        target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj']
    )

    actor_group = ActorGroup(
        engine_args,
        remote_group='actor',
        lora_config=lora_config,
        adapter_name='default',
    )
    
    ref_model = None
    if use_ref_model:
        ref_model = TransformersModel(
            model_id='/home/zyh/model/Qwen3-0.6B', 
            remote_group='ref', 
            device_mesh=ref_device_mesh
        )
        ref_model.set_processor('InputProcessor')
        ref_model.set_template('Qwen3Template', **build_template_kwargs())
    reward = MathReward()
    try:
        from transformers import AutoTokenizer
        _tok = AutoTokenizer.from_pretrained(os.environ.get('TWINKLE_MODEL_ID', '/home/zyh/model/Qwen3-0.6B'))
        _eos_ids = _tok.eos_token_id
        if _eos_ids is None:
            eos_token_ids = []
        elif isinstance(_eos_ids, (list, tuple)):
            eos_token_ids = list(_eos_ids)
        else:
            eos_token_ids = [_eos_ids]
    except Exception:
        eos_token_ids = []
    
    print("Device placement:", get_device_placement())
    
    step = 0
    max_steps = int(os.environ.get('TWINKLE_MAX_STEPS', '20'))
    for batch in dataloader:
        step += 1
        print(f"[step {step}] batch ready", flush=True)
        if isinstance(batch, dict):
            batch_list = [batch]
        else:
            batch_list = list(batch)
        ground_truths = copy.deepcopy(batch_list)

        for record in batch_list:
            if not isinstance(record, dict):
                continue
            gen = record.get('generation_config')
            if gen is None:
                gen = {}
                record['generation_config'] = gen
            if isinstance(gen, list):
                gen = dict(gen)
                record['generation_config'] = gen
            if eos_token_ids:
                gen.setdefault('logit_bias', {int(tid): -100.0 for tid in eos_token_ids})
            gen.setdefault('max_tokens', 128)
            gen.setdefault('temperature', 0.2)
            gen.setdefault('top_p', 0.95)
            gen.setdefault('stop', ['\n', '<|im_end|>'])
            gen.setdefault('min_tokens', 8)
            gen.setdefault('ignore_eos', True)

        trajectories = actor_group.sample(batch_list)

        if callable(trajectories):
            trajectories = trajectories()
        print(f"[step {step}] sampled trajectories: {trajectories}", flush=True)
        def _extract_from_record(record, keys):
            if not isinstance(record, dict):
                return None
            for key in keys:
                if key in record and isinstance(record[key], str):
                    return record[key]
            return None

        def _extract_prompt_gt(container):
            record = None
            if isinstance(container, dict):
                record = container
            elif isinstance(container, (list, tuple)) and container:
                record = container[0]
            if isinstance(record, dict):
                # Try chat-style messages first.
                msgs = record.get('messages')
                if isinstance(msgs, list) and msgs:
                    first = msgs[0]
                    last = msgs[-1]
                    prompt = first.get('content') if isinstance(first, dict) else None
                    gt = last.get('content') if isinstance(last, dict) else None
                    return prompt, gt
                # Fallback to common field names.
                prompt = _extract_from_record(
                    record, ['prompt', 'question', 'query', 'input', 'problem', 'text']
                )
                gt = _extract_from_record(
                    record, ['answer', 'output', 'label', 'solution', 'target']
                )
                return prompt, gt
            return None, None

        pred_msg = None
        try:
            pred_msg = trajectories[0]['messages'][-1]['content']
        except Exception:
            pred_msg = None
        prompt_msg, gt_msg = _extract_prompt_gt(ground_truths)
        print(
            f"[step {step}] prompt={prompt_msg} | pred={pred_msg} | gt={gt_msg}",
            flush=True,
        )
        ref_logits = None
        if use_ref_model:
            ref_outputs = ref_model.forward_only(inputs=trajectories)
            if callable(ref_outputs) and getattr(ref_outputs, '_is_lazy_collect', False):
                ref_outputs = ref_outputs()
            if isinstance(ref_outputs, list):
                ref_logits = [o['logits'] if isinstance(o, dict) else o.logits for o in ref_outputs]
            else:
                ref_logits = ref_outputs['logits'] if isinstance(ref_outputs, dict) else ref_outputs.logits

        rewards = reward.calculate(trajectories, ground_truths)
        if callable(rewards):
            rewards = rewards()
        for trajectory, reward_value in zip(trajectories, rewards):
            trajectory['rewards'] = reward_value
        if isinstance(rewards, (list, tuple)) and rewards:
            rewards_np = np.array(rewards, dtype=np.float32)
            print(
                f"[step {step}] rewards computed (n={len(rewards)}, "
                f"min={rewards_np.min():.4f}, mean={rewards_np.mean():.4f}, "
                f"max={rewards_np.max():.4f})",
                flush=True,
            )
        else:
            print(f"[step {step}] rewards computed (empty)", flush=True)

        loss = actor_group.forward_backward(trajectories, trajectories, ref_logits)
        if callable(loss):
            loss = loss()
        print(f"[step {step}] loss: {loss}", flush=True)
        actor_group.step()
        actor_group.zero_grad()
        actor_group.lr_step()
        if max_steps and step >= max_steps:
            break

if __name__ == '__main__':
    train()
