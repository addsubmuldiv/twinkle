import twinkle
from twinkle.infra import DeviceGroup

device_groups = [
    DeviceGroup(
        name='actor',
        ranks=list(range(0, 4)),
        device_type='GPU',
    ),
    DeviceGroup(
        name='rollout',
        ranks=list(range(4, 6)),
        device_type='GPU',
    ),
    DeviceGroup(
        name='ref',
        ranks=list(range(6, 8)),
        device_type='GPU',
    ),
]


twinkle.initialize(mode='local', groups=device_groups)


def create_model(model_id):
    from transformers import AutoModelForCausalLM
    AutoModelForCausalLM = twinkle.prepare_one(AutoModelForCausalLM)
    return AutoModelForCausalLM.from_pretrained(model_id)


def train():
    dataset =
    rollout = twinkle.sampler.create()