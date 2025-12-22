import twinkle

twinkle.initialize(mode='local',
                   library='transformers')


def create_sampler():
    rollout = twinkle.sampler.create()


def create_model(model_id):
    from transformers import AutoModelForCausalLM
    AutoModelForCausalLM = twinkle.prepare_one(AutoModelForCausalLM)
    return AutoModelForCausalLM.from_pretrained(model_id)


def train():
