# Copyright (c) ModelScope Contributors. All rights reserved.
import os

from transformers import AutoConfig

import twinkle
from twinkle import DeviceMesh, Platform, get_device_placement, get_logger
from twinkle.dataloader import DataLoader
from twinkle.dataset import Dataset, DatasetMeta
from twinkle.model import TransformersModel

logger = get_logger()

MODEL_ID = os.environ.get(
    "MODEL_ID", "ms://Qwen/Qwen3-30B-A3B-Instruct-2507"
)
DATASET_ID = os.environ.get("DATASET_ID", "/path/to/alpaca/dataset")
TEMPLATE_ID = os.environ.get("TEMPLATE_ID", "Template")
STRATEGY = os.environ.get("TRAIN_STRATEGY", "native_fsdp")

WORLD_SIZE = int(os.environ.get("WORLD_SIZE", "2"))
NUM_LAYERS = int(os.environ.get("NUM_LAYERS", "4"))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "4"))
GRAD_ACCUM_STEPS = int(os.environ.get("GRAD_ACCUM_STEPS", "4"))
SAVE_INTERVAL = int(os.environ.get("SAVE_INTERVAL", "50"))


device_mesh = DeviceMesh.from_sizes(
    world_size=WORLD_SIZE,
    fsdp_size=WORLD_SIZE,
    dp_size=1,
    device_type=Platform.get_platform().device_prefix(),
)


twinkle.initialize(
    mode="local",
    global_device_mesh=device_mesh,
)


def train():
    config = AutoConfig.from_pretrained(MODEL_ID, trust_remote_code=True)
    if hasattr(config, "num_hidden_layers"):
        config.num_hidden_layers = min(NUM_LAYERS, config.num_hidden_layers)
    if hasattr(config, "use_cache"):
        config.use_cache = False

    dataset = Dataset(dataset_meta=DatasetMeta(DATASET_ID))
    try:
        dataset.set_template(TEMPLATE_ID, model_id=MODEL_ID)
    except ValueError:
        dataset.set_template("Template", model_id=MODEL_ID)

    processor = "AlpacaProcessor"
    dataset.map(processor)
    dataset.encode(batched=True)
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=BATCH_SIZE,
        device_mesh=device_mesh,
    )

    model = TransformersModel(
        model_id=MODEL_ID,
        config=config,
        strategy=STRATEGY,
        device_mesh=device_mesh,
        fsdp_config={
            "transformer_cls_names_to_wrap": ["Qwen3MoeSparseMoeBlock"],
        },
    )
    model.set_optimizer("AdamW")

    logger.info(get_device_placement())
    logger.info(model.get_train_configs())

    for step, batch in enumerate(dataloader):
        if callable(batch):
            batch = batch()
        model.forward_backward(
            inputs=batch, gradient_accumulation_steps=GRAD_ACCUM_STEPS
        )
        model.clip_grad_and_step(gradient_accumulation_steps=GRAD_ACCUM_STEPS)
        if step % GRAD_ACCUM_STEPS == 0:
            metric = model.calculate_metric(is_training=True)
            if callable(metric):
                metric = metric()
            logger.info(
                f"Current is step {step // GRAD_ACCUM_STEPS}, metric: {metric}"
            )
        if step > 1 and step % SAVE_INTERVAL == 0:
            model.save("./output")


if __name__ == "__main__":
    train()
