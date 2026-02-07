# Twinkle Client - Megatron LoRA Training Example
#
# This script demonstrates how to fine-tune a language model using LoRA
# through the Twinkle client with a Megatron backend.
# Key difference from the Transformers version:
#   - Loss is computed internally by Megatron (no explicit set_loss call)
#   - Optimizer and LR scheduler use Megatron's built-in defaults
# The server must be running first (see server.py and server_config.yaml).

from peft import LoraConfig

from twinkle import get_device_placement, get_logger
from twinkle.dataset import DatasetMeta
from twinkle_client.dataloader import DataLoader
from twinkle_client.dataset import Dataset
from twinkle_client.model import MultiLoraTransformersModel
from twinkle_client import init_twinkle_client

logger = get_logger()

# Step 1: Initialize the Twinkle client to communicate with the remote server.
# - base_url: the address of the running Twinkle server
# - api_key: your authentication token
client = init_twinkle_client(base_url='http://127.0.0.1:8000', api_key='tml-xxxx')

# Step 2: Query the server for existing training runs and their checkpoints.
# This is useful for resuming a previous training session.
runs = client.list_training_runs()

resume_path = None
for run in runs:
    logger.info(run.model_dump_json(indent=2))
    # List all saved checkpoints for this training run
    checkpoints = client.list_checkpoints(run.training_run_id)

    for checkpoint in checkpoints:
        logger.info(checkpoint.model_dump_json(indent=2))
        # Uncomment the line below to resume from a specific checkpoint:
        # resume_path = checkpoint.twinkle_path


def train():
    # Step 3: Prepare the dataset

    # Load the self-cognition dataset from ModelScope
    dataset = Dataset(dataset_meta=DatasetMeta('ms://swift/self-cognition'))

    # Apply a chat template so the data matches the model's expected input format
    dataset.set_template('Template', model_id='ms://Qwen/Qwen2.5-7B-Instruct', max_length=512)

    # Replace placeholder names in the dataset with custom model/author names
    dataset.map('SelfCognitionProcessor', init_args={'model_name': 'twinkle模型', 'model_author': 'twinkle团队'})

    # Tokenize and encode the dataset into model-ready input features
    dataset.encode(batched=True)

    # Wrap the dataset into a DataLoader that yields batches of size 8
    dataloader = DataLoader(dataset=dataset, batch_size=8)

    # Step 4: Configure the model

    # Create a multi-LoRA model pointing to the base model on ModelScope
    model = MultiLoraTransformersModel(model_id='ms://Qwen/Qwen2.5-7B-Instruct')

    # Define LoRA configuration: apply low-rank adapters to all linear layers
    lora_config = LoraConfig(
        target_modules='all-linear'
    )

    # Attach the LoRA adapter named 'default' to the model.
    # gradient_accumulation_steps=2 means gradients accumulate over 2 micro-batches.
    model.add_adapter_to_model('default', lora_config, gradient_accumulation_steps=2)

    # Set the same chat template used during data preprocessing
    model.set_template('Template')

    # Set the input processor (pads sequences on the right side)
    model.set_processor('InputProcessor', padding_side='right')

    # NOTE: No set_loss() call here — Megatron computes loss internally.

    # Use Megatron's default optimizer with learning rate 1e-4
    model.set_optimizer('default', lr=1e-4)

    # Use Megatron's default LR scheduler with linear decay over 1000 steps
    model.set_lr_scheduler('default', lr_decay_steps=1000, max_lr=1e-4)

    # Step 5: Optionally resume from a previous checkpoint
    if resume_path:
        logger.info(f'Resuming training from {resume_path}')
        model.load(resume_path, load_optimizer=True)

    # Step 6: Run the training loop
    logger.info(model.get_train_configs())

    for step, batch in enumerate(dataloader):
        # Forward pass + backward pass (computes gradients)
        output = model.forward_backward(inputs=batch)

        # Log the loss every 2 steps (aligned with gradient accumulation)
        if step % 2 == 0:
            logger.info(f'Current is step {step // 2}, loss: {output}')

        # Clip gradients to prevent exploding gradients (max norm = 1.0)
        model.clip_grad_norm(1.0)

        # Perform one optimizer step (update model weights)
        model.step()

        # Reset gradients to zero for the next iteration
        model.zero_grad()

        # Advance the learning rate scheduler by one step
        model.lr_step()

        # Save a checkpoint every 8 steps for fault tolerance
        if step > 0 and step % 8 == 0:
            logger.info(f'Saving checkpoint at step {step}')
            model.save(f'step-{step}', save_optimizer=True)


if __name__ == '__main__':
    train()
