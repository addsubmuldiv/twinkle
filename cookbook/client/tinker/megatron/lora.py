# Tinker-Compatible Client - Megatron LoRA Training & Sampling Example
#
# This script demonstrates end-to-end LoRA fine-tuning and inference using the
# Tinker-compatible client API with a Megatron backend.
# It covers: connecting to the server, preparing data manually with tokenizers,
# running a training loop, saving checkpoints, and sampling from the model.
# The server must be running first (see server.py and server_config.yaml).

from twinkle_client import init_tinker_compat_client

# Step 1: Initialize the Tinker-compatible client to communicate with the server.
service_client = init_tinker_compat_client(base_url='http://localhost:8000')

# Step 2: List models available on the server to verify the connection
print("Available models:")
for item in service_client.get_server_capabilities().supported_models:
    print("- " + item.model_name)


# Step 3: Create a REST client for querying training runs and checkpoints.
# This is useful for inspecting previous training sessions or resuming training.
rest_client = service_client.create_rest_client()

future = rest_client.list_training_runs(limit=50)
response = future.result()

# You can resume from a twinkle:// path. Example:
# resume_path = "twinkle://20260131_170251-Qwen_Qwen2_5-0_5B-Instruct-7275126c/weights/pig-latin-lora-epoch-1"
resume_path = ""

print(f"Found {len(response.training_runs)} training runs")
for tr in response.training_runs:
    print(tr.model_dump_json(indent=2))
    
    chpts = rest_client.list_checkpoints(tr.training_run_id).result()
    for chpt in chpts.checkpoints:
        print("  " + chpt.model_dump_json(indent=2))
        # Uncomment the line below to resume from the last checkpoint:
        # resume_path = chpt.tinker_path
    
# Step 4: Create or resume a training client.
# If resume_path is set, it restores both model weights and optimizer state.
base_model = "Qwen/Qwen2.5-0.5B-Instruct"
if not resume_path:
    training_client = service_client.create_lora_training_client(
        base_model=base_model
    )
else:
    training_client = service_client.create_training_client_from_state_with_optimizer(path=resume_path)

# Step 5: Prepare training data manually
#
# This example teaches the model to translate English into Pig Latin.
# Each example has an "input" (English phrase) and "output" (Pig Latin).
examples = [
    {"input": "banana split", "output": "anana-bay plit-say"},
    {"input": "quantum physics", "output": "uantum-qay ysics-phay"},
    {"input": "donut shop", "output": "onut-day op-shay"},
    {"input": "pickle jar", "output": "ickle-pay ar-jay"},
    {"input": "space exploration", "output": "ace-spay exploration-way"},
    {"input": "rubber duck", "output": "ubber-ray uck-day"},
    {"input": "coding wizard", "output": "oding-cay izard-way"},
]
 
from tinker import types
from modelscope import AutoTokenizer

# Load the tokenizer locally (avoids a network call to HuggingFace)
tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
 
def process_example(example: dict, tokenizer) -> types.Datum:
    """Convert a raw example dict into a Datum suitable for the training API.

    The Datum contains:
      - model_input: the token IDs fed into the LLM
      - loss_fn_inputs: target tokens and per-token weights (0 = ignore, 1 = train)
    """
    # Build a simple prompt template
    prompt = f"English: {example['input']}\nPig Latin:"
 
    # Tokenize the prompt; weights=0 means the loss ignores these tokens
    prompt_tokens = tokenizer.encode(prompt, add_special_tokens=True)
    prompt_weights = [0] * len(prompt_tokens)

    # Tokenize the completion; weights=1 means the loss is computed on these tokens
    completion_tokens = tokenizer.encode(f" {example['output']}\n\n", add_special_tokens=False)
    completion_weights = [1] * len(completion_tokens)
 
    # Concatenate prompt + completion
    tokens = prompt_tokens + completion_tokens
    weights = prompt_weights + completion_weights
 
    # Shift by one: input is tokens[:-1], target is tokens[1:] (next-token prediction)
    input_tokens = tokens[:-1]
    target_tokens = tokens[1:]
    weights = weights[1:]
 
    return types.Datum(
        model_input=types.ModelInput.from_ints(tokens=input_tokens),
        loss_fn_inputs=dict(weights=weights, target_tokens=target_tokens)
    )
 
# Process all examples into Datum objects
processed_examples = [process_example(ex, tokenizer) for ex in examples]
 
# Visualize the first example to verify tokenization and weight alignment
datum0 = processed_examples[0]
print(f"{'Input':<20} {'Target':<20} {'Weight':<10}")
print("-" * 50)
for i, (inp, tgt, wgt) in enumerate(zip(datum0.model_input.to_ints(), datum0.loss_fn_inputs['target_tokens'].tolist(), datum0.loss_fn_inputs['weights'].tolist())):
    print(f"{repr(tokenizer.decode([inp])):<20} {repr(tokenizer.decode([tgt])):<20} {wgt:<10}")
    
# Step 6: Run the training loop
#
# For each epoch, iterate over multiple batches:
#   - forward_backward: sends data to the server, computes loss & gradients
#   - optim_step: updates model weights using Adam optimizer
import numpy as np
for epoch in range(2):
    for batch in range(5):
        # Send training data and get back logprobs (asynchronous futures)
        fwdbwd_future = training_client.forward_backward(processed_examples, "cross_entropy")
        optim_future = training_client.optim_step(types.AdamParams(learning_rate=1e-4))
    
        # Wait for results from the server
        fwdbwd_result = fwdbwd_future.result()
        optim_result = optim_future.result()
    
        # Compute the weighted average log-loss per token for monitoring
        print(f"Epoch {epoch}, Batch {batch}: ", end="")
        logprobs = np.concatenate([output['logprobs'].tolist() for output in fwdbwd_result.loss_fn_outputs])
        weights = np.concatenate([example.loss_fn_inputs['weights'].tolist() for example in processed_examples])
        print(f"Loss per token: {-np.dot(logprobs, weights) / weights.sum():.4f}")

    # Save checkpoint (model weights + optimizer state) after each epoch
    save_future = training_client.save_state(f"pig-latin-lora-epoch-{epoch}")
    save_result = save_future.result()
    print(f"Saved checkpoint for epoch {epoch} to {save_result.path}")

# Step 7: Sample from the trained model
#
# Save the current weights and create a sampling client to generate text.
sampling_client = training_client.save_weights_and_get_sampling_client(name='pig-latin-model')
 
# Prepare a prompt and sampling parameters
prompt = types.ModelInput.from_ints(tokenizer.encode("English: coffee break\nPig Latin:"))
params = types.SamplingParams(
    max_tokens=20,       # Maximum number of tokens to generate
    temperature=0.0,     # Greedy sampling (deterministic)
    stop=["\n"]          # Stop at newline
)

# Generate 8 completions and print the results
future = sampling_client.sample(prompt=prompt, sampling_params=params, num_samples=8)
result = future.result()
print("Responses:")
for i, seq in enumerate(result.sequences):
    print(f"{i}: {repr(tokenizer.decode(seq.tokens))}")
