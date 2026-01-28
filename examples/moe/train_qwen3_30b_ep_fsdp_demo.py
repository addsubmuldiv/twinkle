# Copyright (c) ModelScope Contributors. All rights reserved.
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.distributed as dist
from transformers import AutoConfig, AutoModelForCausalLM, PretrainedConfig

from twinkle.model.moe import apply_expert_parallel
from twinkle.model.transformers.strategy import NativeFSDPStrategy
from twinkle.utils import DeviceMesh


def _init_dist(args) -> tuple[int, int, torch.device]:
    rank_env = os.environ.get("RANK")
    world_size_env = os.environ.get("WORLD_SIZE")
    local_rank_env = os.environ.get("LOCAL_RANK")

    rank = int(rank_env) if rank_env is not None else args.rank
    world_size = int(
        world_size_env) if world_size_env is not None else args.world_size
    local_rank = int(
        local_rank_env) if local_rank_env is not None else args.local_rank

    if world_size > 1 and rank_env is None and args.rank == 0 and world_size_env is None:
        raise RuntimeError(
            "Distributed training requires torchrun or env:// rendezvous. "
            "Please launch with: torchrun --nproc_per_node <N> examples/moe/train_qwen3_30b_ep_fsdp_demo.py ..."
        )

    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    if world_size > 1:
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            rank=rank,
            world_size=world_size,
            device_id=device,
        )
    else:
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            rank=0,
            world_size=1,
            device_id=device,
        )
    return rank, world_size, device


def _load_qwen3_config(model_id: str, local_files_only: bool):
    return AutoConfig.from_pretrained(
        model_id,
        trust_remote_code=True,
        local_files_only=local_files_only,
    )


def _build_ep_device_mesh(world_size: int, ep_size: int) -> DeviceMesh:
    if world_size % ep_size != 0:
        raise ValueError(
            f"world_size({world_size}) must be divisible by ep_size({ep_size})."
        )
    dp_size = world_size // ep_size
    mesh = np.arange(world_size).reshape(dp_size, ep_size)
    return DeviceMesh(
        device_type="cuda",
        mesh=mesh,
        mesh_dim_names=("dp", "ep"),
    )


def _maybe_set_num_layers(config, num_layers: Optional[int]) -> None:
    if num_layers is None:
        return
    if hasattr(config, "num_hidden_layers"):
        config.num_hidden_layers = int(num_layers)


def main():
    parser = argparse.ArgumentParser(
        description="Qwen3-30B EP+FSDP2 training demo")
    parser.add_argument("--model-id", type=str,
                        required=True, help="Local path or HF ID")
    parser.add_argument("--local-files-only",
                        action="store_true", help="Only load local files")
    parser.add_argument("--num-layers", type=int, default=1,
                        help="Override num_hidden_layers (default: 1)")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=16)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--ep-size", type=int, default=None,
                        help="Enable EP when > 1")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--rank", type=int, default=0,
                        help="Rank for non-torchrun launches")
    parser.add_argument("--world-size", type=int, default=1,
                        help="World size for non-torchrun launches")
    parser.add_argument("--local-rank", type=int, default=0,
                        help="Local rank for non-torchrun launches")
    args = parser.parse_args()

    rank, world_size, device = _init_dist(args)
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True

    config = _load_qwen3_config(args.model_id, args.local_files_only)
    _maybe_set_num_layers(config, args.num_layers)
    if hasattr(config, "use_cache"):
        config.use_cache = False

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        config=config,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        local_files_only=args.local_files_only,
    ).to(device)
    model.train()

    if args.ep_size is not None and args.ep_size > 1:
        device_mesh = _build_ep_device_mesh(world_size, args.ep_size)
        apply_expert_parallel(
            model.model,
            device_mesh,
            config={
                "enabled": True,
                "router_dtype": "fp32",
                "all_to_all": "torch",
                "keep_router_logits": False,
            },
        )
    else:
        device_mesh = DeviceMesh(
            device_type="cuda",
            mesh=np.arange(world_size),
            mesh_dim_names=("fsdp",),
        )

    strategy = NativeFSDPStrategy(
        device_mesh=device_mesh, mixed_precision="bf16", fsdp_config={})
    model.model, _ = strategy.wrap_model(model.model, optimizer=None)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, foreach=False)

    vocab_size = model.config.vocab_size
    input_ids = torch.randint(
        low=0,
        high=vocab_size,
        size=(args.batch_size, args.seq_len),
        device=device,
    )

    for step in range(1, args.steps + 1):
        torch.cuda.reset_peak_memory_stats(device)
        outputs = model(input_ids=input_ids, labels=input_ids)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        loss_val = loss.detach()
        try:
            from torch.distributed.tensor import DTensor  # type: ignore

            if isinstance(loss_val, DTensor):
                loss_val = loss_val.to_local()
        except Exception:
            pass
        loss_val = loss_val.float().to(device)
        dist.all_reduce(loss_val, op=dist.ReduceOp.SUM)
        loss_val = loss_val / world_size
        if rank == 0:
            print(f"[step {step}] loss={loss_val.item():.6f}")

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
