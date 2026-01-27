# Copyright (c) ModelScope Contributors. All rights reserved.
import json
import os
import socket
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_PATH = str(REPO_ROOT / "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

import unittest
from typing import Dict, List

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn
from transformers import AutoConfig, AutoModelForCausalLM, PretrainedConfig

from twinkle.model.moe import apply_expert_parallel
from twinkle.model.transformers.strategy import NativeFSDPStrategy
from twinkle.utils import DeviceMesh


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _ensure_ep_device_mesh(device_mesh: DeviceMesh) -> None:
    if not hasattr(device_mesh, "get_dim_group"):
        device_mesh.get_dim_group = lambda dim: device_mesh.create_process_group([dim])
    if not hasattr(device_mesh, "ep_world_size"):
        device_mesh.ep_world_size = device_mesh.get_dim_size("ep")
    if not hasattr(device_mesh, "ep_rank"):
        device_mesh.ep_rank = device_mesh._get_rank_for_dim("ep")


def _find_moe_blocks(model: nn.Module) -> List[nn.Module]:
    blocks = []
    for module in model.modules():
        experts = getattr(module, "experts", None)
        if not isinstance(experts, nn.ModuleList):
            continue
        gate = getattr(module, "gate", None) or getattr(module, "router", None)
        if gate is None:
            continue
        blocks.append(module)
    return blocks


def _capture_router_logits(model: nn.Module):
    router_logits: List[torch.Tensor] = []
    handles = []
    for block in _find_moe_blocks(model):
        gate = getattr(block, "gate", None) or getattr(block, "router", None)
        if gate is None:
            continue
        def _hook(module, inputs, output):
            router_logits.append(output.detach())
        handles.append(gate.register_forward_hook(_hook))
    return router_logits, handles


def _collect_baseline_local_expert_grads(
    block: nn.Module,
    ep_rank: int,
    ep_world_size: int,
    ep_group,
) -> Dict[int, Dict[str, torch.Tensor]]:
    num_experts = len(block.experts)
    if num_experts % ep_world_size != 0:
        raise ValueError(
            f"num_experts ({num_experts}) must be divisible by ep_world_size ({ep_world_size})."
        )
    experts_per_rank = num_experts // ep_world_size
    local_start = ep_rank * experts_per_rank
    local_end = local_start + experts_per_rank
    local_grads: Dict[int, Dict[str, torch.Tensor]] = {}

    for global_idx, expert in enumerate(block.experts):
        param_grads: Dict[str, torch.Tensor] = {}
        for name, param in expert.named_parameters():
            grad = param.grad
            if grad is None:
                grad = torch.zeros_like(param, dtype=param.dtype)
            dist.all_reduce(grad, op=dist.ReduceOp.SUM, group=ep_group)
            if local_start <= global_idx < local_end:
                param_grads[name] = grad.detach().cpu()
        if local_start <= global_idx < local_end:
            local_grads[global_idx] = param_grads

    return local_grads


def _load_qwen3_moe_config(model_id: str, local_files_only: bool):
    try:
        return AutoConfig.from_pretrained(
            model_id,
            trust_remote_code=True,
            local_files_only=local_files_only,
        )
    except Exception as exc:  # noqa: BLE001
        config_path = Path(model_id) / "config.json"
        if not config_path.exists():
            raise exc
        with config_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        if "model_type" not in data:
            data["model_type"] = "qwen3_moe"
        if "architectures" not in data:
            data["architectures"] = ["Qwen3MoeForCausalLM"]
        try:
            return AutoConfig.from_dict(data)
        except Exception:
            return PretrainedConfig.from_dict(data)



def _load_qwen3_moe_pretrained(model_id: str, local_files_only: bool, device: torch.device) -> nn.Module:
    config = _load_qwen3_moe_config(model_id, local_files_only)
    if hasattr(config, "num_hidden_layers"):
        config.num_hidden_layers = 1
    if hasattr(config, "use_cache"):
        config.use_cache = False
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        config=config,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        local_files_only=local_files_only,
    )
    model.to(device)
    model.eval()
    return model


def _run_worker_ep_fsdp_pretrained(rank: int, world_size: int, port: int, model_id: str, local_files_only: bool):
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    if not torch.cuda.is_available():
        raise RuntimeError("This test requires CUDA (4 GPUs).")
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "1"
    dist.init_process_group(
        backend="nccl",
        rank=rank,
        world_size=world_size,
        init_method=f"tcp://127.0.0.1:{port}",
        device_id=device,
    )
    dist.barrier()

    try:
        torch.manual_seed(1234)
        model = _load_qwen3_moe_pretrained(model_id, local_files_only, device)
        input_ids = torch.randint(
            low=0,
            high=model.config.vocab_size,
            size=(2, 8),
            device=device,
        )

        baseline_router_logits, baseline_handles = _capture_router_logits(model.model)
        baseline_out = model(input_ids=input_ids).logits
        for handle in baseline_handles:
            handle.remove()
        baseline_out_ref = baseline_out.detach()
        baseline_out.sum().backward()

        device_mesh = DeviceMesh(
            device_type="cuda",
            mesh=np.arange(world_size).reshape(2, 2),
            mesh_dim_names=("fsdp", "ep"),
        )
        _ensure_ep_device_mesh(device_mesh)
        ep_group = device_mesh.get_dim_group("ep")

        baseline_blocks = _find_moe_blocks(model.model)
        if not baseline_blocks:
            raise RuntimeError("No MoE blocks found in Qwen3 model.")

        baseline_block_grads = []
        for block in baseline_blocks:
            baseline_block_grads.append(
                _collect_baseline_local_expert_grads(
                    block,
                    device_mesh.ep_rank,
                    device_mesh.ep_world_size,
                    ep_group,
                )
            )

        model.zero_grad(set_to_none=True)

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

        strategy = NativeFSDPStrategy(device_mesh=device_mesh, mixed_precision="bf16", fsdp_config={})
        model.model, _ = strategy.wrap_model(model.model, optimizer=None)

        ep_router_logits, ep_handles = _capture_router_logits(model.model)
        ep_out = model(input_ids=input_ids).logits
        for handle in ep_handles:
            handle.remove()

        out_diff = (ep_out - baseline_out_ref).abs()
        if not torch.allclose(ep_out, baseline_out_ref, rtol=1e-3, atol=1e-4):
            print(
                f"[rank{rank}] ep_out diff mean={out_diff.mean().item():.6e} "
                f"max={out_diff.max().item():.6e}"
            )
        assert torch.allclose(ep_out, baseline_out_ref, rtol=1e-3, atol=1e-4)

        if baseline_router_logits and ep_router_logits:
            for idx, (base_logits, ep_logits) in enumerate(
                zip(baseline_router_logits, ep_router_logits)
            ):
                logits_diff = (ep_logits - base_logits).abs()
                if not torch.allclose(ep_logits, base_logits, rtol=1e-3, atol=1e-4):
                    print(
                        f"[rank{rank}] router_logits[{idx}] diff "
                        f"mean={logits_diff.mean().item():.6e} "
                        f"max={logits_diff.max().item():.6e}"
                    )
        else:
            print(f"[rank{rank}] router_logits not captured for comparison.")

        ep_out.sum().backward()

        ep_blocks = _find_moe_blocks(model.model)
        assert len(ep_blocks) == len(baseline_block_grads)

        for block_idx, ep_block in enumerate(ep_blocks):
            baseline_grads = baseline_block_grads[block_idx]
            printed_grad_diff = False
            for local_idx, expert in enumerate(ep_block.experts):
                global_idx = ep_block._ep_local_start + local_idx
                baseline_params = baseline_grads[global_idx]
                for name, param in expert.named_parameters():
                    baseline_grad = baseline_params[name]
                    ep_grad = param.grad
                    if ep_grad is None:
                        assert torch.allclose(
                            baseline_grad,
                            torch.zeros_like(baseline_grad),
                            rtol=1e-5,
                            atol=1e-6,
                        )
                    else:
                        base = baseline_grad.to(ep_grad.device, dtype=torch.float32)
                        diff = (ep_grad.to(torch.float32) - base)
                        rel = diff.norm() / (base.norm() + 1e-12)
                        if rel.item() > 1e-3 and not printed_grad_diff:
                            abs_diff = diff.abs()
                            base_norm = base.norm().item()
                            ep_norm = ep_grad.norm().item()
                            ratio = ep_norm / base_norm if base_norm != 0 else float("inf")
                            print(
                                f"[rank{rank}] expert{global_idx}.{name} grad diff "
                                f"mean={abs_diff.mean().item():.6e} max={abs_diff.max().item():.6e} "
                                f"ep_norm={ep_norm:.6e} base_norm={base_norm:.6e} ratio={ratio:.6e} "
                                f"rel_norm={rel.item():.6e}"
                            )
                            printed_grad_diff = True
                        assert rel.item() <= 1e-3
    finally:
        dist.destroy_process_group()


class TestExpertParallelFSDPPretrained(unittest.TestCase):
    def test_qwen3_moe_pretrained_ep_fsdp(self):
        if not dist.is_available():
            self.skipTest("torch.distributed is not available")
        if not torch.cuda.is_available():
            self.skipTest("CUDA is required for this test.")
        world_size = 4
        if torch.cuda.device_count() < world_size:
            self.skipTest("Requires at least 4 GPUs for EP+FSDP test.")
        model_id = os.environ.get("QWEN3_MOE_MODEL_ID", "Qwen/Qwen3-30B-A3B-Instruct-2507")
        local_files_only = os.environ.get("QWEN3_MOE_LOCAL_ONLY", "1") != "0"
        try:
            _load_qwen3_moe_config(model_id, local_files_only)
        except Exception as exc:  # noqa: BLE001
            self.skipTest(f"Qwen3 model not available locally: {exc}")
        port = _find_free_port()
        mp.spawn(
            _run_worker_ep_fsdp_pretrained,
            args=(world_size, port, model_id, local_files_only),
            nprocs=world_size,
            join=True,
        )
