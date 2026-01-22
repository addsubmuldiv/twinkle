import json
import os
import shutil
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

from tinker import types


DEFAULT_SAVE_DIR = os.environ.get('TWINKLE_DEFAULT_SAVE_DIR', './outputs')


def get_base_dir() -> Path:
    return Path(DEFAULT_SAVE_DIR)


def get_model_dir(model_id: str) -> Path:
    return get_base_dir() / model_id


def get_dir_size(path: Path) -> int:
    total = 0
    if path.exists():
        for p in path.rglob('*'):
            if p.is_file():
                total += p.stat().st_size
    return total


def _read_metadata(model_id: str) -> Dict[str, Any]:
    metadata_path = get_model_dir(model_id) / "tinker_metadata.json"
    if not metadata_path.exists():
        return {}
    try:
        with open(metadata_path, 'r') as f:
            return json.load(f)
    except Exception:
        return {}


def _write_metadata(model_id: str, data: Dict[str, Any]):
    model_dir = get_model_dir(model_id)
    model_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = model_dir / "tinker_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(data, f, indent=2)


def save_train_info(model_id: str, run: types.TrainingRun):
    # Preserve checkpoints if they exist in file but not in run object (which doesn't have checks list)
    current_data = _read_metadata(model_id)
    checkpoints = current_data.get("checkpoints", [])

    new_data = run.model_dump(mode='json')
    # Restore the checkpoints list which is stored alongside run metadata
    new_data["checkpoints"] = checkpoints

    _write_metadata(model_id, new_data)


def get_training_run(model_id: str) -> Optional[types.TrainingRun]:
    data = _read_metadata(model_id)
    if not data:
        return None
    return types.TrainingRun(**data)


def update_train_info(model_id: str, updates: Dict[str, Any]):
    info = _read_metadata(model_id)
    if info:
        info.update(updates)
        _write_metadata(model_id, info)


def save_checkpoint_info(model_id: str, checkpoint: types.Checkpoint):
    info = _read_metadata(model_id)
    if not info:
        return

    checkpoints = info.get("checkpoints", [])
    ckpt_data = checkpoint.model_dump(mode='json')

    # Update existing or append new
    existing_idx = next((i for i, c in enumerate(checkpoints)
                        if c['checkpoint_id'] == checkpoint.checkpoint_id), -1)
    if existing_idx >= 0:
        checkpoints[existing_idx] = ckpt_data
    else:
        checkpoints.append(ckpt_data)

    info['checkpoints'] = checkpoints
    info['last_checkpoint'] = ckpt_data

    _write_metadata(model_id, info)


def get_run_checkpoints(model_id: str) -> Optional[types.CheckpointsListResponse]:
    info = _read_metadata(model_id)
    if not info:
        return None
    checkpoints = [types.Checkpoint(**c) for c in info.get("checkpoints", [])]
    return types.CheckpointsListResponse(checkpoints=checkpoints, cursor=None)


def list_training_runs(limit: int = 20, offset: int = 0) -> types.TrainingRunsResponse:
    base_dir = get_base_dir()
    if not base_dir.exists():
        return types.TrainingRunsResponse(training_runs=[], cursor=types.Cursor(limit=limit, offset=offset, total_count=0))

    candidates = []
    for d in base_dir.iterdir():
        if d.is_dir() and (d / "tinker_metadata.json").exists():
            candidates.append(d)

    candidates.sort(key=lambda d: (
        d / "tinker_metadata.json").stat().st_mtime, reverse=True)

    total = len(candidates)
    selected = candidates[offset: offset + limit]

    runs = []
    for d in selected:
        info = _read_metadata(d.name)
        if info:
            runs.append(types.TrainingRun(**info))

    return types.TrainingRunsResponse(training_runs=runs, cursor=types.Cursor(limit=limit, offset=offset, total_count=total))


def delete_checkpoint_file(model_id: str, checkpoint_id: str) -> bool:
    if ".." in checkpoint_id:
        return False

    model_dir = get_model_dir(model_id)
    ckpt_full_path = model_dir / checkpoint_id

    # Remove files
    if ckpt_full_path.exists():
        if ckpt_full_path.is_dir():
            shutil.rmtree(ckpt_full_path)
        else:
            ckpt_full_path.unlink()

    # Update metadata
    info = _read_metadata(model_id)
    if info:
        checkpoints = info.get("checkpoints", [])
        new_ckpts = [
            c for c in checkpoints if c['checkpoint_id'] != checkpoint_id]
        info['checkpoints'] = new_ckpts

        # If we deleted the "last_checkpoint", reset it
        if info.get('last_checkpoint') and info['last_checkpoint'].get('checkpoint_id') == checkpoint_id:
            info['last_checkpoint'] = new_ckpts[-1] if new_ckpts else None

        _write_metadata(model_id, info)
        return True
    return False
