# Copyright (c) ModelScope Contributors. All rights reserved.
from datetime import datetime
import json
import os
import re
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel


TWINKLE_DEFAULT_SAVE_DIR = os.environ.get('TWINKLE_DEFAULT_SAVE_DIR', './outputs')

TRAIN_RUN_INFO_FILENAME = 'twinkle_metadata.json'
CHECKPOINT_INFO_FILENAME = 'checkpoint_metadata.json'


# ----- Pydantic Models for Twinkle -----

class Cursor(BaseModel):
    limit: int
    offset: int
    total_count: int


class Checkpoint(BaseModel):
    checkpoint_id: str
    checkpoint_type: str
    time: datetime
    twinkle_path: str
    size_bytes: int
    public: bool = False


class TrainingRun(BaseModel):
    training_run_id: str
    base_model: str
    model_owner: str
    is_lora: bool = False
    corrupted: bool = False
    lora_rank: Optional[int] = None
    last_request_time: Optional[datetime] = None
    last_checkpoint: Optional[Dict[str, Any]] = None
    last_sampler_checkpoint: Optional[Dict[str, Any]] = None
    user_metadata: Optional[Dict[str, Any]] = None


class TrainingRunsResponse(BaseModel):
    training_runs: List[TrainingRun]
    cursor: Cursor


class CheckpointsListResponse(BaseModel):
    checkpoints: List[Checkpoint]
    cursor: Optional[Cursor] = None


class ParsedCheckpointTwinklePath(BaseModel):
    twinkle_path: str
    training_run_id: str
    checkpoint_type: str
    checkpoint_id: str


class WeightsInfoResponse(BaseModel):
    training_run_id: str
    base_model: str
    model_owner: str
    is_lora: bool = False
    lora_rank: Optional[int] = None


class LoraConfig(BaseModel):
    rank: int = 8
    train_unembed: bool = False
    train_mlp: bool = True
    train_attn: bool = True


class CreateModelRequest(BaseModel):
    base_model: str
    lora_config: Optional[LoraConfig] = None
    user_metadata: Optional[Dict[str, Any]] = None


# ----- Permission Control -----

def validate_user_path(token: str, path: str) -> bool:
    """
    Validate that the path is safe and belongs to the user.
    
    This function checks:
    1. Path doesn't contain '..' (directory traversal attack prevention)
    2. Path doesn't start with '/' (absolute path prevention)
    3. Path doesn't contain null bytes
    4. Path components are reasonable
    
    Args:
        token: User's authentication token (used to identify ownership)
        path: The path to validate
        
    Returns:
        True if path is safe, False otherwise
    """
    if not path:
        return False
    
    # Check for directory traversal attempts
    if '..' in path:
        return False
    
    # Check for null bytes (security vulnerability)
    if '\x00' in path:
        return False
    
    # Check for suspicious patterns
    suspicious_patterns = [
        r'\.\./',      # Directory traversal
        r'/\.\.', 
        r'^/',         # Absolute path
        r'^\.\.',      # Starts with ..
        r'~',          # Home directory expansion
    ]
    for pattern in suspicious_patterns:
        if re.search(pattern, path):
            return False
    
    return True


def validate_ownership(token: str, model_owner: str) -> bool:
    """
    Validate that the user owns the resource.
    
    Args:
        token: User's authentication token
        model_owner: The owner of the model/checkpoint
        
    Returns:
        True if user owns the resource, False otherwise
    """
    if not token or not model_owner:
        return False
    return token == model_owner


class FileManager:
    """Base file manager with common utilities."""
    
    @staticmethod
    def get_dir_size(path: Path) -> int:
        """Calculate total size of files in a directory."""
        total = 0
        if path.exists():
            for p in path.rglob('*'):
                if p.is_file():
                    total += p.stat().st_size
        return total


class TrainingRunManager(FileManager):
    """Manager for training run metadata and operations."""

    @staticmethod
    def get_base_dir() -> Path:
        return Path(TWINKLE_DEFAULT_SAVE_DIR)

    @staticmethod
    def get_model_dir(model_id: str) -> Path:
        return TrainingRunManager.get_base_dir() / model_id

    @staticmethod
    def _read_info(model_id: str) -> Dict[str, Any]:
        """Read training run metadata from disk."""
        metadata_path = TrainingRunManager.get_model_dir(model_id) / TRAIN_RUN_INFO_FILENAME
        if not metadata_path.exists():
            return {}
        try:
            with open(metadata_path, 'r') as f:
                return json.load(f)
        except Exception:
            return {}

    @staticmethod
    def _write_info(model_id: str, data: Dict[str, Any]):
        """Write training run metadata to disk."""
        model_dir = TrainingRunManager.get_model_dir(model_id)
        model_dir.mkdir(parents=True, exist_ok=True)
        metadata_path = model_dir / TRAIN_RUN_INFO_FILENAME
        with open(metadata_path, 'w') as f:
            json.dump(data, f, indent=2)

    @classmethod
    def save(cls, model_id: str, run_config: CreateModelRequest, token: str):
        """
        Save training run metadata.
        
        Args:
            model_id: Unique identifier for the model
            run_config: Configuration for the training run
            token: User's authentication token (becomes model_owner)
        """
        lora_config = run_config.lora_config
        train_run_data = TrainingRun(
            training_run_id=model_id,
            base_model=run_config.base_model,
            model_owner=token,
            is_lora=True if lora_config else False,
            corrupted=False,
            lora_rank=lora_config.rank if lora_config else None,
            last_request_time=datetime.now(),
            last_checkpoint=None,
            last_sampler_checkpoint=None,
            user_metadata=run_config.user_metadata
        )

        new_data = train_run_data.model_dump(mode='json')
        # Store lora config details separately if needed
        if lora_config:
            new_data['train_unembed'] = lora_config.train_unembed
            new_data['train_mlp'] = lora_config.train_mlp
            new_data['train_attn'] = lora_config.train_attn

        cls._write_info(model_id, new_data)

    @classmethod
    def get(cls, model_id: str) -> Optional[TrainingRun]:
        """Get training run metadata by model_id."""
        data = cls._read_info(model_id)
        if not data:
            return None
        return TrainingRun(**data)

    @classmethod
    def get_with_permission(cls, model_id: str, token: str) -> Optional[TrainingRun]:
        """
        Get training run metadata with ownership validation.
        
        Args:
            model_id: The model identifier
            token: User's authentication token
            
        Returns:
            TrainingRun if found and user owns it, None otherwise
        """
        run = cls.get(model_id)
        if run and validate_ownership(token, run.model_owner):
            return run
        return None

    @classmethod
    def update(cls, model_id: str, updates: Dict[str, Any]):
        """Update training run metadata."""
        info = cls._read_info(model_id)
        if info:
            info.update(updates)
            cls._write_info(model_id, info)

    @classmethod
    def update_with_permission(cls, model_id: str, updates: Dict[str, Any], token: str) -> bool:
        """
        Update training run metadata with ownership validation.
        
        Returns:
            True if update succeeded, False if permission denied
        """
        run = cls.get(model_id)
        if run and validate_ownership(token, run.model_owner):
            cls.update(model_id, updates)
            return True
        return False

    @classmethod
    def list_runs(cls, limit: int = 20, offset: int = 0, 
                  token: Optional[str] = None) -> TrainingRunsResponse:
        """
        List training runs with optional filtering by owner.
        
        Args:
            limit: Maximum number of results
            offset: Offset for pagination
            token: If provided, only return runs owned by this user
        """
        base_dir = cls.get_base_dir()
        if not base_dir.exists():
            return TrainingRunsResponse(
                training_runs=[],
                cursor=Cursor(limit=limit, offset=offset, total_count=0)
            )

        candidates = []
        for d in base_dir.iterdir():
            if d.is_dir() and (d / TRAIN_RUN_INFO_FILENAME).exists():
                candidates.append(d)

        candidates.sort(
            key=lambda d: (d / TRAIN_RUN_INFO_FILENAME).stat().st_mtime,
            reverse=True
        )

        # Filter by owner if token provided
        runs = []
        for d in candidates:
            run = cls.get(d.name)
            if run:
                if token is None or validate_ownership(token, run.model_owner):
                    runs.append(run)

        total = len(runs)
        selected = runs[offset:offset + limit]

        return TrainingRunsResponse(
            training_runs=selected,
            cursor=Cursor(limit=limit, offset=offset, total_count=total)
        )


class CheckpointManager(FileManager):
    """Manager for checkpoint metadata and operations with permission control."""

    @staticmethod
    def get_ckpt_dir(model_id: str, checkpoint_id: str) -> Path:
        return TrainingRunManager.get_model_dir(model_id) / checkpoint_id

    @staticmethod
    def get_save_dir(model_id: str, is_sampler: bool = False) -> str:
        """Get the directory path for saving weights."""
        weights_type = 'sampler_weights' if is_sampler else 'weights'
        checkpoint_id = Path(model_id) / weights_type
        save_path = Path(TWINKLE_DEFAULT_SAVE_DIR) / checkpoint_id
        return save_path.as_posix()

    @staticmethod
    def get_ckpt_name(name: Optional[str]) -> str:
        """Generate or normalize checkpoint name."""
        if name:
            # Normalize name to avoid issues with filesystem
            name = re.sub(r'[^\w\-]', '_', name)
            return name
        return datetime.now().strftime('%Y%m%d_%H%M%S')

    @classmethod
    def _read_ckpt_info(cls, model_id: str, checkpoint_id: str) -> Optional[Dict[str, Any]]:
        """Read checkpoint metadata from disk."""
        meta_path = cls.get_ckpt_dir(model_id, checkpoint_id) / CHECKPOINT_INFO_FILENAME
        if not meta_path.exists():
            return None
        try:
            with open(meta_path, 'r') as f:
                return json.load(f)
        except Exception:
            return None

    @classmethod
    def _write_ckpt_info(cls, model_id: str, checkpoint_id: str, data: Dict[str, Any]):
        """Write checkpoint metadata to disk."""
        ckpt_dir = cls.get_ckpt_dir(model_id, checkpoint_id)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        meta_path = ckpt_dir / CHECKPOINT_INFO_FILENAME
        with open(meta_path, 'w') as f:
            json.dump(data, f, indent=2)

    @classmethod
    def save(cls, model_id: str, name: str, token: str, 
             is_sampler: bool = False, public: bool = False) -> str:
        """
        Save checkpoint metadata with ownership tracking.
        
        Args:
            model_id: The model identifier
            name: Checkpoint name
            token: User's authentication token
            is_sampler: Whether this is a sampler checkpoint
            public: Whether the checkpoint is public
            
        Returns:
            The twinkle path for the checkpoint
        """
        # Validate path safety
        if not validate_user_path(token, name):
            raise ValueError(f"Invalid checkpoint name: {name}")
        
        weights_type = 'sampler_weights' if is_sampler else 'weights'
        checkpoint_type = 'sampler' if is_sampler else 'training'
        checkpoint_id = f'{weights_type}/{name}'
        twinkle_path = f"twinkle://{model_id}/{checkpoint_id}"
        checkpoint_path = cls.get_ckpt_dir(model_id, checkpoint_id)
        
        checkpoint = Checkpoint(
            checkpoint_id=checkpoint_id,
            checkpoint_type=checkpoint_type,
            time=datetime.now(),
            twinkle_path=twinkle_path,
            size_bytes=cls.get_dir_size(checkpoint_path),
            public=public
        )
        ckpt_data = checkpoint.model_dump(mode='json')
        cls._write_ckpt_info(model_id, checkpoint.checkpoint_id, ckpt_data)

        # Update last_checkpoint in run info
        TrainingRunManager.update(model_id, {'last_checkpoint': ckpt_data})
        return twinkle_path

    @classmethod
    def get(cls, model_id: str, checkpoint_id: str) -> Optional[Checkpoint]:
        """Get checkpoint metadata."""
        data = cls._read_ckpt_info(model_id, checkpoint_id)
        if not data:
            return None
        return Checkpoint(**data)

    @classmethod
    def list_checkpoints(cls, model_id: str, token: Optional[str] = None) -> Optional[CheckpointsListResponse]:
        """
        List checkpoints for a training run.
        
        Args:
            model_id: The model identifier
            token: If provided, validate ownership before listing
        """
        # Validate ownership if token provided
        if token:
            run = TrainingRunManager.get(model_id)
            if not run or not validate_ownership(token, run.model_owner):
                return None
        
        run_dir = TrainingRunManager.get_model_dir(model_id)
        if not run_dir.exists():
            return None

        checkpoints: List[Checkpoint] = []
        # Iterate over weights and sampler_weights directories
        for weights_type in ["weights", "sampler_weights"]:
            type_dir = run_dir / weights_type
            if not type_dir.exists() or not type_dir.is_dir():
                continue
            for d in type_dir.iterdir():
                if d.is_dir() and (d / CHECKPOINT_INFO_FILENAME).exists():
                    checkpoint_id = f"{weights_type}/{d.name}"
                    ckpt = cls.get(model_id, checkpoint_id)
                    if ckpt:
                        checkpoints.append(ckpt)

        # Sort by creation time
        checkpoints.sort(key=lambda x: x.time)

        return CheckpointsListResponse(checkpoints=checkpoints, cursor=None)

    @classmethod
    def delete(cls, model_id: str, checkpoint_id: str, token: str) -> bool:
        """
        Delete a checkpoint with ownership validation.
        
        Args:
            model_id: The model identifier
            checkpoint_id: The checkpoint identifier
            token: User's authentication token
            
        Returns:
            True if deleted successfully, False if permission denied or not found
        """
        # Basic safety check to prevent directory traversal
        if '..' in checkpoint_id:
            return False
        
        # Validate ownership
        run = TrainingRunManager.get(model_id)
        if not run or not validate_ownership(token, run.model_owner):
            return False

        ckpt_dir = cls.get_ckpt_dir(model_id, checkpoint_id)

        if ckpt_dir.exists():
            if ckpt_dir.is_dir():
                shutil.rmtree(ckpt_dir)
            else:
                ckpt_dir.unlink()

            # Update last_checkpoint in run info
            all_ckpts = cls.list_checkpoints(model_id)
            last_ckpt = all_ckpts.checkpoints[-1] if all_ckpts and all_ckpts.checkpoints else None
            TrainingRunManager.update(
                model_id, {
                    'last_checkpoint': last_ckpt.model_dump(mode='json') if last_ckpt else None
                }
            )
            return True
        return False

    @classmethod
    def parse_twinkle_path(cls, twinkle_path: str) -> Optional[ParsedCheckpointTwinklePath]:
        """Parse a twinkle:// path into its components."""
        if not twinkle_path.startswith("twinkle://"):
            return None
        parts = twinkle_path[10:].split("/")
        if len(parts) != 3:
            return None
        if parts[1] not in ["weights", "sampler_weights"]:
            return None
        checkpoint_type = "training" if parts[1] == "weights" else "sampler"
        return ParsedCheckpointTwinklePath(
            twinkle_path=twinkle_path,
            training_run_id=parts[0],
            checkpoint_type=checkpoint_type,
            checkpoint_id="/".join(parts[1:]),
        )

    @classmethod
    def get_weights_info(cls, checkpoint_path: str, token: Optional[str] = None) -> Optional[WeightsInfoResponse]:
        """
        Get weights info with optional ownership validation.
        
        Args:
            checkpoint_path: The twinkle:// path
            token: If provided, validate ownership
        """
        twinkle_path = cls.parse_twinkle_path(checkpoint_path)
        if not twinkle_path:
            return None
        
        ckpt_info = cls.get(twinkle_path.training_run_id, twinkle_path.checkpoint_id)
        if not ckpt_info:
            return None
        
        # Weight info is stored in the training run info
        run_info = TrainingRunManager._read_info(twinkle_path.training_run_id)
        if not run_info:
            return None
        
        # Validate ownership if token provided
        if token and not validate_ownership(token, run_info.get('model_owner', '')):
            return None
        
        return WeightsInfoResponse(
            training_run_id=run_info.get('training_run_id', ''),
            base_model=run_info.get('base_model', ''),
            model_owner=run_info.get('model_owner', ''),
            is_lora=run_info.get('is_lora', False),
            lora_rank=run_info.get('lora_rank'),
        )
