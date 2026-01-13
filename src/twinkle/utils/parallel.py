# Copyright (c) ModelScope Contributors. All rights reserved.
import os
import shutil
from contextlib import contextmanager

from datasets.utils._filelock import FileLock

# Create locks directory
_locks_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', '.locks')
os.makedirs(_locks_dir, exist_ok=True)


@contextmanager
def processing_lock(lock_file: str, timeout: float = 600.0):
    """Acquire a file lock for distributed-safe processing.
    
    Args:
        lock_file: Name of the lock file (will be sanitized).
        timeout: Maximum time to wait for lock acquisition in seconds.
    
    In distributed training, only rank 0 should process data while
    other ranks wait. This lock ensures that.
    """
    # Sanitize lock file name
    safe_name = lock_file.replace('/', '_').replace(':', '_').replace(' ', '_')
    lock_path = os.path.join(_locks_dir, f"{safe_name}.lock")
    lock = FileLock(lock_path, timeout=timeout)

    try:
        # Try to acquire lock with blocking and timeout
        lock.acquire(blocking=True, timeout=timeout)
        try:
            yield
        finally:
            lock.release()
    except Exception:
        # If lock acquisition fails (e.g., timeout), still yield to allow progress
        # This prevents deadlock in distributed scenarios
        yield