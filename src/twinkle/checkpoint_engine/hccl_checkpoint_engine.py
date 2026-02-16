# Copyright (c) ModelScope Contributors. All rights reserved.
# Adapted from https://github.com/volcengine/verl/blob/main/verl/checkpoint_engine/hccl_checkpoint_engine.py
"""HCCL-based checkpoint engine for Ascend NPU.

This engine uses HCCL broadcast for efficient NPU-to-NPU weight transfer
across different processes/nodes. It supports:
- Double buffering for pipelined transfer
- ZMQ for metadata, HCCL for weight data
- Streaming weight transfer to avoid OOM
"""

import asyncio
import os
import time
import torch
import zmq
from dataclasses import dataclass
from typing import Any, AsyncGenerator, Generator

from twinkle import get_logger
from twinkle.utils.network import find_free_port, find_node_ip, is_valid_ipv6_address, stateless_init_process_group
from .base import CheckpointEngine, TensorMeta

logger = get_logger()


@dataclass
class MasterMetadata:
    """Metadata from the master for process group initialization."""
    zmq_ip: str
    zmq_port: int
    zmq_sync_port: int
    dist_ip: str
    dist_port: int


class BroadcastOperation:
    """Async broadcast operation with HCCL or ZMQ fallback in separate thread.

    Args:
        rank: The rank of the current process.
        process_group: The HCCL process group.
        bucket: The tensor buffer to broadcast.
        metadata: The metadata of tensors in the bucket.
        socket: The ZMQ socket for metadata communication.
        topic: The ZMQ topic for pub/sub.
        use_zmq_data: If True, use ZMQ for tensor data transfer instead of HCCL.
    """

    def __init__(
        self,
        rank: int,
        process_group,
        bucket: torch.Tensor,
        metadata: dict[str, TensorMeta],
        socket: zmq.Socket,
        topic: str,
        use_zmq_data: bool = False,
    ) -> None:
        self.rank = rank
        self.pyhccl = process_group
        self.bucket = bucket
        self.metadata = metadata
        self.socket = socket
        self.topic = topic
        self.use_zmq_data = use_zmq_data

        loop = asyncio.get_running_loop()
        self._task = loop.run_in_executor(None, self._run)

    def _compute_data_end(self, metadata: dict) -> int:
        """Compute the end offset of used data in the bucket from metadata."""
        data_end = 0
        for meta in metadata.get('bucket_meta', {}).values():
            end = meta['offset'] + meta['dtype'].itemsize * meta['shape'].numel()
            data_end = max(data_end, end)
        return data_end

    def _run(self):
        """Execute the broadcast operation in a thread."""
        # Broadcast tensor metadata via ZMQ PUB/SUB
        if self.rank == 0:
            self.socket.send_string(self.topic, flags=zmq.SNDMORE)
            self.socket.send_pyobj(self.metadata)
        else:
            try:
                self.socket.recv_string()
                self.metadata = self.socket.recv_pyobj()
            except zmq.error.Again as e:
                raise RuntimeError(
                    'Timed out waiting for checkpoint metadata over ZMQ; '
                    'possible rank desync or subscriber slow-join.'
                ) from e

        if self.use_zmq_data:
            # ZMQ-based data transfer: avoids HCCL device-plane sockets that
            # can fail on certain NPU chip topologies.
            data_end = self._compute_data_end(self.metadata)
            if data_end == 0:
                return
            if self.rank == 0:
                cpu_data = self.bucket[:data_end].cpu().numpy().tobytes()
                self.socket.send_string(self.topic, flags=zmq.SNDMORE)
                self.socket.send(cpu_data, copy=False)
            else:
                self.socket.recv_string()
                data = self.socket.recv()
                cpu_tensor = torch.frombuffer(bytearray(data), dtype=torch.uint8)
                self.bucket[:data_end].copy_(cpu_tensor.to(self.bucket.device))
        else:
            # Broadcast tensor data via HCCL
            self.pyhccl.broadcast(self.bucket, src=0)

    async def wait_for_complete(self) -> dict[str, TensorMeta]:
        """Wait for the broadcast operation to complete.

        Returns:
            The bucket metadata after broadcast.
        """
        await self._task
        return self.metadata


class HCCLCheckpointEngine(CheckpointEngine):
    """HCCL checkpoint engine for Ascend NPU.

    Same lifecycle and semantics as NCCLCheckpointEngine but uses HCCL
    instead of NCCL and stateless_init_process_group instead of
    ray.util.collective.

    Args:
        bucket_size: Bucket size in bytes for weight transfer.
        group_name: Name of the process group.
        rebuild_group: Whether to rebuild the group each sync.
        rollout_dtype: Target dtype for weights.
    """

    def __init__(
        self,
        bucket_size: int = 2048 << 20,
        group_name: str = 'twinkle_ckpt',
        rebuild_group: bool = True,
        rollout_dtype: torch.dtype = torch.bfloat16,
        **kwargs,
    ) -> None:
        self.bucket_size = bucket_size
        self.group_name = group_name
        self.rebuild_group = rebuild_group
        self.rollout_dtype = rollout_dtype
        self.pyhccl = None

        # Get current NPU device
        try:
            self.device = torch.npu.current_device()
        except Exception:
            self.device = 0

        # Set by Manager before prepare() via attribute assignment
        self.is_master = False
        self.topic = 'bucket_metadata'

        # Will be set during prepare / init_process_group
        self.rank = None
        self.world_size = None
        self.send_buf = None
        self.recv_buf = None
        self.socket = None
        self.sync_socket = None
        self._zmq_context = None

        # Track whether resources are ready for reuse
        self._prepared = False
        self._group_initialized = False
        self.slow_joiner_delay_s = float(
            os.environ.get("TWINKLE_HCCL_ZMQ_SLOW_JOINER_DELAY_S", "0")
        )
        self.zmq_recv_timeout_ms = int(
            float(os.environ.get("TWINKLE_HCCL_ZMQ_RECV_TIMEOUT_S", "180")) * 1000
        )
        self.zmq_handshake_timeout_ms = int(
            float(os.environ.get("TWINKLE_HCCL_ZMQ_HANDSHAKE_TIMEOUT_S", "60")) * 1000
        )
        # When HCCL device-plane sockets fail (e.g. cross-module NPU links),
        # fall back to ZMQ host-side TCP for tensor data transfer.
        #
        # Default policy:
        # - If TWINKLE_HCCL_USE_ZMQ_DATA is explicitly set, honor it.
        # - Otherwise, auto-enable on NPU to avoid known HCCL Broadcast hangs
        #   ("wait socket establish timeout" / "Alloc transports failed").
        raw_use_zmq = os.environ.get("TWINKLE_HCCL_USE_ZMQ_DATA")
        if raw_use_zmq is None:
            try:
                self.use_zmq_data = hasattr(torch, 'npu') and torch.npu.is_available()
            except Exception:
                self.use_zmq_data = False
        else:
            self.use_zmq_data = raw_use_zmq.lower() in {'1', 'true', 'yes'}

    # ── ZMQ helpers ──────────────────────────────────────────────────────

    def _start_zmq_server(self):
        """Start ZMQ PUB server for metadata broadcast (master only)."""
        self.ip = find_node_ip()
        self.zmq_port = find_free_port()
        self.zmq_sync_port = find_free_port()
        self.dist_port = find_free_port()

        self._zmq_context = zmq.Context()
        self.socket = self._zmq_context.socket(zmq.PUB)
        self.socket.setsockopt(zmq.LINGER, 0)
        self.sync_socket = self._zmq_context.socket(zmq.REP)
        self.sync_socket.setsockopt(zmq.LINGER, 0)
        self.sync_socket.setsockopt(zmq.RCVTIMEO, self.zmq_handshake_timeout_ms)
        if is_valid_ipv6_address(self.ip):
            address = f'tcp://[{self.ip}]:{self.zmq_port}'
            sync_address = f'tcp://[{self.ip}]:{self.zmq_sync_port}'
            self.socket.setsockopt(zmq.IPV6, 1)
            self.sync_socket.setsockopt(zmq.IPV6, 1)
        else:
            address = f'tcp://{self.ip}:{self.zmq_port}'
            sync_address = f'tcp://{self.ip}:{self.zmq_sync_port}'

        self.socket.bind(address)
        self.sync_socket.bind(sync_address)
        logger.debug(f'ZMQ PUB server started at {address}, sync REP at {sync_address}')

    def _connect_zmq_client(self, metadata: MasterMetadata):
        """Connect to the ZMQ PUB server as a subscriber (receiver only)."""
        self._zmq_context = zmq.Context()
        self.socket = self._zmq_context.socket(zmq.SUB)
        self.socket.setsockopt(zmq.LINGER, 0)
        self.socket.setsockopt(zmq.RCVTIMEO, self.zmq_recv_timeout_ms)
        self.sync_socket = self._zmq_context.socket(zmq.REQ)
        self.sync_socket.setsockopt(zmq.LINGER, 0)
        self.sync_socket.setsockopt(zmq.SNDTIMEO, self.zmq_handshake_timeout_ms)
        self.sync_socket.setsockopt(zmq.RCVTIMEO, self.zmq_handshake_timeout_ms)
        if is_valid_ipv6_address(metadata.zmq_ip):
            address = f'tcp://[{metadata.zmq_ip}]:{metadata.zmq_port}'
            sync_address = f'tcp://[{metadata.zmq_ip}]:{metadata.zmq_sync_port}'
            self.socket.setsockopt(zmq.IPV6, 1)
            self.sync_socket.setsockopt(zmq.IPV6, 1)
        else:
            address = f'tcp://{metadata.zmq_ip}:{metadata.zmq_port}'
            sync_address = f'tcp://{metadata.zmq_ip}:{metadata.zmq_sync_port}'

        self.socket.connect(address)
        self.socket.setsockopt_string(zmq.SUBSCRIBE, self.topic)
        self.sync_socket.connect(sync_address)
        logger.debug(f'ZMQ SUB client connected to {address}, sync REQ to {sync_address}')

    def _sync_subscribers(self):
        """Synchronize rank-0 publisher and subscriber readiness.

        ZMQ PUB/SUB can drop the first message when subscribers are still
        connecting (slow-joiner). We use a REQ/REP handshake to ensure all
        subscribers are ready before rank-0 sends the first metadata frame.
        """
        if self.rank is None or self.rank < 0 or self.world_size is None or self.world_size <= 1:
            return

        if self.rank == 0:
            expected = self.world_size - 1
            ready_ranks: list[int] = []
            for _ in range(expected):
                try:
                    msg = self.sync_socket.recv_pyobj()
                except zmq.error.Again as e:
                    raise RuntimeError(
                        f'ZMQ subscriber handshake timeout on rank-0: '
                        f'received {len(ready_ranks)}/{expected} ready messages.'
                    ) from e
                ready_rank = int(msg.get('rank', -1))
                ready_ranks.append(ready_rank)
                self.sync_socket.send_string('ok')
            logger.debug(f'ZMQ subscriber handshake complete, ready_ranks={sorted(ready_ranks)}')
        else:
            try:
                self.sync_socket.send_pyobj({'rank': self.rank})
                self.sync_socket.recv_string()
            except zmq.error.Again as e:
                raise RuntimeError(
                    f'ZMQ subscriber handshake timeout on rank {self.rank}.'
                ) from e

    # ── Core lifecycle ───────────────────────────────────────────────────

    def prepare(self) -> MasterMetadata | None:
        """Allocate double buffers and start ZMQ server (master only).

        Idempotent: skips if already prepared.

        Returns:
            MasterMetadata with ZMQ/dist IP/port if master, else None.
        """
        if self._prepared:
            if self.is_master:
                return MasterMetadata(
                    zmq_ip=self.ip,
                    zmq_port=self.zmq_port,
                    zmq_sync_port=self.zmq_sync_port,
                    dist_ip=self.ip,
                    dist_port=self.dist_port,
                )
            return None

        self.send_buf = torch.zeros(self.bucket_size, dtype=torch.uint8, device='npu')
        self.recv_buf = torch.zeros(self.bucket_size, dtype=torch.uint8, device='npu')

        if self.is_master:
            self._start_zmq_server()
            self._prepared = True
            return MasterMetadata(
                zmq_ip=self.ip,
                zmq_port=self.zmq_port,
                zmq_sync_port=self.zmq_sync_port,
                dist_ip=self.ip,
                dist_port=self.dist_port,
            )
        self._prepared = True
        return None

    def finalize(self):
        """Clean up resources after a sync.

        When ``rebuild_group=False``: keeps everything alive for reuse.
        When ``rebuild_group=True``: full teardown.
        """
        if self.rebuild_group:
            if self.socket is not None:
                try:
                    self.socket.close()
                except Exception as e:
                    logger.warning(f'Error closing ZMQ socket: {e}')
                self.socket = None
            if self.sync_socket is not None:
                try:
                    self.sync_socket.close()
                except Exception as e:
                    logger.warning(f'Error closing ZMQ sync socket: {e}')
                self.sync_socket = None
            if self._zmq_context is not None:
                try:
                    self._zmq_context.term()
                except Exception as e:
                    logger.warning(f'Error terminating ZMQ context: {e}')
                self._zmq_context = None

            if self.rank is not None and self.rank >= 0 and self.pyhccl is not None:
                try:
                    self.pyhccl.destroyComm(self.pyhccl.comm)
                except Exception:
                    pass
                self.pyhccl = None

            self.rank = None
            self.world_size = None
            self.send_buf = None
            self.recv_buf = None
            self._prepared = False
            self._group_initialized = False

    @classmethod
    def build_topology(
        cls,
        trainer_world_size: int,
        rollout_world_size: int,
        metadata: list[dict],
    ) -> tuple[dict[str, list[Any]], dict[str, list[Any]]]:
        """Build communication topology for HCCL broadcast.

        Same topology as NCCLCheckpointEngine.
        """
        master_metadata = None
        for m in metadata:
            if m is not None:
                master_metadata = m
                break

        trainer_kwargs = {
            'rank': [0] + [-1] * (trainer_world_size - 1),
            'world_size': [rollout_world_size + 1] * trainer_world_size,
            'master_metadata': [master_metadata] * trainer_world_size,
        }
        rollout_kwargs = {
            'rank': list(range(1, rollout_world_size + 1)),
            'world_size': [rollout_world_size + 1] * rollout_world_size,
            'master_metadata': [master_metadata] * rollout_world_size,
        }
        return trainer_kwargs, rollout_kwargs

    def init_process_group(self, rank: int, world_size: int, master_metadata: MasterMetadata):
        """Initialize the HCCL process group.

        Idempotent: if already initialized and ``rebuild_group`` is False,
        this is a fast no-op.

        Args:
            rank: The rank of this worker (-1 for non-participating trainers).
            world_size: Total number of workers in the sync group.
            master_metadata: Metadata from the master.
        """
        # Non-participating trainer ranks
        if rank < 0:
            self.rank = rank
            self.world_size = world_size
            self._group_initialized = True
            return

        # Fast path: already initialized
        if self._group_initialized and not self.rebuild_group:
            return

        if self.rebuild_group or self.pyhccl is None:
            if self.use_zmq_data:
                # Skip HCCL communicator creation: hcclCommInitRank can hang
                # on NPU topologies with broken device-plane sockets.
                self.pyhccl = None
            else:
                self.pyhccl = stateless_init_process_group(
                    master_address=master_metadata.dist_ip,
                    master_port=master_metadata.dist_port,
                    rank=rank,
                    world_size=world_size,
                    device=self.device,
                    backend='hccl',
                )
            self.rank = rank
            self.world_size = world_size
        else:
            assert self.rank == rank
            assert self.world_size == world_size

        # Receivers connect to master's ZMQ PUB server
        if self.rank > 0 and self.socket is None:
            self._connect_zmq_client(master_metadata)
        # Ensure all subscribers are connected before first PUB metadata send.
        self._sync_subscribers()

        # Barrier using all_reduce.
        # Some Ascend runtimes can hang in init-time all_reduce even after
        # communicator setup/heartbeat succeeded. Allow disabling this barrier
        # for troubleshooting, default is disabled.
        if os.environ.get("TWINKLE_HCCL_INIT_BARRIER", "0") == "1" and self.pyhccl is not None:
            signal = torch.tensor([1], dtype=torch.int8, device=torch.npu.current_device())
            self.pyhccl.all_reduce(signal)

        self._group_initialized = True
        logger.info(
            f'init_process_group: rank={self.rank}, world_size={self.world_size}'
            f'{", zmq_data=True" if self.use_zmq_data else ""}'
        )

    # ── Send / Receive ───────────────────────────────────────────────────

    @torch.no_grad()
    async def send_weights(self, weights: Generator[tuple[str, torch.Tensor], None, None]):
        """Send model weights via HCCL broadcast."""
        assert self.rank is not None and self.rank <= 0

        if self.rank < 0:
            for name, weight in weights:
                pass
            return

        send_buf, recv_buf = self.send_buf, self.recv_buf
        broadcast_op = None
        first_publish = True

        start_time = time.time()
        bucket_meta: dict[str, TensorMeta] = {}
        offset = 0

        for name, weight in weights:
            if offset + weight.nbytes > self.bucket_size:
                torch.npu.synchronize()

                if broadcast_op is not None:
                    await broadcast_op.wait_for_complete()

                if first_publish and self.slow_joiner_delay_s > 0:
                    await asyncio.sleep(self.slow_joiner_delay_s)
                    first_publish = False
                broadcast_op = BroadcastOperation(
                    rank=self.rank,
                    process_group=self.pyhccl,
                    bucket=send_buf,
                    metadata={
                        'bucket_meta': bucket_meta,
                        'is_last': False
                    },
                    socket=self.socket,
                    topic=self.topic,
                    use_zmq_data=self.use_zmq_data,
                )

                send_buf, recv_buf = recv_buf, send_buf
                bucket_meta = {}
                offset = 0

            assert offset + weight.nbytes <= self.bucket_size

            bucket_meta[name] = {
                'name': name,
                'shape': weight.shape,
                'dtype': weight.dtype,
                'offset': offset,
            }
            send_buf[offset:offset + weight.nbytes] = weight.view(-1).view(torch.uint8)
            offset += weight.nbytes

        torch.npu.synchronize()
        if broadcast_op is not None:
            await broadcast_op.wait_for_complete()

        # Optional extra delay for debugging unstable runtimes.
        if first_publish and self.slow_joiner_delay_s > 0:
            await asyncio.sleep(self.slow_joiner_delay_s)
            first_publish = False

        broadcast_op = BroadcastOperation(
            rank=self.rank,
            process_group=self.pyhccl,
            bucket=send_buf,
            metadata={
                'bucket_meta': bucket_meta,
                'is_last': True
            },
            socket=self.socket,
            topic=self.topic,
            use_zmq_data=self.use_zmq_data,
        )
        await broadcast_op.wait_for_complete()

        elapsed = time.time() - start_time
        logger.info(f'send_weights done: rank={self.rank}, time={elapsed:.2f}s')

    @torch.no_grad()
    async def receive_weights(self) -> AsyncGenerator[tuple[str, torch.Tensor], None]:
        """Receive model weights via HCCL broadcast."""
        assert self.rank is not None and self.rank > 0

        send_buf, recv_buf = self.send_buf, self.recv_buf
        total_bytes, total_params = 0, 0

        start_time = time.time()
        broadcast_op = BroadcastOperation(
            rank=self.rank,
            process_group=self.pyhccl,
            bucket=recv_buf,
            metadata=None,
            socket=self.socket,
            topic=self.topic,
            use_zmq_data=self.use_zmq_data,
        )
        metadata = await broadcast_op.wait_for_complete()
        total_bytes += self.bucket_size
        total_params += len(metadata['bucket_meta'])

        send_buf, recv_buf = recv_buf, send_buf

        while not metadata['is_last']:
            broadcast_op = BroadcastOperation(
                rank=self.rank,
                process_group=self.pyhccl,
                bucket=recv_buf,
                metadata=None,
                socket=self.socket,
                topic=self.topic,
                use_zmq_data=self.use_zmq_data,
            )

            for name, meta in metadata['bucket_meta'].items():
                dtype, shape = meta['dtype'], meta['shape']
                size = dtype.itemsize * shape.numel()
                tensor = send_buf[meta['offset']:meta['offset'] + size].view(dtype=dtype).view(shape)
                yield name, tensor

            metadata = await broadcast_op.wait_for_complete()
            total_bytes += self.bucket_size
            total_params += len(metadata['bucket_meta'])

            # `wait_for_complete()` already waits for the broadcast thread.
            # A full device synchronize here can hang on some Ascend stacks.
            if os.environ.get("TWINKLE_HCCL_RECEIVE_SYNC", "0") == "1":
                torch.npu.synchronize()
            send_buf, recv_buf = recv_buf, send_buf

        for name, meta in metadata['bucket_meta'].items():
            dtype, shape = meta['dtype'], meta['shape']
            size = dtype.itemsize * shape.numel()
            tensor = send_buf[meta['offset']:meta['offset'] + size].view(dtype=dtype).view(shape)
            yield name, tensor

        elapsed = time.time() - start_time
        bandwidth = total_bytes / elapsed / (1024 * 1024 * 1024)
        logger.info(f'receive_weights done: rank={self.rank}, params={total_params}, '
                    f'time={elapsed:.2f}s, bandwidth={bandwidth:.2f} GB/s')
