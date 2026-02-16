# Copyright (c) ModelScope Contributors. All rights reserved.
import os
import re
import socket
import subprocess
import logging
import torch
from datetime import timedelta
from typing import Optional

# ref: https://www.hiascend.com/document/detail/zh/canncommercial/83RC1/maintenref/envvar/envref_07_0144.html
# HCCL base port anchor. HCCL derives internal listen/connect ports from this base.
_HCCL_IF_BASE_PORT_ENV = 'HCCL_IF_BASE_PORT'
# Host-side socket port pool used by HCCL in multi-process communication.
_HCCL_HOST_SOCKET_PORT_RANGE_ENV = 'HCCL_HOST_SOCKET_PORT_RANGE'
# NPU-side socket port pool used by HCCL for device communication channels.
_HCCL_NPU_SOCKET_PORT_RANGE_ENV = 'HCCL_NPU_SOCKET_PORT_RANGE'
_HCCL_IF_IP_ENV = 'HCCL_IF_IP'
_HCCL_SOCKET_IFNAME_ENV = 'HCCL_SOCKET_IFNAME'
_HCCL_SOCKET_FAMILY_ENV = 'HCCL_SOCKET_FAMILY'

logger = logging.getLogger(__name__)


def _derive_hccl_socket_env_defaults(master_port: int) -> dict:
    """Derive deterministic default HCCL socket env values from master_port."""
    # Keep values stable per job and spread jobs across non-overlapping ranges.
    # A 512-port pool is too tight under multi-process NPU jobs (model + sampler
    # + other communicators) and can lead to partial link-establish timeouts.
    # Use a wider default pool while still keeping deterministic derivation.
    host_offset = master_port % 7000
    range_width = int(os.environ.get('TWINKLE_HCCL_PORT_RANGE_WIDTH', '4096'))
    range_width = max(512, min(range_width, 8192))
    host_start = 38000 + host_offset
    host_end = host_start + range_width - 1

    # Keep the ranges within valid TCP port bounds.
    if host_end > 65000:
        shift = host_end - 65000
        host_start -= shift
        host_end -= shift

    defaults = {
        _HCCL_IF_BASE_PORT_ENV: str(20000 + ((master_port + 997) % 20000)),
        _HCCL_HOST_SOCKET_PORT_RANGE_ENV: f'{host_start}-{host_end}',
    }

    # Only derive HCCL_NPU_SOCKET_PORT_RANGE when explicitly requested.
    # Forcing NPU-side socket ports can cause link-establish timeouts when
    # certain chip-to-chip NPU network paths are unavailable (observed as
    # "wait socket establish timeout" in HCCL plog). Leaving it unset lets
    # HCCL use its own port selection and transport fallback logic.
    if os.environ.get('TWINKLE_HCCL_SET_NPU_PORT_RANGE', '0').lower() in {'1', 'true', 'yes'}:
        npu_start = 50000 + host_offset
        npu_end = npu_start + range_width - 1
        if npu_end > 65000:
            shift = npu_end - 65000
            npu_start -= shift
            npu_end -= shift
        defaults[_HCCL_NPU_SOCKET_PORT_RANGE_ENV] = f'{npu_start}-{npu_end}'

    return defaults


def _ensure_hccl_socket_env(master_port: int, environ: Optional[dict] = None) -> None:
    """Set deterministic HCCL socket env defaults to avoid port collisions.

    In multi-job environments, HCCL's default base port (60000) can collide
    across concurrent jobs and lead to:
    `ra_hdc_socket_listen_start ... ret(-98)`.

    We derive a per-job port layout from `master_port` so all ranks use the
    same values while reducing cross-job conflicts. Explicit user settings are
    preserved and never overwritten.
    """
    env = os.environ if environ is None else environ
    for key, value in _derive_hccl_socket_env_defaults(master_port).items():
        env.setdefault(key, value)

    # Increase HCCL link-establish timeout for large checkpoint groups.
    # Default 120s can be insufficient when many ranks establish links
    # concurrently, leading to partial timeouts and broadcast hangs.
    env.setdefault('HCCL_CONNECT_TIMEOUT', os.environ.get(
        'TWINKLE_HCCL_CONNECT_TIMEOUT', '600'
    ))
    env.setdefault('HCCL_EXEC_TIMEOUT', os.environ.get(
        'TWINKLE_HCCL_EXEC_TIMEOUT', '1800'
    ))


def _get_default_route_ip_and_ifname() -> tuple[Optional[str], Optional[str]]:
    """Get (src_ip, ifname) from the system default IPv4 route."""
    try:
        output = subprocess.check_output(
            ['ip', '-o', '-4', 'route', 'show', 'to', 'default'],
            text=True,
            stderr=subprocess.STDOUT,
            timeout=3,
        )
    except Exception:
        return None, None

    # Typical line:
    # default via 195.27.0.1 dev enp196s0f0 proto static metric 100 src 195.27.1.3
    for line in output.splitlines():
        m_if = re.search(r'\bdev\s+(\S+)', line)
        m_ip = re.search(r'\bsrc\s+((?:\d{1,3}\.){3}\d{1,3})', line)
        if m_if and m_ip:
            return m_ip.group(1), m_if.group(1)
        if m_if:
            # Some systems omit "src" in default route output.
            # Resolve IPv4 from the default route interface.
            ifname = m_if.group(1)
            ip = _get_ipv4_by_ifname(ifname)
            if ip:
                return ip, ifname
    return None, None


def _get_ipv4_by_ifname(ifname: str) -> Optional[str]:
    """Resolve first IPv4 address of a network interface by name."""
    try:
        output = subprocess.check_output(
            ['ip', '-o', '-4', 'addr', 'show', 'dev', ifname],
            text=True,
            stderr=subprocess.STDOUT,
            timeout=3,
        )
    except Exception:
        return None

    for line in output.splitlines():
        m_ip = re.search(r'\binet\s+((?:\d{1,3}\.){3}\d{1,3})/\d+', line)
        if m_ip:
            return m_ip.group(1)
    return None


def _get_ifname_by_ip(ip: str) -> Optional[str]:
    """Resolve interface name by local IPv4 address."""
    try:
        import psutil
    except Exception:
        return None

    for name, addrs in psutil.net_if_addrs().items():
        for addr in addrs:
            if addr.family == socket.AF_INET and addr.address == ip:
                return name
    return None


def _is_local_ipv4(ip: str) -> bool:
    if not ip or is_valid_ipv6_address(ip):
        return False
    return _get_ifname_by_ip(ip) is not None


def _ensure_hccl_iface_env(master_address: str, environ: Optional[dict] = None) -> None:
    """Set deterministic HCCL host networking env for link setup.

    Why this is needed:
    - HCCL chooses host NIC by priority:
      HCCL_IF_IP > HCCL_SOCKET_IFNAME > auto-pick by interface name order.
    - In mixed NIC environments, auto-pick can select virtual links and cause
      `wait socket establish timeout` or `not found specific resource`.
    """
    env = os.environ if environ is None else environ

    # 1) Prefer explicit HCCL_IF_IP (highest priority in HCCL).
    hccl_if_ip = env.get(_HCCL_IF_IP_ENV)
    if not hccl_if_ip:
        if master_address and _is_local_ipv4(master_address):
            hccl_if_ip = master_address
        if not hccl_if_ip:
            default_ip, _ = _get_default_route_ip_and_ifname()
            if default_ip and not default_ip.startswith('127.'):
                hccl_if_ip = default_ip
        if hccl_if_ip:
            env.setdefault(_HCCL_IF_IP_ENV, hccl_if_ip)

    # 2) Set exact-match IFNAME based on selected IP when user did not specify.
    if not env.get(_HCCL_SOCKET_IFNAME_ENV):
        ifname = _get_ifname_by_ip(env.get(_HCCL_IF_IP_ENV, ''))
        if ifname is None and master_address and not is_valid_ipv6_address(master_address):
            ifname = _get_ifname_by_ip(master_address)
        if ifname is None:
            _, ifname = _get_default_route_ip_and_ifname()
        if ifname:
            # Exact-match mode, per CANN docs:
            # HCCL_SOCKET_IFNAME==eth0
            env.setdefault(_HCCL_SOCKET_IFNAME_ENV, f'=={ifname}')

    # 3) Pin IPv4 family for IPv4 host links to avoid address-family mismatch.
    selected_ip = env.get(_HCCL_IF_IP_ENV, '')
    if selected_ip and not is_valid_ipv6_address(selected_ip):
        env.setdefault(_HCCL_SOCKET_FAMILY_ENV, 'AF_INET')


def should_enable_hccl_port_derive(environ: Optional[dict] = None) -> bool:
    """Whether to enable HCCL port derivation for checkpoint process groups.

    Priority:
    1. Explicit env override via ``TWINKLE_ENABLE_HCCL_PORT_DERIVE``.
    2. Auto-enable on NPU runtime by default.
    """
    env = os.environ if environ is None else environ
    raw = env.get('TWINKLE_ENABLE_HCCL_PORT_DERIVE')
    if raw is not None:
        return raw.strip().lower() in {'1', 'true', 'yes', 'on'}
    try:
        return hasattr(torch, 'npu') and torch.npu.is_available()
    except Exception:
        return False


def is_valid_ipv6_address(ip: str) -> bool:
    """Check if the given string is a valid IPv6 address."""
    try:
        socket.inet_pton(socket.AF_INET6, ip)
        return True
    except OSError:
        return False


def find_node_ip() -> Optional[str]:
    # Explicit override for special routing setups.
    if os.environ.get('TWINKLE_NODE_IP'):
        return os.environ['TWINKLE_NODE_IP']

    # Prefer default-route source IP to avoid selecting virtual links
    # (e.g., NodeBabyLink) as the checkpoint master address.
    prefer_default = os.environ.get('TWINKLE_PREFER_DEFAULT_ROUTE_IP', '1').lower() in {
        '1', 'true', 'yes', 'on'
    }
    if prefer_default:
        default_ip, _ = _get_default_route_ip_and_ifname()
        if default_ip and not default_ip.startswith('127.'):
            return default_ip

    import psutil
    main_ip, virtual_ip = None, None
    for name, addrs in sorted(psutil.net_if_addrs().items()):
        for addr in addrs:
            if addr.family.name == 'AF_INET' and not addr.address.startswith('127.'):
                # Heuristic to prefer non-virtual interfaces
                if any(s in name for s in ['lo', 'docker', 'veth', 'vmnet']):
                    if virtual_ip is None:
                        virtual_ip = addr.address
                else:
                    if main_ip is None:
                        main_ip = addr.address
    return main_ip or virtual_ip


def find_free_port(address: str = '', start_port: Optional[int] = None, retry: int = 100) -> int:
    family = socket.AF_INET
    if address and is_valid_ipv6_address(address):
        family = socket.AF_INET6
    if start_port is None:
        start_port = 0
    for port in range(start_port, start_port + retry):
        with socket.socket(family, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
            try:
                sock.bind(('', port))
                port = sock.getsockname()[1]
                break
            except OSError:
                pass
    return port


def stateless_init_process_group(
    master_address: str,
    master_port: int,
    rank: int,
    world_size: int,
    device: int | torch.device = None,
    backend: str = 'nccl',
    listen_socket: socket.socket = None,
    listen_fd: int = None,
):
    """Create a stateless process group using vLLM's StatelessProcessGroup.

    vLLM provides `StatelessProcessGroup` to create a process group
    without considering the global process group in torch.distributed.
    It is recommended to create `StatelessProcessGroup`, and then initialize
    the data-plane communication (NCCL/HCCL) between external (train processes)
    and vLLM workers.

    Args:
        master_address: The IP address of the master (rank 0).
        master_port: The port of the master.
        rank: The rank of this process.
        world_size: Total number of processes.
        device: The CUDA device to use. If None, uses current device.
        backend: The communication backend ("nccl" or "hccl").
        listen_socket: Optional pre-created listening socket for master (rank 0).
            If provided, this socket will be reused instead of creating a new one.
        listen_fd: Optional file descriptor of the listening socket.

    Returns:
        PyNcclCommunicator or PyHcclCommunicator instance.
    """
    from torch.distributed import TCPStore
    from vllm.distributed.utils import StatelessProcessGroup
    if backend == 'hccl':
        # Default: enable on NPU to avoid HCCL port collisions for checkpoint PG.
        # Users can still override with TWINKLE_ENABLE_HCCL_PORT_DERIVE=0/1.
        if should_enable_hccl_port_derive():
            _ensure_hccl_socket_env(master_port)
        _ensure_hccl_iface_env(master_address)
        from vllm_ascend.distributed.device_communicators.pyhccl import PyHcclCommunicator as Communicator
    else:
        from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator as Communicator

    if device is None:
        device = torch.cuda.current_device() if backend == 'nccl' else torch.npu.current_device()

    # Create the stateless process group
    launch_server = rank == 0

    if launch_server and listen_socket is None:
        # For master, create a listening socket if not provided
        if is_valid_ipv6_address(master_address):
            listen_socket = socket.socket(socket.AF_INET6, socket.SOCK_STREAM)
        else:
            listen_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listen_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listen_socket.bind((master_address, master_port))
        listen_socket.listen()
        listen_fd = listen_socket.fileno()
    elif launch_server and listen_fd is None:
        listen_fd = listen_socket.fileno()

    store = TCPStore(
        host_name=master_address,
        port=master_port,
        world_size=world_size,
        is_master=launch_server,
        timeout=timedelta(seconds=300),
        use_libuv=False,  # for compatibility
        master_listen_fd=listen_fd,
    )

    pg = StatelessProcessGroup(
        rank=rank,
        world_size=world_size,
        store=store,
        socket=listen_socket,
        data_expiration_seconds=3600,
    )

    if backend == 'hccl':
        logger.info(
            f'HCCL checkpoint PG env (rank {rank}): '
            f"{_HCCL_IF_BASE_PORT_ENV}={os.environ.get(_HCCL_IF_BASE_PORT_ENV)}, "
            f"{_HCCL_HOST_SOCKET_PORT_RANGE_ENV}={os.environ.get(_HCCL_HOST_SOCKET_PORT_RANGE_ENV)}, "
            f"{_HCCL_NPU_SOCKET_PORT_RANGE_ENV}={os.environ.get(_HCCL_NPU_SOCKET_PORT_RANGE_ENV, 'not set')}, "
            f"{_HCCL_IF_IP_ENV}={os.environ.get(_HCCL_IF_IP_ENV)}, "
            f"{_HCCL_SOCKET_FAMILY_ENV}={os.environ.get(_HCCL_SOCKET_FAMILY_ENV)}, "
            f"{_HCCL_SOCKET_IFNAME_ENV}={os.environ.get(_HCCL_SOCKET_IFNAME_ENV)}, "
            f"HCCL_CONNECT_TIMEOUT={os.environ.get('HCCL_CONNECT_TIMEOUT', 'not set')}"
        )

    communicator = Communicator(pg, device=device)
    return communicator
