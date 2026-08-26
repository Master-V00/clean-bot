"""
Unified hardware detection module.

Single source of truth for GPU/CUDA/MPS/CPU detection.
All consumers (CLI, GUI, server) must read from this module.
"""

from __future__ import annotations

import logging
import platform
from enum import Enum, auto
from typing import Optional

logger = logging.getLogger(__name__)


class DeviceKind(Enum):
    CPU = auto()
    CUDA = auto()
    MPS = auto()


# ── cached globals (lazy, set once) ──────────────────────────────────────────
_DEVICE_KIND: Optional[DeviceKind] = None
_DEVICE_NAME: Optional[str] = None
_CUDA_AVAILABLE: Optional[bool] = None
_MPS_AVAILABLE: Optional[bool] = None


def _probe_torch() -> None:
    """Lazy-import torch and populate cached flags."""
    global _CUDA_AVAILABLE, _MPS_AVAILABLE, _DEVICE_NAME, _DEVICE_KIND

    try:
        import torch
    except ImportError:
        _CUDA_AVAILABLE = False
        _MPS_AVAILABLE = False
        _DEVICE_NAME = None
        _DEVICE_KIND = DeviceKind.CPU
        return

    # CUDA
    try:
        _CUDA_AVAILABLE = torch.cuda.is_available()
    except Exception:
        _CUDA_AVAILABLE = False

    # MPS (Apple Silicon)
    try:
        _MPS_AVAILABLE = torch.backends.mps.is_available()
    except Exception:
        _MPS_AVAILABLE = False

    # Device name
    if _CUDA_AVAILABLE:
        try:
            _DEVICE_NAME = torch.cuda.get_device_name(0)
        except Exception:
            _DEVICE_NAME = "CUDA device"
        _DEVICE_KIND = DeviceKind.CUDA
    elif _MPS_AVAILABLE:
        _DEVICE_NAME = "Apple MPS"
        _DEVICE_KIND = DeviceKind.MPS
    else:
        _DEVICE_NAME = "CPU"
        _DEVICE_KIND = DeviceKind.CPU


# ── public API ───────────────────────────────────────────────────────────────


def cuda_available() -> bool:
    if _CUDA_AVAILABLE is None:
        _probe_torch()
    return bool(_CUDA_AVAILABLE)


def mps_available() -> bool:
    if _MPS_AVAILABLE is None:
        _probe_torch()
    return bool(_MPS_AVAILABLE)


def device_kind() -> DeviceKind:
    if _DEVICE_KIND is None:
        _probe_torch()
    return _DEVICE_KIND  # type: ignore[return-value]


def device_name() -> str:
    if _DEVICE_NAME is None:
        _probe_torch()
    return _DEVICE_NAME or "CPU"


def torch_device_str() -> str:
    """Return the string to pass to ``torch.device(...)``."""
    kind = device_kind()
    if kind == DeviceKind.CUDA:
        return "cuda"
    elif kind == DeviceKind.MPS:
        return "mps"
    return "cpu"


def gpu_available() -> bool:
    """True if *any* hardware accelerator (CUDA or MPS) is usable."""
    return cuda_available() or mps_available()


# ── startup banner ───────────────────────────────────────────────────────────


def startup_banner() -> str:
    """
    Return a human-readable banner printed once at startup.

    Examples::

        Running on: GPU (CUDA) — NVIDIA GeForce RTX 4090
        Running on: Apple MPS
        Running on: CPU (no GPU accelerator found)
    """
    kind = device_kind()
    name = device_name()

    if kind == DeviceKind.CUDA:
        return f"Running on: GPU (CUDA) — {name}"
    elif kind == DeviceKind.MPS:
        return "Running on: Apple MPS"
    else:
        return "Running on: CPU (no GPU accelerator found)"


def hardware_info() -> str:
    """Return a multi-line summary of all detected hardware."""
    lines = [f"Device:              {device_name()}"]
    lines.append(f"Accelerator:         {device_kind().name}")
    lines.append(f"CUDA available:      {cuda_available()}")
    lines.append(f"MPS available:       {mps_available()}")

    if cuda_available():
        try:
            import torch

            lines.append(f"CUDA version:        {torch.version.cuda or 'unknown'}")
            lines.append(
                f"GPU memory:          {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GiB"
            )
        except Exception:
            pass

    lines.append(f"Platform:            {platform.system()} {platform.release()}")
    return "\n".join(lines)
