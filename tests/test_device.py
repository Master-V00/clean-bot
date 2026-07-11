"""Runtime device detection tests.

These tests verify that the unified device detection module:
- always returns sensible defaults (CPU fallback when no GPU)
- never raises during probe
- produces consistent output
"""

import pytest


def test_cpu_fallback_behavior():
    """CPU fallback must work when CUDA/MPS are unavailable.

    This runs in the test environment (which may or may not have GPU).
    The key assertion: device_kind is always a valid DeviceKind, never None.
    """
    from pcleaner.runtime.device import (
        device_kind,
        device_name,
        cuda_available,
        mps_available,
        gpu_available,
        torch_device_str,
        DeviceKind,
    )

    kind = device_kind()
    assert kind in (DeviceKind.CPU, DeviceKind.CUDA, DeviceKind.MPS)
    assert device_name() != ""
    assert isinstance(cuda_available(), bool)
    assert isinstance(mps_available(), bool)
    assert isinstance(gpu_available(), bool)
    assert torch_device_str() in ("cpu", "cuda", "mps")

    # If neither CUDA nor MPS is available, we must be on CPU.
    if not cuda_available() and not mps_available():
        assert kind == DeviceKind.CPU
        assert torch_device_str() == "cpu"
        assert gpu_available() is False


def test_startup_banner_never_empty():
    from pcleaner.runtime.device import startup_banner

    banner = startup_banner()
    assert len(banner) > 10
    assert "Running on:" in banner


def test_hardware_info_always_complete():
    from pcleaner.runtime.device import hardware_info

    info = hardware_info()
    assert "Device:" in info
    assert "Accelerator:" in info
    assert "CUDA available:" in info
    assert "MPS available:" in info
    assert "Platform:" in info


def test_repeated_calls_consistent():
    """Probe is cached; repeated calls must return the same value."""
    from pcleaner.runtime.device import device_kind, device_name, cuda_available

    k1, n1, c1 = device_kind(), device_name(), cuda_available()
    k2, n2, c2 = device_kind(), device_name(), cuda_available()
    assert k1 == k2
    assert n1 == n2
    assert c1 == c2
