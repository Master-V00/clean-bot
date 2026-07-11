"""Smoke tests: verify all key modules import without error."""


def test_pcleaner_import():
    import pcleaner
    assert pcleaner.__version__


def test_device_module():
    from pcleaner.runtime.device import (
        cuda_available,
        mps_available,
        gpu_available,
        device_kind,
        device_name,
        torch_device_str,
        startup_banner,
        hardware_info,
        DeviceKind,
    )
    kind = device_kind()
    assert isinstance(kind, DeviceKind)
    assert isinstance(device_name(), str)
    assert isinstance(gpu_available(), bool)
    assert isinstance(cuda_available(), bool)
    assert isinstance(mps_available(), bool)
    assert torch_device_str() in ("cpu", "cuda", "mps")


def test_helpers_module():
    import pcleaner.helpers as hp

    assert hp.sys_virtual_memory_total() > 0
    assert 0 <= hp.sys_virtual_memory_used_percent() <= 100
    gpu, mps = hp.get_gpu_status()
    assert isinstance(gpu, bool)
    assert isinstance(mps, bool)


def test_cli_utils():
    import pcleaner.cli_utils


def test_config_module():
    import pcleaner.config


def test_model_downloader():
    import pcleaner.model_downloader

    assert pcleaner.model_downloader.INTERNAL_MODELS_DIR.exists()


def test_memory_watcher():
    import pcleaner.memory_watcher


def test_server_config():
    from server.config import settings

    assert settings.API_HOST == "0.0.0.0"
    assert settings.API_PORT == 8000
    assert settings.DEBUG is False


def test_server_models():
    from server.models import ProcessingRequest, ProcessingResponse, HealthResponse, RequestStatus


def test_server_api_import():
    from server.api import router, app
