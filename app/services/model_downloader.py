import os
import shutil
from hashlib import sha256
from pathlib import Path

import requests
import tqdm
from loguru import logger


# ========== INTERNAL SANDBOXING: ALL MODELS ISOLATED IN app/models/ (project permanent storage, NOT cache) ==========
# __file__ is app/services/model_downloader.py -> parent.parent is app/ -> app/models is the canonical permanent location
# This matches main.py TORCH_HOME and start.sh TORCH_HOME (both point to app/models)
INTERNAL_MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
INTERNAL_INPAINTING_DIR = INTERNAL_MODELS_DIR / "inpainting"
INTERNAL_TEXT_DETECTOR_DIR = INTERNAL_MODELS_DIR / "text_detector"
INTERNAL_CTD_DIR = INTERNAL_TEXT_DETECTOR_DIR / "ctd"
INTERNAL_YOLO_DIR = INTERNAL_TEXT_DETECTOR_DIR / "yolo"
HI_SAM_SRC_DIR = Path(__file__).resolve().parent / "hi_sam_src"
HI_SAM_PRETRAINED_DIR = HI_SAM_SRC_DIR / "pretrained_checkpoint"

# ========== ComicTextDetector (CTD) — zyddnys/manga-image-translator ==========
CTD_MODEL_URL = "https://github.com/zyddnys/manga-image-translator/releases/download/beta-0.3/comictextdetector.pt"
CTD_ONNX_URL = "https://github.com/zyddnys/manga-image-translator/releases/download/beta-0.3/comictextdetector.pt.onnx"

# ========== YOLO11s lordtrilink v0 — HuggingFace ==========
YOLO11S_URL = "https://huggingface.co/lordtrilink/manga-text-detector-v0/resolve/main/best.pt"
YOLO11S_ONNX_URL = "https://huggingface.co/lordtrilink/manga-text-detector-v0/resolve/main/best.onnx"


def _ensure_model_dirs():
    """Create internal model directories on first use. Also migrate legacy app/services/models → app/models."""
    # Migrate legacy location (app/services/models) to canonical app/models to avoid re-download
    try:
        legacy_base = Path(__file__).resolve().parent / "models"
        if legacy_base.is_dir() and legacy_base != INTERNAL_MODELS_DIR:
            for sub in ["inpainting", "text_detector"]:
                legacy_sub = legacy_base / sub
                new_sub = INTERNAL_MODELS_DIR / sub
                if legacy_sub.is_dir() and not new_sub.is_dir():
                    # Move entire directory
                    new_sub.parent.mkdir(parents=True, exist_ok=True)
                    try:
                        shutil.move(str(legacy_sub), str(new_sub))
                        logger.info(f"Migrated legacy models {legacy_sub} → {new_sub}")
                    except Exception:
                        pass
                elif legacy_sub.is_dir() and new_sub.is_dir():
                    # Move individual files missing in new
                    for f in legacy_sub.iterdir():
                        dest = new_sub / f.name
                        if not dest.exists():
                            try:
                                shutil.move(str(f), str(dest))
                            except Exception:
                                try:
                                    shutil.copy2(f, dest)
                                except Exception:
                                    pass
            # Also migrate hub cache if present in old TORCH_HOME
            legacy_hub = legacy_base / "hub"
            new_hub = INTERNAL_MODELS_DIR / "hub"
            if legacy_hub.is_dir() and not new_hub.is_dir():
                try:
                    shutil.move(str(legacy_hub), str(new_hub))
                except Exception:
                    pass
    except Exception:
        pass
    INTERNAL_INPAINTING_DIR.mkdir(parents=True, exist_ok=True)
    INTERNAL_TEXT_DETECTOR_DIR.mkdir(parents=True, exist_ok=True)
    INTERNAL_CTD_DIR.mkdir(parents=True, exist_ok=True)
    INTERNAL_YOLO_DIR.mkdir(parents=True, exist_ok=True)
    HI_SAM_PRETRAINED_DIR.mkdir(parents=True, exist_ok=True)


# ========== Text Detector (Hi-SAM) ==========
_SAM_CHECKPOINT_MAP = {
    "vit_b": "sam_vit_b_01ec64.pth",
    "vit_l": "sam_vit_l_0b3195.pth",
    "vit_h": "sam_vit_h_4b8939.pth",
}
_SAM_CHECKPOINT_URLS = {
    "vit_b": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
    "vit_l": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
    "vit_h": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
}

# Hi-SAM specific checkpoints (OneDrive links from official repo)
_HISAM_CHECKPOINTS = {
    "vit_b_maskdecoder.pth": "https://1drv.ms/u/s!AimBgYV7JjTlgcth1ceH68P-vOF87g?e=lK2bIL",
    "vit_l_maskdecoder.pth": "https://1drv.ms/u/s!AimBgYV7JjTlgctjx03utTjx31EexA?e=HG7zZD",
    "vit_h_maskdecoder.pth": "https://1drv.ms/u/s!AimBgYV7JjTlgctig8BXzlCQaPm1ng?e=6XOCid",
    "word_detection_totaltext.pth": "https://1drv.ms/u/s!AimBgYV7JjTlgco6PgIiYeItOjffnA?e=qb6G0s",
    "line_detection_ctw1500.pth": "https://1drv.ms/u/s!AimBgYV7JjTlgco5llba2msYi3eWXg?e=zKLX4n",
}

HISAM_DEFAULT_MODEL_TYPE = "vit_h"
HISAM_DEFAULT_DETECTION_CKPT = "word_detection_totaltext.pth"


def _download_onedrive(share_url: str, dest_path: Path) -> bool:
    """
    Download a file from a OneDrive sharing link using the Badger token flow.
    Based on: https://gist.github.com/NTFSvolume/e0395fe6eeaa9b47a5c8874ec8133987
    """
    try:
        from urllib.parse import urlparse, parse_qs

        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:135.0) Gecko/20100101 Firefox/135.0"
        }

        # Step 1: Follow the share link redirect to get the real URL with params
        resp = requests.get(share_url, headers=headers, allow_redirects=True, timeout=30)
        resp.raise_for_status()
        parsed = urlparse(resp.url)
        params = parse_qs(parsed.query)

        resid = params.get("resid", [None])[0] or params.get("id", [None])[0]
        cid = params.get("cid", [None])[0]
        authkey = params.get("authkey", [None])[0]
        redeem = params.get("redeem", [None])[0]

        if not resid:
            logger.error(f"Could not parse OneDrive share URL: no resid found in {resp.url}")
            return False

        logger.info(f"Downloading from OneDrive: {dest_path.name}...")

        # Step 2: Get a Badger token (needed for redeem flow)
        session = requests.Session()
        session.headers.update(headers)

        if redeem:
            badger_resp = session.post(
                "https://api-badgerp.svc.ms/v1.0/token",
                headers={"AppId": "1141147648"},
                json={"appId": "5cbed6ac-a083-4e14-b191-b4ba07653de2"},
                timeout=30,
            )
            badger_resp.raise_for_status()
            badger_token = badger_resp.json()["token"]
            session.headers.update({
                "Authorization": f"Badger {badger_token}",
                "Prefer": "autoredeem",
            })

        # Step 3: Get file metadata to find download URL
        if redeem:
            # Personal OneDrive share with redeem token
            api_url = f"https://my.microsoftpersonalcontent.com/_api/v2.0/shares/u!{redeem}/driveItem"
        elif cid:
            # Drive-based API
            api_url = f"https://api.onedrive.com/v1.0/drives/{cid}/items/{resid}"
            if authkey:
                api_url += f"?authkey={authkey}"
        else:
            api_url = f"https://api.onedrive.com/v1.0/shares/u!{resid}/driveItem"

        meta_resp = session.get(api_url, timeout=30)
        meta_resp.raise_for_status()
        meta = meta_resp.json()

        download_url = meta.get("@content.downloadUrl") or meta.get("content.downloadUrl")
        if not download_url:
            logger.error("No download URL in OneDrive metadata response")
            return False

        # Step 4: Download the file
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        dl_resp = session.get(download_url, stream=True, timeout=600)
        dl_resp.raise_for_status()

        with open(dest_path, "wb") as f:
            for chunk in dl_resp.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

        if dest_path.is_file() and dest_path.stat().st_size > 1000:
            logger.info(f"Downloaded: {dest_path.name} ({dest_path.stat().st_size / 1024 / 1024:.1f} MB)")
            return True
        else:
            dest_path.unlink(missing_ok=True)
            logger.error(f"Downloaded file too small or empty: {dest_path.name}")
            return False

    except Exception as e:
        dest_path.unlink(missing_ok=True)
        logger.error(f"OneDrive download failed for {dest_path.name}: {e}")
        return False


def ensure_hi_sam_available(model_type: str = "vit_h") -> Path:
    """Download all Hi-SAM checkpoints needed. Returns text detector dir."""
    _ensure_model_dirs()

    # 1. SAM backbone (from Facebook, direct URL)
    ckpt_name = _SAM_CHECKPOINT_MAP.get(model_type, _SAM_CHECKPOINT_MAP["vit_b"])
    sam_path = INTERNAL_TEXT_DETECTOR_DIR / ckpt_name
    if not sam_path.is_file():
        url = _SAM_CHECKPOINT_URLS.get(model_type, _SAM_CHECKPOINT_URLS["vit_b"])
        logger.info(f"Downloading SAM backbone: {ckpt_name} ...")
        result = download_file(url, INTERNAL_TEXT_DETECTOR_DIR)
        if result is None or not result.is_file():
            raise RuntimeError(f"Failed to download SAM checkpoint from {url}")

    # 2. Mask decoder init weights (for H-Decoder) — store in text_detector dir
    maskdecoder_name = f"{model_type}_maskdecoder.pth"
    maskdecoder_path = INTERNAL_TEXT_DETECTOR_DIR / maskdecoder_name
    if not maskdecoder_path.is_file() and maskdecoder_name in _HISAM_CHECKPOINTS:
        _download_onedrive(_HISAM_CHECKPOINTS[maskdecoder_name], maskdecoder_path)

    # 3. Text detection model
    detection_path = INTERNAL_TEXT_DETECTOR_DIR / HISAM_DEFAULT_DETECTION_CKPT
    if not detection_path.is_file() and HISAM_DEFAULT_DETECTION_CKPT in _HISAM_CHECKPOINTS:
        _download_onedrive(_HISAM_CHECKPOINTS[HISAM_DEFAULT_DETECTION_CKPT], detection_path)

    logger.info("All Hi-SAM checkpoints ready")
    return INTERNAL_TEXT_DETECTOR_DIR


def get_text_detector_model_dir() -> Path:
    """Return the internal directory where text detector checkpoints are stored."""
    _ensure_model_dirs()
    return INTERNAL_TEXT_DETECTOR_DIR


def get_ctd_model_path() -> Path:
    """Return default CTD model path (prefers .pt, falls back to .onnx)."""
    _ensure_model_dirs()
    pt = INTERNAL_CTD_DIR / "comictextdetector.pt"
    onnx = INTERNAL_CTD_DIR / "comictextdetector.pt.onnx"
    # Prefer existing file; default to pt
    if onnx.is_file() and not pt.is_file():
        return onnx
    return pt


def get_yolo_model_path() -> Path:
    """Return default YOLO11s model path."""
    _ensure_model_dirs()
    return INTERNAL_YOLO_DIR / "best.pt"


def ensure_ctd_available() -> Path:
    """Download ComicTextDetector if missing. Returns CTD dir."""
    _ensure_model_dirs()
    pt_path = INTERNAL_CTD_DIR / "comictextdetector.pt"
    onnx_path = INTERNAL_CTD_DIR / "comictextdetector.pt.onnx"
    if pt_path.is_file() or onnx_path.is_file():
        logger.info("CTD model already present")
        return INTERNAL_CTD_DIR
    # Prefer .pt (CUDA), fallback .onnx is handled by ensure logic
    logger.info(f"Downloading CTD model from {CTD_MODEL_URL} ...")
    result = download_file(CTD_MODEL_URL, INTERNAL_CTD_DIR)
    if result is None or not result.is_file():
        logger.warning(f"CTD .pt download failed, trying ONNX: {CTD_ONNX_URL}")
        result = download_file(CTD_ONNX_URL, INTERNAL_CTD_DIR)
        if result is None or not result.is_file():
            raise RuntimeError(f"Failed to download CTD model from {CTD_MODEL_URL} / {CTD_ONNX_URL}")
    logger.info(f"CTD model ready: {result} ({result.stat().st_size/1024/1024:.1f} MB)")
    return INTERNAL_CTD_DIR


def ensure_yolo11s_available() -> Path:
    """Download YOLO11s lordtrilink v0 if missing. Returns YOLO dir."""
    _ensure_model_dirs()
    pt_path = INTERNAL_YOLO_DIR / "best.pt"
    onnx_path = INTERNAL_YOLO_DIR / "best.onnx"
    if pt_path.is_file() or onnx_path.is_file():
        logger.info("YOLO11s model already present")
        return INTERNAL_YOLO_DIR
    logger.info(f"Downloading YOLO11s model from {YOLO11S_URL} ...")
    result = download_file(YOLO11S_URL, INTERNAL_YOLO_DIR)
    if result is None or not result.is_file():
        logger.warning(f"YOLO11s .pt download failed, trying ONNX: {YOLO11S_ONNX_URL}")
        result = download_file(YOLO11S_ONNX_URL, INTERNAL_YOLO_DIR)
        if result is None or not result.is_file():
            raise RuntimeError(f"Failed to download YOLO11s model from {YOLO11S_URL} / {YOLO11S_ONNX_URL}")
    # Ensure filename is best.pt if server returned different name (e.g. best.pt?download)
    if result.name != "best.pt" and result.suffix == ".pt":
        try:
            target = INTERNAL_YOLO_DIR / "best.pt"
            if not target.exists():
                result.rename(target)
                result = target
        except Exception:
            pass
    logger.info(f"YOLO11s model ready: {result} ({result.stat().st_size/1024/1024:.1f} MB)")
    return INTERNAL_YOLO_DIR


def ensure_bubble_detector_available(model_dir: Path) -> None:
    """No-op — bubble detector removed, only Hi-SAM is supported."""
    logger.debug("Bubble detector not used (Hi-SAM only). Skipping.")


# ========== Inpainting Models Registry ==========
# Each model: name -> (url, sha256_or_None, format)
# Supported formats: .pt (TorchScript), .ckpt (pickle), .safetensors
INPAINTING_MODELS = {
    # --- LaMa Architecture (compatible with simple-lama-inpainting) ---
    "anime-manga-big-lama": (
        "https://github.com/Sanster/models/releases/download/AnimeMangaInpainting/anime-manga-big-lama.pt",
        "479d3afdcb7ed2fd944ed4ebcc39ca45b33491f0f2e43eb1000bd623cfb41823",
        ".pt",
    ),
    "big-lama": (
        "https://github.com/Sanster/models/releases/download/AnimeMangaInpainting/big-lama.pt",
        None,
        ".pt",
    ),
    "anime-manga-inpainting": (
        "https://huggingface.co/dreMaz/AnimeMangaInpainting/resolve/main/lama_large_512px.ckpt",
        None,
        ".ckpt",
    ),
    "anime-big-lama-hf": (
        "https://huggingface.co/df1412/anime-big-lama/resolve/main/anime-manga-big-lama.pt",
        None,
        ".pt",
    ),
    "lama-manga": (
        "https://huggingface.co/mayocream/lama-manga/resolve/main/lama-manga.safetensors",
        None,
        ".safetensors",
    ),
}

# Default model name
DEFAULT_INPAINTING_MODEL = "anime-manga-big-lama"


def download_named_inpainting_model(name: str, save_dir: Path) -> Path | None:
    """Download a named inpainting model from the registry."""
    _ensure_model_dirs()
    if name not in INPAINTING_MODELS:
        logger.error(f"Unknown inpainting model name: {name}")
        logger.info(f"Available models: {', '.join(INPAINTING_MODELS.keys())}")
        return None
    url, sha, fmt = INPAINTING_MODELS[name]
    save_dir.mkdir(parents=True, exist_ok=True)
    return download_file(url, save_dir, sha_hash=sha)


def check_hash(file_path: Path, sha_hash: str) -> bool:
    """Check if the file has the expected sha256 hash."""
    return sha256(file_path.read_bytes()).hexdigest() == sha_hash


def download_file(url: str, save_dir: Path, sha_hash: str = None) -> Path | None:
    """Download a file from a url and return the path to the downloaded file."""
    response = requests.get(url, stream=True, timeout=120)
    try:
        response.raise_for_status()
    except requests.exceptions.HTTPError as e:
        print(f"Error downloading file from url: {url}\n{e}")
        print("Manually download the file and save the path to it in a profile's settings.")
        return None

    file_name = os.path.basename(url)
    save_path = save_dir / file_name

    save_dir.mkdir(parents=True, exist_ok=True)
    if save_path.is_file():
        save_path.unlink()
    elif save_path.is_dir():
        save_path.rmdir()

    file_size = int(response.headers.get("Content-Length", 0))
    chunk_size = 8192

    with open(save_path, "wb") as f, tqdm.tqdm(total=file_size, unit="B", unit_scale=True) as pbar:
        for chunk in response.iter_content(chunk_size=chunk_size):
            if chunk:
                f.write(chunk)
                pbar.update(len(chunk))

    if not save_path.exists():
        print(f"Error downloading file from url: {url}\nFailed to save the file to {save_path}")
    if sha_hash is not None:
        if not check_hash(save_path, sha_hash):
            print(
                f"Error downloading file from url: {url}\nThe file content is different from expected."
            )
            save_path.unlink()
            return None

    return save_path


def get_old_inpainting_model_path(config) -> Path:
    """Get the path to the old inpainting model."""
    return config.get_model_cache_dir() / "big-lama.pt"


def get_inpainting_model_path(config) -> Path:
    """
    Get the path to the inpainting model from the internal sandbox.
    """
    env_override = os.getenv('PCL_INPAINTING_MODEL')
    if env_override:
        env_path = Path(env_override)
        if env_path.is_file():
            logger.debug(f"Inpainting Model Path (Env Override): {env_path}")
            return env_path
        else:
            logger.warning(f"PCL_INPAINTING_MODEL is set but does not point to a file: {env_override}")

    if hasattr(config, 'inpainter') and config.inpainter.custom_inpainting_model_path:
        custom_path = Path(config.inpainter.custom_inpainting_model_path)
        if custom_path.is_file():
            logger.debug(f"Inpainting Model Path (Profile Custom Path): {custom_path}")
            return custom_path

    model_name = f"{DEFAULT_INPAINTING_MODEL}.pt"

    if hasattr(config, 'inpainter') and config.inpainter.inpainting_model_name:
        raw_name = config.inpainter.inpainting_model_name
        model_name = f"{raw_name}.pt" if not raw_name.endswith((".pt", ".ckpt", ".safetensors")) else raw_name
    elif hasattr(config, 'config') and config.config.has_option('Inpainter', 'inpainting_model_name'):
        model_name = config.config.get('Inpainter', 'inpainting_model_name')

    model_path = INTERNAL_INPAINTING_DIR / model_name
    logger.debug(f"Inpainting Model Path (Sandboxed): {model_path}")
    return model_path


def download_inpainting_model(cache_dir: Path | None = None) -> Path | None:
    """Download the default inpainting model."""
    _ensure_model_dirs()
    save_dir = INTERNAL_INPAINTING_DIR
    print(f"Downloading inpainting model to {save_dir}...")
    url, sha, _ = INPAINTING_MODELS[DEFAULT_INPAINTING_MODEL]
    return download_file(url, save_dir, sha_hash=sha)


def is_inpainting_downloaded(config) -> bool:
    """Check if the inpainting model is downloaded."""
    return get_inpainting_model_path(config).is_file()


def has_old_inpainting_model(config) -> bool:
    """Check if the old inpainting model is downloaded."""
    return get_old_inpainting_model_path(config).is_file()


def check_upgrade_inpainting_model(config) -> None:
    if has_old_inpainting_model(config):
        logger.info(
            "A new version of the inpainting model is available.\n"
            "Switching to the new model automatically."
        )
        get_old_inpainting_model_path(config).unlink()
        download_inpainting_model()
    else:
        download_inpainting_model()


def ensure_inpainting_available(config) -> None:
    """Ensure that the inpainting model is available, download it if it's not."""
    model_path = get_inpainting_model_path(config)

    if model_path.is_file():
        return

    # Check torch.hub cache (simple_lama_inpainting downloads there) - check both default and selected name
    for cand in [model_path.name, f"{DEFAULT_INPAINTING_MODEL}.pt"]:
        torch_hub_path = Path(os.environ.get("TORCH_HOME", "")) / "hub" / "checkpoints" / cand
        if torch_hub_path.is_file():
            model_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(torch_hub_path, model_path)
            logger.info(f"Copied inpainting model from torch hub cache: {model_path}")
            return

    # Resolve which model was actually selected (respect master.conf choice)
    selected = DEFAULT_INPAINTING_MODEL
    try:
        if hasattr(config, 'inpainter') and getattr(config.inpainter, 'inpainting_model_name', None):
            raw = config.inpainter.inpainting_model_name.strip()
            if raw:
                selected = raw
        elif hasattr(config, 'inpainting_model_name') and config.inpainting_model_name:
            selected = config.inpainting_model_name.strip()
    except Exception:
        pass
    # Normalize name (strip extension for registry lookup)
    lookup = selected.replace(".pt","").replace(".ckpt","").replace(".safetensors","")
    if lookup not in INPAINTING_MODELS:
        lookup = DEFAULT_INPAINTING_MODEL
        logger.warning(f"Unknown inpainting model '{selected}', falling back to {DEFAULT_INPAINTING_MODEL}")

    logger.info(f"Downloading inpainting model '{lookup}' to {model_path}...")
    url, sha, _ = INPAINTING_MODELS[lookup]
    result = download_file(url, model_path.parent, sha_hash=sha)
    if result is None or not result.is_file():
        raise RuntimeError(f"Failed to download inpainting model '{lookup}'")
    if result != model_path:
        result.rename(model_path)
    logger.info(f"Inpainting model '{lookup}' downloaded successfully -> {model_path}")


def list_available_inpainting_models() -> dict:
    """Return a dict of available inpainting models with their info."""
    result = {}
    for name, (url, sha, fmt) in INPAINTING_MODELS.items():
        size_est = "~200MB" if "lama" in name.lower() else "varies"
        result[name] = {
            "url": url,
            "format": fmt,
            "size": size_est,
            "sha256": sha,
        }
    return result


def download_models(config, force: bool = False, cuda: bool = False, cpu: bool = False) -> None:
    """Download the models needed by the pipeline."""
    ensure_inpainting_available(config)


def delete_models() -> None:
    """Delete the downloaded inpainting models."""
    inpainting_dir = INTERNAL_INPAINTING_DIR
    if inpainting_dir.is_dir():
        logger.info(f"Deleting inpainting model directory: {inpainting_dir}")
        shutil.rmtree(inpainting_dir)
    else:
        logger.warning(f"Inpainting model directory does not exist: {inpainting_dir}")
