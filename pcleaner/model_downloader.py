import os
import shutil
from hashlib import sha256
from pathlib import Path

import requests
import tqdm
from loguru import logger
import pcleaner.cli_utils as cu
from pcleaner.runtime.device import cuda_available


# ========== INTERNAL SANDBOXING: ALL MODELS ISOLATED IN pcleaner/models/ ==========
INTERNAL_MODELS_DIR = Path(__file__).resolve().parent / "models"
INTERNAL_INPAINTING_DIR = INTERNAL_MODELS_DIR / "inpainting"
INTERNAL_TEXT_DETECTOR_DIR = INTERNAL_MODELS_DIR / "text_detector"
INTERNAL_OCR_DIR = INTERNAL_MODELS_DIR / "ocr"


def _ensure_model_dirs():
    """Create internal model directories on first use (lazy, not at import)."""
    INTERNAL_INPAINTING_DIR.mkdir(parents=True, exist_ok=True)
    INTERNAL_TEXT_DETECTOR_DIR.mkdir(parents=True, exist_ok=True)
    INTERNAL_OCR_DIR.mkdir(parents=True, exist_ok=True)

# Legacy model URLs (kept for backward compatibility)
MODEL_URL = "https://github.com/zyddnys/manga-image-translator/releases/download/beta-0.3/"
TORCH_MODEL_NAME = "comictextdetector.pt"
TORCH_SHA256 = "1f90fa60aeeb1eb82e2ac1167a66bf139a8a61b8780acd351ead55268540cccb"
CV2_MODEL_NAME = "comictextdetector.pt.onnx"
CV2_SHA256 = "1a86ace74961413cbd650002e7bb4dcec4980ffa21b2f19b86933372071d718f"
OCR_DIR_NAME = "models--kha-white--manga-ocr-base"

# Bubble-aware text detector (HuggingFace)
BUBBLE_HF_MODEL_ID = "ogkalu/comic-text-and-bubble-detector"
# Primary and alternative inpainting model URLs. Add more entries here as needed.
INPAINTING_URL = "https://github.com/Sanster/models/releases/download/AnimeMangaInpainting/anime-manga-big-lama.pt"
INPAINTING_SHA256 = "479d3afdcb7ed2fd944ed4ebcc39ca45b33491f0f2e43eb1000bd623cfb41823"
# Alternative model names supported by the interactive selector. The SHA for alternatives may be unknown
# so the download will skip hash verification if not provided.
INPAINTING_ALTERNATIVES = {
    "anime-manga-big-lama.pt": (INPAINTING_URL, INPAINTING_SHA256),
    # Big-lama is a commonly-referenced filename -- use same base path if available.
    # If you have a more accurate URL/SHA, replace the URL below.
    "big-lama.pt": (INPAINTING_URL.replace("anime-manga-big-lama.pt", "big-lama.pt"), None),
}


def download_named_inpainting_model(name: str, save_dir: Path) -> Path | None:
    """
    Download a named inpainting model from the known alternatives mapping.

    :param name: The filename key (e.g., 'anime-manga-big-lama.pt' or 'big-lama.pt').
    :param save_dir: Directory to save the model into.
    :return: Path to the downloaded file or None on failure.
    """
    _ensure_model_dirs()
    if name not in INPAINTING_ALTERNATIVES:
        logger.error(f"Unknown inpainting model name: {name}")
        return None
    url, sha = INPAINTING_ALTERNATIVES[name]
    save_dir.mkdir(parents=True, exist_ok=True)
    return download_file(url, save_dir, sha_hash=sha)


def check_hash(file_path: Path, sha_hash: str) -> bool:
    """
    Check if the file has the expected sha256 hash.

    :param file_path: The path to the file.
    :param sha_hash: The expected sha256 hash.
    :return: True if the file has the expected hash.
    """
    return sha256(file_path.read_bytes()).hexdigest() == sha_hash


def download_file(url: str, save_dir: Path, sha_hash: str = None) -> Path | None:
    """
    Download a file from a url and return the path to the downloaded file.

    :param url: The url to download the file from.
    :param save_dir: The directory where the file will be saved.
    :param sha_hash: The sha256 hash of the file. If it doesn't match, the download will be aborted.
    :return: The path to the downloaded file.
    """
    response = requests.get(url, stream=True, timeout=120)
    try:
        response.raise_for_status()
    except requests.exceptions.HTTPError as e:
        print(f"Error downloading file from url: {url}\n{e}")
        print("Manually download the file and save the path to it in a profile's settings.")
        return None

    file_name = os.path.basename(url)
    save_path = save_dir / file_name

    # Ensure the file is not already downloaded, and the folder exists.
    save_dir.mkdir(parents=True, exist_ok=True)
    if save_path.is_file():
        save_path.unlink()
    elif save_path.is_dir():
        save_path.rmdir()

    file_size = int(response.headers.get("Content-Length", 0))
    chunk_size = 8192  # 8KB

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


def download_torch_model() -> Path | None:
    """
    Download the torch model and return the path to the downloaded file.
    Saved in pcleaner/models/text_detector/ for persistence across restarts.

    :return: The path to the downloaded file. Or None if the download failed.
    """
    _ensure_model_dirs()
    print(f"Downloading Torch model for CUDA to {INTERNAL_TEXT_DETECTOR_DIR}...")
    return download_file(MODEL_URL + TORCH_MODEL_NAME, INTERNAL_TEXT_DETECTOR_DIR, sha_hash=TORCH_SHA256)


def download_cv2_model() -> Path | None:
    """
    Download the cv2 model and return the path to the downloaded file.
    Saved in pcleaner/models/text_detector/ for persistence across restarts.

    :return: The path to the downloaded file. Or None if the download failed.
    """
    _ensure_model_dirs()
    print(f"Downloading OpenCV model for CPU to {INTERNAL_TEXT_DETECTOR_DIR}...")
    return download_file(MODEL_URL + CV2_MODEL_NAME, INTERNAL_TEXT_DETECTOR_DIR, sha_hash=CV2_SHA256)


def get_old_inpainting_model_path(config) -> Path:
    """
    Get the path to the inpainting model.

    :return: The path to the inpainting model.
    """
    return config.get_model_cache_dir() / "big-lama.pt"


def get_inpainting_model_path(config) -> Path:
    """
    Get the path to the inpainting model from the internal sandbox.

    All inpainting models are stored strictly within pcleaner/models/inpainting/
    to prevent fallback to user home directories or HuggingFace caches.

    Environment override: PCL_INPAINTING_MODEL can be set to the absolute path of a model file
    to force using that model (useful for interactive selection).

    :param config: Configuration object
    :return: Path to the inpainting model
    """
    import os
    # If overridden via environment variable (set by interactive selector), honor it first.
    env_override = os.getenv('PCL_INPAINTING_MODEL')
    if env_override:
        env_path = Path(env_override)
        if env_path.is_file():
            logger.debug(f"🔒 Inpainting Model Path (Env Override): {env_path}")
            return env_path
        else:
            logger.warning(f"PCL_INPAINTING_MODEL is set but does not point to a file: {env_override}")

    # 1️⃣ ميزة البروفايل الجديدة: نتحقق أولاً هل وضع المستخدم مساراً مخصصاً بالكامل داخل البروفايل؟
    if hasattr(config, 'inpainter') and config.inpainter.custom_inpainting_model_path:
        custom_path = Path(config.inpainter.custom_inpainting_model_path)
        if custom_path.is_file():
            logger.debug(f"🔒 Inpainting Model Path (Profile Custom Path): {custom_path}")
            return custom_path

    # Get model name from config, default to anime-manga-big-lama.pt
    model_name = "anime-manga-big-lama.pt"
    
    # 2️⃣ ميزة البروفايل الجديدة: نرى هل حدد اسماً مخصصاً للموديل داخل البروفايل؟
    if hasattr(config, 'inpainter') and config.inpainter.inpainting_model_name:
        raw_name = config.inpainter.inpainting_model_name
        model_name = f"{raw_name}.pt" if not raw_name.endswith(".pt") else raw_name
    # كود التحقق القديم الخاص بك (يظل كاحتياط في حال لم يجد الإعداد في البروفايل)
    elif hasattr(config, 'config') and config.config.has_option('Models', 'inpainting_model_name'):
        model_name = config.config.get('Models', 'inpainting_model_name')

    # STRICT INTERNAL SANDBOXING: All models in pcleaner/models/inpainting/
    model_path = INTERNAL_INPAINTING_DIR / model_name
    logger.debug(f"🔒 Inpainting Model Path (Sandboxed): {model_path}")
    return model_path


def download_inpainting_model(cache_dir: Path | None = None) -> Path | None:
    """
    Download the inpainting model and return the path to the downloaded file.
    The model is always saved to the persistent internal path: pcleaner/models/inpainting/

    :param cache_dir: Ignored. Kept for backward compatibility.
    :return: The path to the downloaded file. Or None if the download failed.
    """
    _ensure_model_dirs()
    save_dir = INTERNAL_INPAINTING_DIR
    print(f"Downloading inpainting model to {save_dir}...")
    return download_file(INPAINTING_URL, save_dir, sha_hash=INPAINTING_SHA256)


def is_inpainting_downloaded(config) -> bool:
    """
    Check if the inpainting model is downloaded.

    :return: True if the inpainting model is downloaded.
    """
    return get_inpainting_model_path(config).is_file()


def has_old_inpainting_model(config) -> bool:
    """
    Check if the old inpainting model is downloaded.

    :return: True if the old inpainting model is downloaded.
    """
    return get_old_inpainting_model_path(config).is_file()


def check_upgrade_inpainting_model(config) -> None:
    # Check if we have the old version sitting around and offer to
    # delete or keep it.
    if has_old_inpainting_model(config):
        if cu.get_confirmation(
            "A new version of the inpainting model is available.\n"
            "You can delete the model later if you don't want to upgrade yet.\n"
            "Switch to the new model?",
            default=True,
        ):
            get_old_inpainting_model_path(config).unlink()
            download_inpainting_model()
        else:
            print("Old model kept.")
            shutil.move(get_old_inpainting_model_path(config), get_inpainting_model_path(config))
    else:
        download_inpainting_model()


def ensure_inpainting_available(config) -> None:
    """
    Ensure that the inpainting model is available, download it if it's not.
    Downloads the correct anime-tuned model (Sanster/anime-manga-big-lama.pt),
    NOT the generic LaMa (advimman/big-lama.pt).
    """
    model_path = get_inpainting_model_path(config)

    if model_path.is_file():
        return

    logger.info(f"Downloading inpainting model to {model_path}...")
    result = download_file(INPAINTING_URL, model_path.parent, sha_hash=INPAINTING_SHA256)
    if result is None or not result.is_file():
        raise RuntimeError("Failed to download inpainting model")
    # Rename to match the expected model name if download_file used a different filename
    if result != model_path:
        result.rename(model_path)
    logger.info("Inpainting model downloaded successfully.")


def download_models(config, force: bool = False, cuda: bool = False, cpu: bool = False) -> None:
    """
    Download the models needed by PanelCleaner.

    The legacy comictextdetector.pt / .onnx models are no longer downloaded by default
    — they have been replaced by the BubbleAwareTextDetector (HuggingFace RT-DETRv2).
    If ``force`` is True, legacy models are also re-downloaded for backward compatibility.

    :param config: The config to save the paths in (legacy fields preserved for compat).
    :param force: If True, re-downloads legacy models even if already present.
    :param cuda: Ignored (kept for backward compatibility).
    :param cpu: Ignored (kept for backward compatibility).
    :return:
    """

    # Legacy model paths: only download/re-download if force is True.
    if force:
        if cuda or (not cuda and not cpu):
            config.default_torch_model_path = download_torch_model()
        if cpu or (not cuda and not cpu):
            config.default_cv2_model_path = download_cv2_model()
        config.save()

    # Download the inpainting model.
    ensure_inpainting_available(config)

    # Download the bubble-aware text detector (replaces legacy comictextdetector).
    ensure_bubble_detector_available()

    # Also load the OCR model (profile model name if available, otherwise default).
    # Lazy import: MangaOcr is heavy and only needed when explicitly downloading.
    from manga_ocr import MangaOcr as _MangaOcr
    ocr_model_name = getattr(config, "ocr_model_name", None)
    if ocr_model_name:
        _MangaOcr(ocr_model_name)
    else:
        _MangaOcr()


def get_text_detector_model_dir() -> Path:
    """
    Get the path to the text detector model directory (INTERNAL SANDBOX).

    All text detector models (including HuggingFace bubble detector)
    are stored strictly within pcleaner/models/text_detector/
    to prevent fallback to ~/.cache/ or HuggingFace caches.

    :return: The path to the text detector model directory
    """
    _ensure_model_dirs()
    logger.debug(f"Text Detector Model Directory (Sandboxed): {INTERNAL_TEXT_DETECTOR_DIR}")
    return INTERNAL_TEXT_DETECTOR_DIR


def is_bubble_detector_downloaded() -> bool:
    """
    Check if the bubble text detector model is already on disk.

    HuggingFace stores models in a subdirectory named
    models--<org>--<name>  inside the cache_dir.

    :return: True if the model is already cached locally.
    """
    model_dir = get_text_detector_model_dir()
    hf_cache_subdir = model_dir / f"models--{BUBBLE_HF_MODEL_ID.replace('/', '--')}"
    return hf_cache_subdir.exists()


def ensure_bubble_detector_available(model_dir: Path | None = None) -> Path:
    """
    Ensure the bubble-aware text detector (ogkalu/comic-text-and-bubble-detector)
    is downloaded and cached in the internal sandboxed directory.

    This ONLY downloads the model files (does NOT load into GPU memory).
    It is safe to call at server startup or before any pipeline run.

    :param model_dir: Directory to cache the model in.
                      If None, defaults to INTERNAL_TEXT_DETECTOR_DIR.
    :return: The model directory path.
    """
    if model_dir is None:
        model_dir = get_text_detector_model_dir()

    if is_bubble_detector_downloaded():
        logger.info(
            f"Bubble text detector already cached at {model_dir}, skipping download."
        )
        return model_dir

    logger.info(f"Downloading bubble text detector from {BUBBLE_HF_MODEL_ID} to {model_dir}...")
    from transformers import RTDetrV2ForObjectDetection, RTDetrImageProcessor

    _ensure_model_dirs()
    RTDetrImageProcessor.from_pretrained(BUBBLE_HF_MODEL_ID, cache_dir=str(model_dir))
    RTDetrV2ForObjectDetection.from_pretrained(BUBBLE_HF_MODEL_ID, cache_dir=str(model_dir))
    logger.info("Bubble text detector downloaded successfully.")
    return model_dir


def get_ocr_model_directory() -> Path:
    """
    Get the path to the OCR model directory (INTERNAL SANDBOX).
    
    All OCR models are stored strictly within pcleaner/models/ocr/
    to prevent fallback to ~/.cache/ or HuggingFace caches.

    :return: The path to the OCR model directory
    """
    logger.debug(f"OCR Model Directory (Sandboxed): {INTERNAL_OCR_DIR}")
    return INTERNAL_OCR_DIR


def is_ocr_downloaded() -> bool:
    """
    Check if the OCR model is downloaded.

    :return: True if the OCR model is downloaded.
    """
    return get_ocr_model_directory().is_dir()


def delete_models(text_detector_cache_dir: Path) -> None:
    """
    Delete the downloaded models.

    :param text_detector_cache_dir: The path to the text detector cache directory.
    """
    ocr_dir = get_ocr_model_directory()
    if ocr_dir.is_dir():
        logger.info(f"Deleting OCR model directory: {ocr_dir}")
        shutil.rmtree(ocr_dir)
    else:
        logger.warning(f"OCR model directory does not exist and cannot be deleted: {ocr_dir}")

    if text_detector_cache_dir.is_dir():
        logger.info(f"Deleting text detector cache directory: {text_detector_cache_dir}")
        shutil.rmtree(text_detector_cache_dir)
