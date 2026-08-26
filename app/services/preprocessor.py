import json
import sys
import re
import shutil
from collections import Counter
from copy import copy
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw
from loguru import logger

import app.config as cfg
import app.services.model_downloader as md
import app.services.output_structures as ost
import app.services.structures as st


# ============================================
# Hi-SAM Text Detection (proper integration)
# ============================================

_hisam_amg = None
_hisam_model_type = None
_hisam_device = None


def _load_hisam_model(model_type: str = "vit_b", device: str = "cuda"):
    """Load Hi-SAM model with AutoMaskGenerator for proper text detection."""
    global _hisam_amg, _hisam_model_type, _hisam_device

    if _hisam_amg is not None and _hisam_model_type == model_type and _hisam_device == device:
        return _hisam_amg

    # Free previous model if switching types or device (e.g. vit_h cuda OOM -> vit_h cpu)
    if _hisam_amg is not None and (_hisam_model_type != model_type or _hisam_device != device):
        try:
            del _hisam_amg
            _hisam_amg = None
            import gc
            gc.collect()
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
            logger.info(f"♻️ Unloaded previous Hi-SAM {_hisam_model_type}/{_hisam_device} to free VRAM")
        except Exception:
            pass

    # Ensure checkpoints are downloaded
    md.ensure_hi_sam_available(model_type)

    # Add hi_sam_src to Python path
    hi_sam_src = str(md.HI_SAM_SRC_DIR)
    if hi_sam_src not in sys.path:
        sys.path.insert(0, hi_sam_src)

    try:
        from hi_sam.modeling.build import model_registry
        from hi_sam.modeling.auto_mask_generator import AutoMaskGenerator

        # Build a minimal args object matching what Hi-SAM expects
        class _Args:
            pass
        args = _Args()
        args.model_type = model_type
        args.checkpoint = str(md.INTERNAL_TEXT_DETECTOR_DIR / md.HISAM_DEFAULT_DETECTION_CKPT)
        args.checkpoint_dir = str(md.INTERNAL_TEXT_DETECTOR_DIR)
        args.hier_det = True
        args.attn_layers = 1
        args.prompt_len = 12
        args.device = device

        logger.info(f"Loading Hi-SAM model ({model_type})...")
        hisam_model = model_registry[model_type](args)
        hisam_model.eval()
        hisam_model.to(device)

        amg = AutoMaskGenerator(hisam_model)

        _hisam_amg = amg
        _hisam_model_type = model_type
        _hisam_device = device
        logger.info(f"Hi-SAM model loaded ({model_type}/{device}) with AutoMaskGenerator")
        return amg

    except Exception as e:
        logger.error(f"Failed to load Hi-SAM model: {e}")
        raise


def _detect_text_hisam(image: Image.Image, model_type: str = "vit_b", device: str = "cuda"):
    """
    Run text detection using Hi-SAM's AutoMaskGenerator.predict_text_detection().
    Returns a list of bounding boxes in [x1, y1, x2, y2] format.
    """
    amg = _load_hisam_model(model_type, device)

    # Convert PIL to numpy RGB
    image_np = np.array(image.convert("RGB"))

    # Set image in AutoMaskGenerator
    amg.set_image(image_np)

    # Run text detection
    # zero_shot=False → uses NMS path → returns individual masks (not merged)
    # fg_points_num=1500 (paper default) for better small-text recall
    # nms_thresh=0.3 to keep more overlapping masks (less aggressive suppression)
    # Tuned for production: 900 points + 150 batch = 40% faster, same recall for manga small text
    # nms 0.42 suppresses false positives (was 0.3 -> 266 boxes), vit_b recommended in master.conf
    masks, scores = amg.predict_text_detection(
        from_low_res=False,
        fg_points_num=900,
        batch_points_num=150,
        score_thresh=0.5,
        nms_thresh=0.42,
        zero_shot=False,
        dataset='totaltext',
    )

    if masks is None or len(masks) == 0:
        return []

    # Convert masks to bounding boxes
    boxes = []
    for i in range(len(masks)):
        mask = masks[i]
        # mask shape: (1, H, W) or (H, W)
        if mask.ndim == 3:
            mask_2d = mask[0]
        else:
            mask_2d = mask

        # Find bounding box from mask
        y_indices, x_indices = np.where(mask_2d > 0)
        if len(y_indices) == 0 or len(x_indices) == 0:
            continue

        x1 = int(x_indices.min())
        y1 = int(y_indices.min())
        x2 = int(x_indices.max()) + 1
        y2 = int(y_indices.max()) + 1

        # Clamp to image bounds
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(image.width, x2)
        y2 = min(image.height, y2)

        area = (x2 - x1) * (y2 - y1)
        if area >= 50:  # Minimum ~7x7 pixels — catch small manga text/SFX
            boxes.append([x1, y1, x2, y2])

    return boxes


# ============================================
# ComicTextDetector (CTD) — zyddnys/manga-image-translator
# ============================================
_ctd_model = None
_ctd_model_path_cached = None
_ctd_device_cached = None


def _load_ctd_model(model_path: Path, device: str = "cuda"):
    """Lazy-load CTD model (torch .pt). Returns model object or None for ONNX path."""
    global _ctd_model, _ctd_model_path_cached, _ctd_device_cached
    if _ctd_model is not None and _ctd_model_path_cached == str(model_path) and _ctd_device_cached == device:
        return _ctd_model
    # Try torch-based CTD (common: comictextdetector.pt is a YOLO-like ckpt)
    try:
        ckpt = torch.load(str(model_path), map_location=device)
        # Some CTD pts are state_dict, others are full model
        if isinstance(ckpt, dict) and "model" in ckpt:
            ckpt = ckpt["model"]
        _ctd_model = ckpt
        _ctd_model_path_cached = str(model_path)
        _ctd_device_cached = device
        logger.info(f"CTD torch model loaded: {model_path.name} ({device})")
        return _ctd_model
    except Exception as e:
        logger.debug(f"CTD torch load failed ({e}), will try ONNX/cv2 fallback")
        return None


def _detect_text_ctd(image: Image.Image, model_path: Path | str | None = None, device: str = "cuda"):
    """
    Run ComicTextDetector inference. Returns boxes [x1,y1,x2,y2].
    Tries backends in order: comic_text_detector lib → onnxruntime → cv2.dnn → torch → empty fallback.
    Always returns [] on failure (pipeline continues with empty mask).
    """
    # Resolve model path
    if model_path is None or str(model_path).strip() == "":
        try:
            model_path = md.get_ctd_model_path()
        except Exception:
            model_path = Path("app/models/text_detector/ctd/comictextdetector.pt")
    else:
        model_path = Path(model_path)
    # Ensure available (auto-download like Hi-SAM)
    try:
        if not model_path.is_file():
            # try default discovery
            alt = md.get_ctd_model_path()
            if alt.is_file():
                model_path = alt
            else:
                md.ensure_ctd_available()
                model_path = md.get_ctd_model_path()
    except Exception as e:
        logger.warning(f"CTD ensure failed: {e}")

    if not Path(model_path).is_file():
        logger.warning(f"CTD model not found: {model_path}, returning 0 boxes")
        return []

    img_np = np.array(image.convert("RGB"))
    h, w = image.size[1], image.size[0]  # PIL size is (W,H)

    # 1) Try official comic_text_detector package if installed
    try:
        from comic_text_detector.inference import ComicTextDetector  # type: ignore
        det = ComicTextDetector(str(model_path), device=device if device == "cuda" and torch.cuda.is_available() else "cpu")
        # detector returns list of [x1,y1,x2,y2, score] or dict
        result = det.detect(img_np) if hasattr(det, "detect") else det(img_np)
        boxes = []
        for r in (result or []):
            if isinstance(r, dict) and "bbox" in r:
                x1, y1, x2, y2 = r["bbox"][:4]
            elif isinstance(r, (list, tuple, np.ndarray)) and len(r) >= 4:
                x1, y1, x2, y2 = r[:4]
            else:
                continue
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            if (x2 - x1) * (y2 - y1) >= 50:
                boxes.append([max(0, x1), max(0, y1), min(w, x2), min(h, y2)])
        if boxes:
            logger.debug(f"CTD (comic_text_detector lib) → {len(boxes)} boxes")
            return boxes
    except ImportError:
        pass
    except Exception as e:
        logger.debug(f"CTD comic_text_detector backend failed: {e}")

    # 2) ONNX via onnxruntime (preferred for .onnx)
    if str(model_path).endswith(".onnx") or Path(str(model_path) + ".onnx").is_file():
        onnx_path = Path(model_path) if str(model_path).endswith(".onnx") else Path(str(model_path) + ".onnx")
        if not onnx_path.is_file():
            # try alternate .onnx name
            onnx_path = Path(str(model_path)).with_suffix(".onnx")
        try:
            import onnxruntime as ort  # type: ignore
            sess = ort.InferenceSession(str(onnx_path), providers=["CUDAExecutionProvider", "CPUExecutionProvider"] if device == "cuda" else ["CPUExecutionProvider"])
            # Generic preproc: resize to 640, normalize
            inp_name = sess.get_inputs()[0].name
            img_resized = cv2.resize(img_np, (640, 640))
            img_in = img_resized.astype(np.float32) / 255.0
            img_in = np.transpose(img_in, (2, 0, 1))[None]
            outs = sess.run(None, {inp_name: img_in})
            # outs[0] expected shape [N, 6]  (x1,y1,x2,y2,conf,cls) or [1,N,6]
            preds = outs[0]
            if preds.ndim == 3:
                preds = preds[0]
            boxes = []
            for det in preds:
                if len(det) < 6:
                    continue
                x1, y1, x2, y2, conf = float(det[0]), float(det[1]), float(det[2]), float(det[3]), float(det[4])
                if conf < 0.5:
                    continue
                # Scale from 640 back to orig
                x1 = int(x1 / 640 * w); y1 = int(y1 / 640 * h); x2 = int(x2 / 640 * w); y2 = int(y2 / 640 * h)
                if (x2 - x1) * (y2 - y1) >= 50:
                    boxes.append([max(0,x1), max(0,y1), min(w,x2), min(h,y2)])
            if boxes:
                logger.debug(f"CTD (onnxruntime {onnx_path.name}) → {len(boxes)} boxes")
                return boxes
        except ImportError:
            pass
        except Exception as e:
            logger.debug(f"CTD onnxruntime failed: {e}")

    # 3) ONNX via cv2.dnn
    try:
        onnx_path = Path(model_path) if str(model_path).endswith(".onnx") else Path(str(model_path) + ".onnx")
        if onnx_path.is_file():
            net = cv2.dnn.readNetFromONNX(str(onnx_path))
            blob = cv2.dnn.blobFromImage(img_np, 1/255.0, (640, 640), swapRB=True, crop=False)
            net.setInput(blob)
            outs = net.forward()
            # outs shape handling similar to above
            if isinstance(outs, (list, tuple)):
                outs = outs[0]
            if outs.ndim == 3:
                outs = outs[0]
            boxes = []
            for det in outs:
                if len(det) < 6:
                    continue
                x1, y1, x2, y2, conf = float(det[0]), float(det[1]), float(det[2]), float(det[3]), float(det[4])
                if conf < 0.5:
                    continue
                x1 = int(x1 / 640 * w); y1 = int(y1 / 640 * h); x2 = int(x2 / 640 * w); y2 = int(y2 / 640 * h)
                if (x2 - x1) * (y2 - y1) >= 50:
                    boxes.append([max(0,x1), max(0,y1), min(w,x2), min(h,y2)])
            if boxes:
                logger.debug(f"CTD (cv2.dnn {onnx_path.name}) → {len(boxes)} boxes")
                return boxes
    except Exception as e:
        logger.debug(f"CTD cv2.dnn fallback failed: {e}")

    # 4) Torch .pt generic (try ultralytics-like or raw torch)
    # Attempt ultralytics YOLO as generic for CTD pt that is actually YOLO format
    try:
        from ultralytics import YOLO  # type: ignore
        # YOLO may support CTD pt if it's YOLO architecture
        ymodel = YOLO(str(model_path))
        results = ymodel.predict(source=img_np, device=device if device == "cuda" and torch.cuda.is_available() else "cpu", verbose=False, conf=0.5)
        boxes = []
        for r in results:
            if hasattr(r, "boxes") and r.boxes is not None:
                for b in r.boxes:
                    xyxy = b.xyxy[0].cpu().numpy().tolist() if hasattr(b.xyxy[0], "cpu") else b.xyxy[0].tolist()
                    x1, y1, x2, y2 = map(int, xyxy[:4])
                    if (x2 - x1) * (y2 - y1) >= 50:
                        boxes.append([max(0,x1), max(0,y1), min(w,x2), min(h,y2)])
        if boxes:
            logger.debug(f"CTD (ultralytics YOLO) → {len(boxes)} boxes")
            return boxes
    except ImportError:
        pass
    except Exception as e:
        logger.debug(f"CTD ultralytics failed: {e}")

    logger.warning("CTD inference fell through all backends — returning 0 boxes (install comic_text_detector / onnxruntime / ultralytics for CTD)")
    return []


# ============================================
# YOLO11s lordtrilink v0 — Ultralytics YOLO
# ============================================
_yolo_model = None
_yolo_path_cached = None
_yolo_device_cached = None


def _load_yolo_model(model_path: Path, device: str = "cuda"):
    global _yolo_model, _yolo_path_cached, _yolo_device_cached
    if _yolo_model is not None and _yolo_path_cached == str(model_path) and _yolo_device_cached == device:
        return _yolo_model
    try:
        from ultralytics import YOLO  # type: ignore
        _yolo_model = YOLO(str(model_path))
        _yolo_path_cached = str(model_path)
        _yolo_device_cached = device
        logger.info(f"YOLO11s model loaded: {Path(model_path).name} ({device})")
        return _yolo_model
    except ImportError as e:
        logger.warning(f"ultralytics not installed, cannot load YOLO11s: {e}")
        return None
    except Exception as e:
        logger.error(f"Failed to load YOLO11s model: {e}")
        return None


def _detect_text_yolo(image: Image.Image, model_path: Path | str | None = None, device: str = "cuda"):
    """
    Run YOLO11s lordtrilink inference. Returns boxes [x1,y1,x2,y2].
    Backends: ultralytics → onnxruntime → cv2.dnn (CPU fallback).
    """
    if model_path is None or str(model_path).strip() == "":
        try:
            model_path = md.get_yolo_model_path()
        except Exception:
            model_path = Path("app/models/text_detector/yolo/best.pt")
    else:
        model_path = Path(model_path)
    try:
        if not model_path.is_file():
            alt = md.get_yolo_model_path()
            if alt.is_file():
                model_path = alt
            else:
                # try onnx alt
                alt_onnx = Path(str(alt).replace(".pt", ".onnx"))
                if alt_onnx.is_file():
                    model_path = alt_onnx
                else:
                    md.ensure_yolo11s_available()
                    model_path = md.get_yolo_model_path()
                    if not model_path.is_file():
                        alt_onnx = Path(str(model_path).replace(".pt", ".onnx"))
                        if alt_onnx.is_file():
                            model_path = alt_onnx
    except Exception as e:
        logger.warning(f"YOLO ensure failed: {e}")

    if not Path(model_path).is_file():
        logger.warning(f"YOLO model not found: {model_path}, returning 0 boxes")
        return []

    img_np = np.array(image.convert("RGB"))
    w, h = image.size  # (W,H)
    use_device = device if device == "cuda" and torch.cuda.is_available() else "cpu"

    # 1) ultralytics (preferred)
    try:
        from ultralytics import YOLO  # type: ignore
        ymodel = _load_yolo_model(Path(model_path), use_device)
        if ymodel is not None:
            results = ymodel.predict(source=img_np, device=use_device, verbose=False, conf=0.3, iou=0.5)
            boxes = []
            for r in results:
                if hasattr(r, "boxes") and r.boxes is not None:
                    for b in r.boxes:
                        xyxy = b.xyxy[0].cpu().numpy().tolist() if hasattr(b.xyxy[0], "cpu") else b.xyxy[0].tolist()
                        x1, y1, x2, y2 = map(int, xyxy[:4])
                        # Clamp
                        x1 = max(0, x1); y1 = max(0, y1); x2 = min(w, x2); y2 = min(h, y2)
                        if (x2 - x1) * (y2 - y1) >= 50:
                            boxes.append([x1, y1, x2, y2])
            logger.debug(f"YOLO11s (ultralytics) → {len(boxes)} boxes")
            return boxes
    except ImportError:
        pass
    except Exception as e:
        logger.debug(f"YOLO ultralytics failed ({e}), trying ONNX fallback")
        # fallback to ONNX below

    # 2) onnxruntime fallback
    onnx_path = Path(str(model_path).replace(".pt", ".onnx")) if str(model_path).endswith(".pt") else Path(model_path)
    if onnx_path.is_file():
        try:
            import onnxruntime as ort  # type: ignore
            sess = ort.InferenceSession(str(onnx_path), providers=["CUDAExecutionProvider", "CPUExecutionProvider"] if use_device == "cuda" else ["CPUExecutionProvider"])
            inp_name = sess.get_inputs()[0].name
            img_resized = cv2.resize(img_np, (640, 640))
            img_in = img_resized.astype(np.float32) / 255.0
            img_in = np.transpose(img_in, (2, 0, 1))[None]
            outs = sess.run(None, {inp_name: img_in})
            preds = outs[0]
            if preds.ndim == 3:
                preds = preds[0]
            boxes = []
            for det in preds:
                if len(det) < 6:
                    continue
                x1, y1, x2, y2, conf = float(det[0]), float(det[1]), float(det[2]), float(det[3]), float(det[4])
                if conf < 0.3:
                    continue
                x1 = int(x1 / 640 * w); y1 = int(y1 / 640 * h); x2 = int(x2 / 640 * w); y2 = int(y2 / 640 * h)
                if (x2 - x1) * (y2 - y1) >= 50:
                    boxes.append([max(0,x1), max(0,y1), min(w,x2), min(h,y2)])
            logger.debug(f"YOLO11s (onnxruntime {onnx_path.name}) → {len(boxes)} boxes")
            return boxes
        except ImportError:
            pass
        except Exception as e:
            logger.debug(f"YOLO onnxruntime failed: {e}")

        # 3) cv2.dnn
        try:
            net = cv2.dnn.readNetFromONNX(str(onnx_path))
            blob = cv2.dnn.blobFromImage(img_np, 1/255.0, (640, 640), swapRB=True, crop=False)
            net.setInput(blob)
            outs = net.forward()
            if isinstance(outs, (list, tuple)):
                outs = outs[0]
            if outs.ndim == 3:
                outs = outs[0]
            boxes = []
            for det in outs:
                if len(det) < 6:
                    continue
                x1, y1, x2, y2, conf = float(det[0]), float(det[1]), float(det[2]), float(det[3]), float(det[4])
                if conf < 0.3:
                    continue
                x1 = int(x1 / 640 * w); y1 = int(y1 / 640 * h); x2 = int(x2 / 640 * w); y2 = int(y2 / 640 * h)
                if (x2 - x1) * (y2 - y1) >= 50:
                    boxes.append([max(0,x1), max(0,y1), min(w,x2), min(h,y2)])
            logger.debug(f"YOLO11s (cv2.dnn {onnx_path.name}) → {len(boxes)} boxes")
            return boxes
        except Exception as e:
            logger.debug(f"YOLO cv2.dnn failed: {e}")

    logger.warning("YOLO11s inference fell through all backends — returning 0 boxes (install ultralytics / onnxruntime)")
    return []


def _normalize_detector_choice(raw: str) -> str:
    """Map aliases to canonical detector name."""
    m = {
        "bubble": "bubble",
        "hi-sam": "hi-sam", "hisam": "hi-sam", "hi_sam": "hi-sam",
        "ctd": "ctd", "comic_text_detector": "ctd", "comictextdetector": "ctd", "comic-text-detector": "ctd",
        "yolo": "yolo11s", "yolo11s": "yolo11s", "yolo11s_lordtrilink": "yolo11s", "yolo11s-lordtrilink": "yolo11s",
    }
    r = str(raw or "bubble").strip().lower()
    return m.get(r, r)


def generate_mask_data(
    image_paths: list[Path],
    config_general: cfg.GeneralConfig,
    config_detector: cfg.TextDetectorConfig,
    model_path: Path,
    output_dir: Path,
) -> None:
    """
    Run text detection (Hi-SAM / CTD / YOLO11s) to generate masks and box data.
    Dispatches based on config_detector.text_detector_model like inpainting model selection.
    Output is identical: mask.png + #raw.json per image.
    """
    if not image_paths:
        logger.warning("No images provided for text detection")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_type = getattr(config_detector, "hi_sam_model_type", "vit_h")
    raw_choice = getattr(config_detector, "text_detector_model", "bubble")
    detector = _normalize_detector_choice(raw_choice)
    # bubble is legacy → map to hi-sam for backward compat (or keep as hi-sam)
    if detector == "bubble":
        logger.info("ℹ️ text_detector_model=bubble (legacy) → using Hi-SAM path for compatibility")
        detector = "hi-sam"
    ctd_path = Path(getattr(config_detector, "ctd_model_path", "") or "")
    yolo_path = Path(getattr(config_detector, "yolo_model_path", "") or "")

    if detector == "ctd":
        if not ctd_path or not ctd_path.is_file():
            try:
                ctd_path = md.get_ctd_model_path()
            except Exception:
                ctd_path = Path("")
        logger.info(f"Running text detection with CTD ({ctd_path if ctd_path else 'auto'}) on {len(image_paths)} images (device={device})...")
    elif detector == "yolo11s":
        if not yolo_path or not yolo_path.is_file():
            try:
                yolo_path = md.get_yolo_model_path()
            except Exception:
                yolo_path = Path("")
        logger.info(f"Running text detection with YOLO11s lordtrilink v0 ({yolo_path if yolo_path else 'auto'}) on {len(image_paths)} images (device={device})...")
    else:
        logger.info(f"Running text detection with Hi-SAM ({model_type}) on {len(image_paths)} images...")

    # Always copy images to cache dir first, even if detection fails later
    for img_path in image_paths:
        cached_image_path = output_dir / img_path.name
        if not cached_image_path.exists():
            try:
                shutil.copy2(img_path, cached_image_path)
            except Exception as e:
                logger.error(f"Failed to copy {img_path.name} to cache: {e}")

    # Preload selected model once (like start.sh ensures)
    try:
        if detector == "ctd":
            # ensure_ctd handles download if missing
            try:
                if not (ctd_path and ctd_path.is_file()):
                    md.ensure_ctd_available()
            except Exception as e:
                logger.warning(f"CTD ensure failed: {e}")
        elif detector == "yolo11s":
            try:
                if not (yolo_path and yolo_path.is_file()):
                    md.ensure_yolo11s_available()
            except Exception as e:
                logger.warning(f"YOLO11s ensure failed: {e}")
        else:
            _load_hisam_model(model_type, device)
    except Exception as e:
        logger.error(f"Cannot load text detection model ({detector}): {e}")
        logger.warning("Skipping text detection — no masks will be generated")
        return

    for img_path in image_paths:
        try:
            image = Image.open(img_path).convert("RGB")
            cached_image_path = output_dir / img_path.name

            # Dispatch to correct detector with OOM recovery
            def _run_detection():
                if detector == "ctd":
                    return _detect_text_ctd(image, ctd_path if ctd_path and ctd_path.is_file() else None, device)
                elif detector == "yolo11s":
                    return _detect_text_yolo(image, yolo_path if yolo_path and yolo_path.is_file() else None, device)
                else:
                    return _detect_text_hisam(image, model_type, device)

            try:
                boxes = _run_detection()
            except RuntimeError as det_err:
                if "out of memory" in str(det_err).lower() or "cuda out of memory" in str(det_err).lower():
                    logger.warning(f"⚠️ CUDA OOM on {img_path.name} ({detector}/{device}), clearing cache...")
                    try:
                        torch.cuda.empty_cache()
                        import gc; gc.collect()
                    except Exception:
                        pass
                    # Fallback: CPU
                    try:
                        logger.warning(f"↩️ Retrying {img_path.name} on CPU ({detector})...")
                        if detector == "ctd":
                            boxes = _detect_text_ctd(image, ctd_path if ctd_path and ctd_path.is_file() else None, "cpu")
                        elif detector == "yolo11s":
                            boxes = _detect_text_yolo(image, yolo_path if yolo_path and yolo_path.is_file() else None, "cpu")
                        else:
                            boxes = _detect_text_hisam(image, model_type, "cpu")
                    except RuntimeError as e2:
                        if "out of memory" in str(e2).lower() and detector == "hi-sam" and model_type != "vit_b":
                            # Fallback 2: lighter vit_b on CPU (hi-sam only)
                            logger.warning(f"⚠️ Still OOM, retrying {img_path.name} with vit_b on CPU...")
                            try:
                                torch.cuda.empty_cache()
                            except Exception:
                                pass
                            boxes = _detect_text_hisam(image, "vit_b", "cpu")
                        elif "out of memory" in str(e2).lower():
                            raise
                        else:
                            raise
                else:
                    raise

            # Create mask image from boxes using ImageDraw (fast)
            mask = Image.new("L", image.size, 0)
            draw = ImageDraw.Draw(mask)
            for box in boxes:
                x1, y1, x2, y2 = box
                draw.rectangle([x1, y1, x2, y2], fill=255)

            # Save mask
            mask_path = output_dir / f"{img_path.stem}_mask.png"
            mask.save(mask_path)

            # Create raw JSON for prep_json_file
            raw_data = {
                "image_path": str(cached_image_path),
                "mask_path": str(mask_path),
                "original_path": str(img_path),
                "scale": 1.0,
                "blk_list": [
                    {"xyxy": box, "language": None}
                    for box in boxes
                ],
            }

            json_path = output_dir / f"{img_path.stem}#raw.json"
            json_path.write_text(json.dumps(raw_data, indent=2), encoding="utf-8")

            logger.info(f"Detected {len(boxes)} text regions in {img_path.name}")

        except Exception as e:
            logger.error(f"Failed to process {img_path.name}: {e}")
            # Create empty mask/json so pipeline continues (0 boxes) instead of crashing
            try:
                mask_path = output_dir / f"{img_path.stem}_mask.png"
                if not mask_path.exists():
                    Image.new("L", Image.open(img_path).size, 0).save(mask_path)
                json_path = output_dir / f"{img_path.stem}#raw.json"
                if not json_path.exists():
                    cached_image_path = output_dir / img_path.name
                    raw_data = {"image_path": str(cached_image_path), "mask_path": str(mask_path), "original_path": str(img_path), "scale": 1.0, "blk_list": []}
                    json_path.write_text(json.dumps(raw_data, indent=2), encoding="utf-8")
            except Exception:
                pass
            try:
                torch.cuda.empty_cache()
                import gc; gc.collect()
            except Exception:
                pass

        finally:
            # Free per-image GPU memory (important when processing 34 parts sequentially)
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
            # Close image to free PIL memory
            try:
                image.close()
            except Exception:
                pass

    logger.info(f"Text detection complete for {len(image_paths)} images")


def are_boxes_aligned(a: st.Box, b: st.Box, axis: int, tolerance: int = 10) -> bool:
    if axis == 0:
        return (a.y1 - tolerance) < b.y2 and (b.y1 - tolerance) < a.y2
    else:
        return (a.x1 - tolerance) < b.x2 and (b.x1 - tolerance) < a.x2


def cluster_boxes(
    bboxes: list[st.Box], axis: int = 0, tolerance: int = 10, depth: int = 0
) -> list[list[st.Box]]:
    if depth > 10:
        return [bboxes]

    clusters = []
    for bbox in bboxes:
        for cluster in clusters:
            if any(are_boxes_aligned(b, bbox, axis=axis, tolerance=tolerance) for b in cluster):
                cluster.append(bbox)
                break
        else:
            clusters.append([bbox])

    if axis == 0:
        clusters.sort(key=lambda c: c[0].y1)
    else:
        clusters.sort(key=lambda c: c[0].x1)

    for i, cluster in enumerate(clusters):
        if len(cluster) > 1:
            clusters[i] = cluster_boxes(
                cluster, axis=1 if axis == 0 else 0, tolerance=tolerance, depth=depth + 1
            )

    return clusters


def flatten(l):
    for el in l:
        if isinstance(el, list):
            yield from flatten(el)
        else:
            yield el


def prep_json_file(
    json_file_path: Path,
    preprocessor_conf: cfg.PreprocessorConfig,
    cache_masks: bool,
) -> None:
    """
    Load the generated json file, and clean the data, leaving only the
    relevant data for the following steps.

    Check that this file wasn't already processed, and if it was, skip it.
    Processed json files have the file name ending with '#clean.json'.
    """
    logger.debug(f"Processing json file: {json_file_path}")

    if not json_file_path.name.endswith("#raw.json"):
        return None

    json_data = json.loads(json_file_path.read_text(encoding="utf-8"))

    image_path: str = json_data["image_path"]
    mask_path: str = json_data["mask_path"]
    original_path: str = json_data["original_path"]
    scale: float = json_data["scale"]
    boxes: list[st.Box] = []
    box_langs: list[None] = []

    path_gen = ost.OutputPathGenerator(Path(original_path), Path(mask_path).parent, Path(mask_path))

    for data in json_data["blk_list"]:
        box = st.Box(*data["xyxy"])
        # Only filter tiny boxes (< 10x10 pixels). Do NOT filter by suspicious_box_min_size
        # since Hi-SAM already validated the boxes as text regions.
        if box.area < preprocessor_conf.box_min_size:
            continue

        box_langs.append(data.get("language"))
        boxes.append(box)

    page_data = st.PageData(
        image_path, mask_path, original_path, scale, box_langs, boxes, [], [], []
    )

    page_data.resolve_total_overlaps()

    page_data.grow_boxes(preprocessor_conf.box_padding_initial, st.BoxType.BOX)
    page_data.right_pad_boxes(preprocessor_conf.box_right_padding_initial, st.BoxType.BOX)

    if len(page_data.boxes) > 1:
        page_data.boxes, page_data.box_language = zip(
            *sorted(
                zip(page_data.boxes, page_data.box_language),
                key=lambda x: x[0].y1,
            )
        )
    page_data.boxes = list(page_data.boxes)
    page_data.box_language = list(page_data.box_language)

    if cache_masks:
        page_data.visualize(page_data.image_path, path_gen.boxes)

    page_data.extended_boxes = copy(page_data.boxes)

    page_data.grow_boxes(preprocessor_conf.box_padding_extended, st.BoxType.EXTENDED_BOX)
    page_data.right_pad_boxes(preprocessor_conf.box_right_padding_extended, st.BoxType.EXTENDED_BOX)

    page_data.resolve_overlaps(
        from_type=st.BoxType.EXTENDED_BOX,
        to_type=st.BoxType.MERGED_EXT_BOX,
        threshold=preprocessor_conf.box_overlap_threshold,
    )

    page_data.reference_boxes = copy(page_data.merged_extended_boxes)
    page_data.grow_boxes(preprocessor_conf.box_reference_padding, st.BoxType.REFERENCE_BOX)

    json_out_path = path_gen.clean_json
    json_out_path.write_text(page_data.to_json(), encoding="utf-8")

    if cache_masks:
        page_data.visualize(page_data.image_path, path_gen.final_boxes)
