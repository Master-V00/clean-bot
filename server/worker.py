"""
Background worker for processing image cleaning tasks
Uses background tasks queue system
"""
import os
import shutil
import asyncio
import glob
import time
from pathlib import Path
from uuid import uuid4
from server.config import settings
from server.database import db
from server.google_drive import gdrive
from server.rclone_manager import get_rclone, FastDriveManager
from loguru import logger
import json as json_module


class ImageProcessor:
    """Handle image processing tasks"""
    
    def __init__(self):
        self.temp_dir = Path(settings.TEMP_CHAPTERS_DIR)
        self.temp_dir.mkdir(parents=True, exist_ok=True)
    
    def calculate_pixel_count(self, image_paths: list) -> int:
        """Calculate total height from images.
        Returns total_height (sum of all page heights).
        Raises ValueError if any single page exceeds 16k height
        or if any dimension exceeds WebP limit of 16383.
        """
        total_height = 0
        try:
            from PIL import Image
            for image_path in image_paths:
                img = Image.open(image_path)
                width, height = img.size
                if height > 16000:
                    raise ValueError(
                        f"One of the pages exceeds the 16k height limit. "
                        f"Found page with height {height}px: {os.path.basename(image_path)}"
                    )
                if width > 16383 or height > 16383:
                    raise ValueError(
                        f"One of the pages exceeds the WebP image size limit of 16383 pixels per side. "
                        f"Found page with dimensions {width}x{height}px: {os.path.basename(image_path)}. "
                        f"The image is too large for WebP format — please resize or split it."
                    )
                total_height += height
            logger.info(f"✅ Total height count: {total_height}")
            return total_height
        except ValueError:
            raise
        except Exception as e:
            logger.error(f"❌ Failed to calculate pixel count: {e}")
            return 0
    
    def calculate_credits_needed(self, pixel_count: int) -> int:
        """Calculate credits needed based on pixel count"""
        if pixel_count > 500000:
            return -1  # Error: too large
        elif pixel_count <= 200000:
            return 1
        elif pixel_count <= 300000:
            return 2
        elif pixel_count <= 400000:
            return 3
        elif pixel_count <= 500000:
            return 4
        else:
            return 5
    
    def convert_to_webp(self, image_path: str, output_path: str, quality: int = 85) -> bool:
        """Convert image to WebP format"""
        try:
            from PIL import Image
            
            img = Image.open(image_path)
            img.save(output_path, 'WEBP', quality=quality)
            logger.info(f"✅ Image converted to WebP: {output_path}")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to convert image to WebP: {e}")
            return False
            
    def create_temp_chapter_dir(self) -> str:
        """Create a temporary directory for chapter processing"""
        unique_id = str(uuid4())
        chapter_dir = self.temp_dir / unique_id
        chapter_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"✅ Created temporary chapter directory: {chapter_dir}")
        return str(chapter_dir)
    
    def cleanup_temp_dir(self, directory: str) -> bool:
        """Delete temporary directory after processing"""
        try:
            if os.path.exists(directory):
                shutil.rmtree(directory)
                logger.info(f"✅ Cleaned up temporary directory: {directory}")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to cleanup temporary directory: {e}")
            return False


# ─── Startup preload ─────────────────────────────────────────────────────
def preload_text_detector():
    """Pre-download the BubbleAware text detector model at server startup
    into the sandboxed pcleaner/models/text_detector/ directory.
    This ensures the model is available on disk before the first user request."""
    try:
        import os
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        from pcleaner import model_downloader as md

        model_dir = md.get_text_detector_model_dir()
        md.ensure_bubble_detector_available(model_dir)
        logger.info(f"BubbleAware text detector ready at {model_dir}")
    except Exception as e:
        logger.warning(f"Could not pre-load text detector: {e}")


def preload_inpainting_model():
    """Pre-download the LaMa inpainting model at server startup (not loaded into GPU)."""
    try:
        from pcleaner.config import Profile
        from pcleaner import model_downloader as md

        profiles_dir = getattr(settings, "PROFILES_DIR", "pcleaner/profiles")
        conf_files = glob.glob(os.path.join(profiles_dir, "*.conf"))
        if conf_files:
            config = Profile().load(conf_files[0])
        else:
            config = Profile()

        md.ensure_inpainting_available(config)
        logger.info("✅ LaMa inpainting model file ready on disk")
    except Exception as e:
        logger.warning(f"⚠️ Could not pre-download inpainting model: {e}")


def preload_opencv_detector():
    """Pre-download the BubbleAware text detection model at server startup (replaces legacy OpenCV model).
    The model is saved in the sandboxed pcleaner/models/text_detector/ directory."""
    try:
        from pcleaner import model_downloader as md

        model_dir = md.get_text_detector_model_dir()
        md.ensure_bubble_detector_available(model_dir)
        logger.info(f"BubbleAware text detector ready at {model_dir}")
    except Exception as e:
        logger.warning(f"Could not pre-load text detector: {e}")


# ─── استخراج النص قبل التنظيف ───────────────────────────────────────────
def _get_pcleaner_cache_dir() -> Path | None:
    """الحصول على مسار cache directory الخاص بـ pcleaner."""
    try:
        from pcleaner.config import Config as PcleanerConfig
        pcleaner_config = PcleanerConfig()
        cache_dir = pcleaner_config.get_cleaner_cache_dir()
        logger.info(f"📂 pcleaner cache dir: {cache_dir}")
        return cache_dir
    except Exception as e:
        logger.warning(f"⚠️ Could not get pcleaner cache dir: {e}")
        return None


def _find_raw_json_files(stem: str, image_dir: str, cache_dir: Path | None) -> list[Path]:
    """
    البحث عن كل ملفات raw.json الخاصة بصورة معينة (بما في ذلك الـ splits).
    pcleaner بيقسم الصور الطويلة لأجزاء زي 124_split_1#raw.json
    
    :return: قائمة بمسارات raw.json (عدة ملفات لو الصورة اتقسمت)
    """
    all_raw = []

    if cache_dir and cache_dir.exists():
        # بحث1: الملفات اللي اسمها فيها الـ stem + _split_
        for pattern in [f"*{stem}_split_*#raw.json", f"*{stem}#raw.json"]:
            all_raw.extend(cache_dir.glob(pattern))
        
        # بحث2: في المجلدات الفرعية (splits/)
        for sub in cache_dir.iterdir():
            if sub.is_dir():
                for pattern in [f"*{stem}_split_*#raw.json", f"*{stem}#raw.json"]:
                    all_raw.extend(sub.glob(pattern))

    if not all_raw:
        # بحث3: في ~/.cache/pcleaner/ بشكل شامل
        try:
            home_cache = Path.home() / ".cache" / "pcleaner"
            if home_cache.exists():
                for pattern in [f"*{stem}_split_*#raw.json", f"*{stem}#raw.json"]:
                    all_raw.extend(home_cache.rglob(pattern))
        except Exception:
            pass

    return sorted(all_raw, key=lambda p: p.name)


def _extract_text_before_cleanup(
    image_paths: list[str],
    image_dir: str,
) -> dict[str, list[dict]]:
    """
    استخراج النصوص من الصور الأصلية قبل التنظيف.
    
    يقرأ raw.json الناتج من خطوة Detection (اللي بتنفذ قبل التنظيف)
    ويستخرج النصوص من الصورة الأصلية.
    يدعم الـ splits (الصور الطويلة بتتقسم لأجزاء).
    
    :param image_paths: مسارات الصور الأصلية
    :param image_dir: مسار المجلد (image_dir)
    :return: قاموس {stem: [{"text": ..., "type": ..., "bbox": ..., "symbol": ...}]}
    """
    results = {}
    
    cache_dir = _get_pcleaner_cache_dir()
    
    for image_path in image_paths:
        base_name = os.path.basename(image_path)
        stem = os.path.splitext(base_name)[0]
        
        # البحث عن كل ملفات raw.json (بما في ذلك splits)
        raw_json_files = _find_raw_json_files(stem, image_dir, cache_dir)
        
        if not raw_json_files:
            logger.warning(f"⚠️ No raw.json found for {base_name} (searched cache={cache_dir})")
            continue
        
        logger.info(f"📋 Found {len(raw_json_files)} raw.json file(s) for {base_name}")
        
        # قراءة ودمج كل ملفات raw.json
        all_detections = []
        for raw_json_path in raw_json_files:
            try:
                with open(raw_json_path, "r", encoding="utf-8") as f:
                    json_data = json_module.load(f)
                detection_results = json_data.get("blk_list", [])
                all_detections.extend(detection_results)
            except Exception as e:
                logger.warning(f"⚠️ Failed to read {raw_json_path.name}: {e}")
                continue
        
        # استخراج النصوص
        extracted = []
        for box_data in all_detections:
            xyxy = box_data.get("xyxy", [0, 0, 0, 0])
            bubble_type = box_data.get("bubble_type", "text_bubble")
            lang = box_data.get("language", "unknown")
            
            if bubble_type == "text_bubble":
                dialogue_type = "SPEECH"
            elif bubble_type == "text_free":
                dialogue_type = "OT"
            else:
                dialogue_type = "SPEECH"
            
            extracted.append({
                "xyxy": xyxy,
                "bubble_type": bubble_type,
                "language": lang,
                "dialogue_type": dialogue_type,
            })
        
        results[stem] = extracted
        logger.info(f"📋 Detected {len(extracted)} text regions in {base_name}")
    
    return results


def _classify_and_extract_text(
    image_paths: list[str],
    image_dir: str,
    detection_data: dict[str, list[dict]],
) -> dict[str, str]:
    """
    تصنيف الفقاعات واستخراج النصوص باستخدام CLIP + multilingual OCR.
    
    :param image_paths: مسارات الصور الأصلية
    :param image_dir: مسار المجلد
    :param detection_data: بيانات الكشف من _extract_text_before_cleanup
    :return: قاموس {stem: text_content}
    """
    from PIL import Image as PILImage
    import numpy as np
    import cv2
    
    # تحميل مُصنف الفقاعات (CLIP embeddings)
    try:
        from pcleaner.comic_text_detector.bubble_classifier import BubbleClassifier, BubbleType, SYMBOL_MAP
        classifier = BubbleClassifier()
    except Exception as e:
        logger.warning(f"⚠️ Bubble classifier not available, using defaults: {e}")
        classifier = None
    
    # تحميل محرك OCR المتعدد اللغات (EasyOCR — ja + en فقط)
    # ملاحظة: ch_sim مش متوافقة مع ja/ko في EasyOCR، لازمReader منفصل
    try:
        from pcleaner.ocr.ocr_advanced import DynamicEasyOcrEngine
        ocr_engine = DynamicEasyOcrEngine(languages=['ja', 'en'])
        logger.info("🚀 Using EasyOCR (ja + en)")
    except Exception as e:
        logger.warning(f"⚠️ EasyOCR not available: {e}")
        try:
            from pcleaner.ocr.ocr_advanced import DynamicPaddleOcrEngine
            ocr_engine = DynamicPaddleOcrEngine()
            logger.info("🚀 Using PaddleOCR (multilingual)")
        except Exception as e2:
            logger.error(f"❌ No multilingual OCR available: {e2}")
            return {}
    
    text_results = {}
    
    for image_path in image_paths:
        base_name = os.path.basename(image_path)
        stem = os.path.splitext(base_name)[0]
        
        if stem not in detection_data:
            continue
        
        regions = detection_data[stem]
        if not regions:
            text_results[stem] = ""
            continue
        
        # قراءة الصورة
        try:
            img_pil = PILImage.open(image_path)
            img_np = cv2.imread(image_path)
        except Exception as e:
            logger.warning(f"⚠️ Failed to open image {base_name}: {e}")
            continue
        
        lines = []
        
        for region in regions:
            xyxy = region["xyxy"]
            bubble_type = region["bubble_type"]
            dialogue_type = region["dialogue_type"]
            
            x1, y1, x2, y2 = [int(v) for v in xyxy]
            
            # قص الفقاعة من الصورة
            width, height = img_pil.size
            x1 = max(0, min(x1, width))
            y1 = max(0, min(y1, height))
            x2 = max(0, min(x2, width))
            y2 = max(0, min(y2, height))
            
            if x2 <= x1 or y2 <= y1:
                continue
            
            crop_pil = img_pil.crop((x1, y1, x2, y2))
            crop_np = img_np[y1:y2, x1:x2]
            
            # تصنيف الفقاعة
            symbol = ""
            if classifier:
                try:
                    classification = classifier.classify_bubble(
                        crop_pil, bubble_type_hint=bubble_type
                    )
                    symbol = classification.symbol
                except Exception:
                    # fallback حسب bubble_type
                    if bubble_type == "text_free":
                        symbol = SYMBOL_MAP.get(BubbleType.ON_TEXT, "OT:")
                    else:
                        symbol = SYMBOL_MAP.get(BubbleType.SPEECH, '""')
            else:
                if bubble_type == "text_free":
                    symbol = "OT:"
                else:
                    symbol = '""'
            
            # استخراج النص
            try:
                text = ocr_engine(crop_pil)
                text = text.strip() if text else ""
            except Exception as e:
                logger.warning(f"⚠️ OCR failed for {base_name} region: {e}")
                text = ""
            
            if text:
                # التنسيق: Symbol text
                lines.append(f"{symbol}{text}")
        
        # حفظ النص
        txt_path = os.path.join(image_dir, f"{stem}.txt")
        content = "\n".join(lines)
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(content)
        
        text_results[stem] = content
        logger.info(f"📝 Text for {base_name}: {len(lines)} lines")
    
    return text_results


# ─── إنشاء mask للنصوص اللي برا الفقاعات ─────────────────────────────
def _create_text_mask(
    image_paths: list[str],
    image_dir: str,
    detection_data: dict[str, list[dict]],
) -> dict[str, str]:
    """
    إنشاء mask للنصوص على الصورة مباشرة (text_free) والنصوص الجانبية.
    هذا الـ mask يُدمج مع الصور النظيفة في النهاية عشان النصوص دي تت cleans.
    
    :return: قاموس {stem: mask_path} — مسارات ملفات الـ mask
    """
    mask_paths = {}
    try:
        import numpy as np
        import cv2
        
        for image_path in image_paths:
            base_name = os.path.basename(image_path)
            stem = os.path.splitext(base_name)[0]
            
            if stem not in detection_data:
                continue
            
            regions = detection_data[stem]
            
            mask_regions = [r for r in regions if r.get("bubble_type") == "text_free"]
            
            if not mask_regions:
                continue
            
            img = cv2.imread(image_path)
            if img is None:
                continue
            
            h, w = img.shape[:2]
            mask = np.zeros((h, w), dtype=np.uint8)
            
            for region in mask_regions:
                xyxy = region["xyxy"]
                x1, y1, x2, y2 = [int(v) for v in xyxy]
                
                padding = 10
                x1 = max(0, x1 - padding)
                y1 = max(0, y1 - padding)
                x2 = min(w, x2 + padding)
                y2 = min(h, y2 + padding)
                
                if x2 > x1 and y2 > y1:
                    mask[y1:y2, x1:x2] = 255
            
            mask_path = os.path.join(image_dir, f"{stem}_text_mask.png")
            cv2.imwrite(mask_path, mask)
            mask_paths[stem] = mask_path
            logger.debug(f"🎭 Text mask saved: {mask_path} ({len(mask_regions)} regions)")
        
    except Exception as e:
        logger.warning(f"⚠️ Failed to create text masks: {e}")
    
    return mask_paths


def _apply_text_masks(output_paths: list[str], mask_paths: dict[str, str]):
    """
    دمج الـ text masks مع الصور النظيفة.
    يُطبّق أبيض على المناطق المحددة في الـ mask عشان يشيل النص من على الصورة.
    """
    import numpy as np
    import cv2
    
    for output_path in output_paths:
        base_name = os.path.basename(output_path)
        stem = os.path.splitext(base_name)[0]
        
        if stem not in mask_paths:
            continue
        
        mask_path = mask_paths[stem]
        if not os.path.exists(mask_path):
            continue
        
        img = cv2.imread(output_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        if img is None or mask is None:
            continue
        
        if img.shape[:2] != mask.shape[:2]:
            logger.warning(f"⚠️ Size mismatch for {base_name}: img={img.shape[:2]} mask={mask.shape[:2]}")
            continue
        
        img[mask > 0] = [255, 255, 255]
        cv2.imwrite(output_path, img)
        logger.info(f"🎭 Applied text mask to {base_name}")


class TaskWorker:
    """Main worker for processing image cleaning requests"""
    
    def __init__(self):
        self.processor = ImageProcessor()
    
    # ─── Retry helper ───────────────────────────────────────────────────────────
    async def _retry_async(self, fn, max_retries=3, base_delay=2.0, label="operation"):
        """Retry an async call with exponential backoff."""
        for attempt in range(1, max_retries + 1):
            try:
                return await fn()
            except Exception as e:
                if attempt == max_retries:
                    raise
                delay = base_delay * (2 ** (attempt - 1))
                logger.warning(f"⚠️ {label} failed (attempt {attempt}/{max_retries}): {e}. Retrying in {delay:.0f}s...")
                await asyncio.sleep(delay)
    
    # =======================================================================================
    # 🧹 تنظيف الصور باستخدام PanelCleaner CLI
    # - الخطوة 1: تشغيل Detection فقط (توليد raw.json)
    # - الخطوة 2: استخراج النصوص + تصنيف الفقاعات + حفظ .txt
    # - الخطوة 3: تشغيل باقي الخطوات (masking + denoising + inpainting)
    # - الخطوة 4: البحث عن الإخراج النهائي
    # =======================================================================================
    async def _clean_images(self, image_paths: list, image_dir: str) -> list:
        output_paths = []
        logger.info(f"🚀 Batch processing {len(image_paths)} images...")

        profiles_dir = getattr(settings, "PROFILES_DIR", "pcleaner/profiles")
        profile_argument = "default"
        try:
            conf_files = glob.glob(os.path.join(profiles_dir, "*.conf"))
            if conf_files:
                profile_argument = conf_files[0]
                logger.info(f"⚙️ Profile: {profile_argument}")
        except Exception as e:
            logger.warning(f"⚠️ Profile scan failed: {e}")

        # ─── الخطوة 1: Detection فقط ──────────────────
        logger.info("🔍 Step 1/3: Running text detection...")
        cmd_step1 = ["pcleaner", "clean"] + image_paths + [
            f"--output_dir={image_dir}",
            f"--profile={profile_argument}",
            "--skip-pre-processing",
            "--skip-masking",
            "--skip-denoising",
            "--skip-inpainting",
            "--keep-cache",
        ]

        subprocess_env = os.environ.copy()
        subprocess_env["PYTHONUNBUFFERED"] = "1"
        subprocess_env["OPENBLAS_NUM_THREADS"] = "1"
        subprocess_env["OMP_NUM_THREADS"] = "1"
        subprocess_env["MKL_NUM_THREADS"] = "1"
        subprocess_env["NUMEXPR_NUM_THREADS"] = "1"

        import re
        _skip_prefixes = ("DEBUG", "INFO", "| INFO", "| DEBUG", "Loading weights:", "[transformers]",
                          "Key ", "---", "Notes:", "Warning: You are sending", "Saving image")
        _skip_contains = ("n_points_scale", "UNEXPECTED", "resource_tracker")
        _loguru_re = re.compile(r'^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3} \|')

        async def _stream(stream, log_fn):
            try:
                buf = ""
                while True:
                    chunk = await stream.read(1024)
                    if not chunk:
                        break
                    buf += chunk.decode(errors="ignore")
                    while "\n" in buf:
                        line, buf = buf.split("\n", 1)
                        if "\r" in line:
                            line = line.split("\r")[-1]
                        line = line.strip()
                        if line and not any(line.startswith(p) for p in _skip_prefixes) \
                                and not any(s in line for s in _skip_contains) \
                                and not _loguru_re.match(line):
                            log_fn(line)
                    if "\r" in buf:
                        parts = buf.split("\r")
                        for p in parts[:-1]:
                            p = p.strip()
                            if p and not any(p.startswith(prefix) for prefix in _skip_prefixes) \
                                    and not any(s in p for s in _skip_contains) \
                                    and not _loguru_re.match(p):
                                log_fn(p)
                        buf = parts[-1]
                if buf.strip():
                    remaining = buf.strip()
                    if not any(remaining.startswith(p) for p in _skip_prefixes) \
                            and not any(s in remaining for s in _skip_contains) \
                            and not _loguru_re.match(remaining):
                        log_fn(remaining)
            except Exception:
                pass

        process1 = await asyncio.create_subprocess_exec(
            *cmd_step1,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=subprocess_env,
        )

        out_task = asyncio.ensure_future(_stream(process1.stdout, logger.info))
        err_task = asyncio.ensure_future(_stream(process1.stderr, logger.warning))
        wait_task = asyncio.ensure_future(process1.wait())
        done, pending = await asyncio.wait(
            [out_task, err_task, wait_task],
            return_when=asyncio.ALL_COMPLETED,
            timeout=600.0,
        )
        if wait_task not in done:
            logger.warning("⚠️ Step 1 timed out — killing")
            process1.kill()
            for t in pending:
                t.cancel()
            raise RuntimeError("pcleaner step 1 timed out")
        
        exit_code = wait_task.result()
        if exit_code != 0:
            raise RuntimeError(f"pcleaner step 1 failed (exit {exit_code})")
        
        logger.info("✅ Step 1 complete: detection + pre-processing done")

        # Debug: list cache dir contents
        cache_dir = _get_pcleaner_cache_dir()
        if cache_dir and cache_dir.exists():
            all_files = list(cache_dir.glob("*"))
            json_files = [f for f in all_files if f.suffix == ".json"]
            raw_jsons = [f for f in json_files if "#raw.json" in f.name]
            clean_jsons = [f for f in json_files if "#clean.json" in f.name]
            logger.info(f"📂 Cache dir: {cache_dir}")
            logger.info(f"📂 Total files: {len(all_files)}, JSON: {len(json_files)}, raw.json: {len(raw_jsons)}, clean.json: {len(clean_jsons)}")
            if raw_jsons:
                for rj in raw_jsons[:5]:
                    logger.info(f"  📄 {rj.name}")
            else:
                logger.warning("⚠️ No #raw.json files found in cache dir!")
                if json_files:
                    logger.warning(f"   Available JSON: {[f.name for f in json_files[:10]]}")

        # ─── الخطوة 2: استخراج النصوص + تصنيف الفقاعات + حفظ .txt ──────
        logger.info("📝 Step 2/3: Extracting text + classifying bubbles...")
        
        mask_paths = {}
        try:
            # قراءة بيانات الكشف من raw.json
            detection_data = _extract_text_before_cleanup(image_paths, image_dir)
            
            # استخراج النصوص وتصنيف الفقاعات وحفظ .txt
            text_results = _classify_and_extract_text(image_paths, image_dir, detection_data)
            
            # إنشاء mask للنصوص على الصورة مباشرة
            mask_paths = _create_text_mask(image_paths, image_dir, detection_data)
            
        except Exception as text_err:
            logger.warning(f"⚠️ Text extraction failed (non-critical): {text_err}")

        # ─── الخطوة 3: Pre-processing + تنظيف الصور (masking + denoising + inpainting) ──
        logger.info("🧹 Step 3/3: Running pre-processing + masking + denoising + inpainting...")
        cmd_step3 = ["pcleaner", "clean"] + image_paths + [
            f"--output_dir={image_dir}",
            f"--profile={profile_argument}",
            "--save-only-cleaned",
            "--skip-text-detection",
            "--keep-cache",
        ]

        process3 = await asyncio.create_subprocess_exec(
            *cmd_step3,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=subprocess_env,
        )

        out_task3 = asyncio.ensure_future(_stream(process3.stdout, logger.info))
        err_task3 = asyncio.ensure_future(_stream(process3.stderr, logger.warning))
        wait_task3 = asyncio.ensure_future(process3.wait())
        done3, pending3 = await asyncio.wait(
            [out_task3, err_task3, wait_task3],
            return_when=asyncio.ALL_COMPLETED,
            timeout=600.0,
        )
        if wait_task3 not in done3:
            logger.warning("⚠️ Step 3 timed out — killing")
            process3.kill()
            for t in pending3:
                t.cancel()
            raise RuntimeError("pcleaner step 3 timed out")
        
        exit_code3 = wait_task3.result()
        if exit_code3 != 0:
            raise RuntimeError(f"pcleaner step 3 failed (exit {exit_code3})")
        
        logger.info("✅ Step 3 complete: images cleaned")

        # ─── البحث عن ملفات الإخراج ──────────────────────────────────
        for image_path in image_paths:
            base_name = os.path.basename(image_path)
            stem = os.path.splitext(base_name)[0]

            all_outputs = glob.glob(os.path.join(image_dir, "**", f"*{stem}*.*"), recursive=True)
            original_basenames = {os.path.basename(p) for p in image_paths}
            all_outputs = [
                f for f in all_outputs
                if os.path.isfile(f) and "_raw_" not in f and "#" not in f
                and os.path.basename(f) not in original_basenames
            ]

            final_output = None
            for suffix in ["_clean_inpainted", "_clean_denoised", "_clean", ""]:
                for f in all_outputs:
                    fname = os.path.basename(f)
                    base = os.path.splitext(fname)[0]
                    if suffix:
                        if base == f"{stem}{suffix}":
                            final_output = f
                            break
                    else:
                        if base == stem:
                            final_output = f
                            break
                if final_output:
                    break

            if not final_output:
                direct_output = os.path.join(image_dir, base_name)
                if os.path.exists(direct_output) and os.path.abspath(direct_output) != os.path.abspath(image_path):
                    final_output = direct_output

            if not final_output or not os.path.exists(final_output):
                raise RuntimeError(f"No cleaned output found for {base_name} after processing")

            output_ext = os.path.splitext(final_output)[1]
            base_webp = os.path.splitext(base_name)[0] + output_ext
            new_fixed_path = os.path.join(image_dir, base_webp)
            try:
                if os.path.abspath(final_output) != os.path.abspath(new_fixed_path):
                    if os.path.exists(new_fixed_path):
                        os.remove(new_fixed_path)
                    shutil.move(final_output, new_fixed_path)
                elif "_clean" in os.path.basename(final_output):
                    os.rename(final_output, new_fixed_path)
                final_output = new_fixed_path
            except Exception as move_err:
                raise RuntimeError(f"Failed to normalize output for {base_name}: {move_err}")

            output_paths.append(final_output)

        logger.info(f"✅ All {len(output_paths)} images cleaned successfully")

        # ─── الخطوة 4: دمج الـ text masks مع الصور النظيفة ──────
        if mask_paths:
            logger.info(f"🎭 Applying {len(mask_paths)} text masks to cleaned images...")
            try:
                _apply_text_masks(output_paths, mask_paths)
            except Exception as mask_err:
                logger.warning(f"⚠️ Failed to apply text masks (non-critical): {mask_err}")

        return output_paths
            
    async def process_request(self, request_id: int, user_id: str, folder_url: str):
        """Main processing function for a request"""
        try:
            logger.info(f"🔄 Processing request {request_id} for user {user_id}")
            
            db.update_request_status(request_id, "processing")
            
            drive = get_rclone()
            
            folder_id = drive.get_folder_id_from_url(folder_url)
            if not folder_id:
                raise ValueError("Invalid Google Drive folder URL")
            
            temp_chapter_dir = self.processor.create_temp_chapter_dir()
            
            # تحميل الصور من Google Drive
            logger.info(f"⬇️ Downloading images from Google Drive...")
            download_ok = await self._retry_async(
                lambda: asyncio.to_thread(drive.download_folder, folder_id, temp_chapter_dir),
                max_retries=3, base_delay=3.0,
                label="Google Drive download"
            )
            if not download_ok:
                raise ValueError("Failed to download images from Google Drive")
            
            # فلترة الصور
            image_paths = [
                os.path.join(temp_chapter_dir, f)
                for f in os.listdir(temp_chapter_dir)
                if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp'))
                and os.path.isfile(os.path.join(temp_chapter_dir, f))
            ]
            
            if not image_paths:
                raise ValueError("No images found in the downloaded folder")
            
            pixel_count = self.processor.calculate_pixel_count(image_paths)
            credits_needed = self.processor.calculate_credits_needed(pixel_count)
            
            if credits_needed == -1:
                raise ValueError(
                    "This chapter exceeds the 500k pixel limit and cannot be processed."
                )
            
            user_credits = db.get_user_credits(user_id)
            if user_credits < credits_needed:
                raise ValueError(f"Insufficient credits. Need {credits_needed}, but user has {user_credits}")
            
            output_paths = await self._clean_images(image_paths, temp_chapter_dir)
            
            if not output_paths:
                raise ValueError("Failed to clean images")
            
            master_folder_id = settings.GOOGLE_OUTPUT_FOLDER_ID
            logger.info(f"📂 Master folder: {master_folder_id}")

            try:
                new_folder_name = f"Cleaned_Order_{request_id}"
                logger.info(f"📁 Creating subfolder: {new_folder_name}...")

                async def _create_folder():
                    folder_metadata = {
                        'name': new_folder_name,
                        'mimeType': 'application/vnd.google-apps.folder',
                        'parents': [master_folder_id]
                    }
                    created_folder = gdrive.service.files().create(
                        body=folder_metadata, 
                        fields='id'
                    ).execute()
                    return created_folder.get('id')

                sub_folder_id = await self._retry_async(
                    _create_folder, max_retries=5, base_delay=3.0,
                    label=f"Google Drive folder '{new_folder_name}'"
                )
                logger.info(f"✅ Created subfolder: {sub_folder_id}")

                try:
                    user_permission = {
                        'type': 'anyone',
                        'role': 'reader'
                    }
                    gdrive.service.permissions().create(
                        fileId=sub_folder_id,
                        body=user_permission,
                        fields='id'
                    ).execute()
                    logger.info(f"🔓 Made folder public")
                except Exception as perm_err:
                    logger.warning(f"⚠️ Failed to make folder public: {perm_err}")

            except Exception as folder_err:
                logger.error(f"❌ Failed to create subfolder: {folder_err}")
                raise folder_err
            
            # 🔄 رفع الصور المبيضة + ملفات النص باستخدام FastDriveManager
            logger.info(f"⬆️ Uploading {len(output_paths)} cleaned images + text files...")
            
            # نسخ الملفات المطلوبة فقط في مجلد مؤقت للرفع
            upload_dir = os.path.join(temp_chapter_dir, "_upload")
            os.makedirs(upload_dir, exist_ok=True)
            
            for output_path in output_paths:
                if os.path.exists(output_path):
                    shutil.copy2(output_path, upload_dir)
            
            txt_files = glob.glob(os.path.join(temp_chapter_dir, "*.txt"))
            for txt_file in txt_files:
                shutil.copy2(txt_file, upload_dir)
            
            # رفع المجلد المفلّتر
            upload_ok = await self._retry_async(
                lambda: asyncio.to_thread(drive.upload_folder, upload_dir, sub_folder_id),
                max_retries=3, base_delay=3.0,
                label="Google Drive upload"
            )
            
            uploaded_count = len(output_paths) + len(txt_files)
            shutil.rmtree(upload_dir, ignore_errors=True)

            if not upload_ok or uploaded_count == 0:
                raise ValueError("Failed to upload images to Google Drive")
            
            output_url_result = f"https://drive.google.com/drive/folders/{sub_folder_id}"
            logger.info(f"🎯 Final URL: {output_url_result}")
            
            db.deduct_credits(user_id, credits_needed)
            
            try:
                db.add_total_credits_used(user_id, credits_needed)
                logger.info(f"📊 Added {credits_needed} to total_credits_used for user {user_id}")
            except Exception as total_err:
                logger.warning(f"⚠️ Failed to update total_credits_used: {total_err}")
            
            db.update_request_status(
                request_id,
                "completed",
                pixel_count=pixel_count,
                output_url=output_url_result,
                credits_deducted=credits_needed
            )
            
            self.processor.cleanup_temp_dir(temp_chapter_dir)
            
            logger.info(f"✅ Request {request_id} completed successfully")
            
        except Exception as e:
            logger.error(f"❌ Error processing request {request_id}: {e}")
            db.update_request_status(request_id, "failed", error_message=f"[Request #{request_id}] {e}")
            temp_dir = locals().get("temp_chapter_dir")
            if temp_dir:
                self.processor.cleanup_temp_dir(temp_dir)
            sub_folder_id = locals().get("sub_folder_id")
            if sub_folder_id:
                try:
                    drive = get_rclone()
                    drive.delete_folder(sub_folder_id)
                    logger.info(f"🗑️ Deleted empty Drive folder {sub_folder_id}")
                except Exception as del_err:
                    logger.warning(f"⚠️ Could not delete Drive folder {sub_folder_id}: {del_err}")
    
    async def process_queue(self):
        """Process queued requests by polling the database sequentially"""
        logger.info("📥 Worker started: polling database for pending requests...")
        while True:
            try:
                request_data = db.get_next_pending_request()

                if request_data:
                    request_id = request_data["id"]
                    logger.info(f"🚀 Processing request #{request_id} sequentially...")
                    await self.process_request(
                        request_id,
                        request_data["user_id"],
                        request_data["folder_url"],
                    )
                    logger.info(f"✅ Request #{request_id} fully completed")
                else:
                    await asyncio.sleep(5)

            except Exception as e:
                logger.error(f"❌ Queue processing error: {e}")
                await asyncio.sleep(5)


# Global worker instance
worker = TaskWorker()

# Pre-load models at import time (avoids download delay on first user request).
preload_text_detector()
preload_opencv_detector()
preload_inpainting_model()
