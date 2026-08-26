"""
Background worker for processing image cleaning tasks
Uses background tasks queue system
"""
import os
import re
import shutil
import asyncio
import glob
import time
from pathlib import Path
from uuid import uuid4
from app.server_config import settings
from app.services.database import db
from app.services.google_drive import gdrive
from app.services.rclone_manager import get_rclone, FastDriveManager
from app.services.clean_pipeline import CleanPipeline
from loguru import logger


def _natural_sort_key(p: Path | str) -> list:
    s = p.name if isinstance(p, Path) else str(p)
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


def _natural_sort(paths: list[Path]) -> list[Path]:
    return sorted(paths, key=_natural_sort_key)


class ImageProcessor:
    """Handle image processing tasks"""
    
    def __init__(self):
        self.temp_dir = Path(settings.TEMP_CHAPTERS_DIR)
        self.temp_dir.mkdir(parents=True, exist_ok=True)
    
    def calculate_pixel_count(self, image_paths: list) -> int:
        """Calculate total height from images."""
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
            return -1
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
        """Delete temporary chapter directory. With per-chapter _work, a single rmtree cleans everything (like Upscale bot)."""
        try:
            if os.path.exists(directory):
                shutil.rmtree(directory)
                logger.info(f"✅ Cleaned up temporary directory (incl. _work/_upload): {directory}")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to cleanup temporary directory: {e}")
            return False

    def cleanup_cleaner_cache_for_stems(self, stems: list[str]) -> None:
        """Legacy no-op: per-chapter _work is auto-cleaned with rmtree, no global cache to purge."""
        return


def preload_text_detector():
    """Pre-download the text detector model at server startup."""
    try:
        import os
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        from app.services import model_downloader as md

        detector_choice = "bubble"
        try:
            from app.config import Profile
            profiles_dir = getattr(settings, "PROFILES_DIR", "app/profiles")
            conf_files = glob.glob(os.path.join(profiles_dir, "*.conf"))
            if conf_files:
                profile = Profile.load(Path(conf_files[0]))
                detector_choice = getattr(profile.text_detector, "text_detector_model", "bubble")
        except Exception as profile_err:
            logger.debug(f"Could not read profile for detector selection: {profile_err}")

        # Normalize like preprocessor / config
        _alias = {"bubble":"bubble","hi-sam":"hi-sam","hisam":"hi-sam","hi_sam":"hi-sam","ctd":"ctd","comic_text_detector":"ctd","comictextdetector":"ctd","yolo":"yolo11s","yolo11s":"yolo11s"}
        detector_norm = _alias.get(str(detector_choice).strip().lower(), str(detector_choice).strip().lower())
        if detector_norm == "hi-sam":
            model_type = "vit_b"
            try:
                from app.config import Profile
                profiles_dir = getattr(settings, "PROFILES_DIR", "app/profiles")
                conf_files = glob.glob(os.path.join(profiles_dir, "*.conf"))
                if conf_files:
                    profile = Profile.load(Path(conf_files[0]))
                    model_type = getattr(profile.text_detector, "hi_sam_model_type", "vit_h")
            except Exception:
                pass
            ckpt_dir = md.ensure_hi_sam_available(model_type)
            logger.info(f"Hi-SAM text detector ready at {ckpt_dir}")
        elif detector_norm == "ctd":
            ctd_dir = md.ensure_ctd_available()
            logger.info(f"CTD text detector ready at {ctd_dir}")
        elif detector_norm == "yolo11s":
            yolo_dir = md.ensure_yolo11s_available()
            logger.info(f"YOLO11s text detector ready at {yolo_dir}")
        else:
            model_dir = md.get_text_detector_model_dir()
            md.ensure_bubble_detector_available(model_dir)
            logger.info(f"BubbleAware text detector ready at {model_dir} (fallback for {detector_choice})")
    except Exception as e:
        logger.warning(f"Could not pre-load text detector: {e}")


def preload_inpainting_model():
    """Pre-download + eager-load the LaMa inpainting model at server startup (prevents 4x lazy download)."""
    try:
        from app.config import Profile
        from app.services import model_downloader as md
        from app.services.clean_pipeline import _ModelCache
        import app.config as cfg

        profiles_dir = getattr(settings, "PROFILES_DIR", "app/profiles")
        conf_files = glob.glob(os.path.join(profiles_dir, "*.conf"))
        if conf_files:
            profile = Profile.load(Path(conf_files[0]))
            # Build a Config wrapper so get_inpainting_model_path works
            tmp_cfg = cfg.load_config()
            tmp_cfg.current_profile = profile
            md.ensure_inpainting_available(tmp_cfg.current_profile)
            # Eager load weights into memory (single download, no race)
            _ModelCache.get_inpainter(tmp_cfg)
            logger.info("✅ LaMa inpainting model file ready on disk (eager loaded, no lazy 4x download)")
        else:
            tmp_cfg = cfg.load_config()
            md.ensure_inpainting_available(tmp_cfg.current_profile)
            _ModelCache.get_inpainter(tmp_cfg)
            logger.info("✅ LaMa inpainting model file ready on disk")
    except Exception as e:
        logger.warning(f"⚠️ Could not pre-download inpainting model: {e}")


def preload_opencv_detector():
    """Legacy alias — pre-downloads the text detector at server startup."""
    preload_text_detector()


class TaskWorker:
    """Main worker for processing image cleaning requests"""
    
    def __init__(self):
        self.processor = ImageProcessor()
    
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
    
    async def _clean_images(self, image_paths: list, image_dir: str) -> list:
        """Pre-compress → Split → Detect/Clean parts → Merge (Upscale-bot style).

        Organised layout inside ``temp_chapters/<uuid>/``:
          - flat input (pre-compressed .jpg)
          - parts/  (+ parts/_work intermediates)
          - output/ (final merged images → 001.jpg, 002.jpg …)
        Single ``rmtree`` on ``image_dir`` cleans everything.
        """
        temp_chapter_path = Path(image_dir).resolve()
        parts_dir = temp_chapter_path / "parts"
        output_dir = temp_chapter_path / "output"
        parts_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)

        pipeline = CleanPipeline()
        logger.info(f"🚀 Batch processing {len(image_paths)} images (pre-compress→split→clean→merge) ...")
        logger.info(f"📁 Layout: input={temp_chapter_path} | parts={parts_dir} | output={output_dir}")

        # ── Split stage ──────────────────────────────────────────────────
        # pre_compress is expected to have run in process_request already.
        # If parts_dir is empty, we (re)run split. Otherwise reuse existing parts
        # (idempotent retry support).
        existing_parts = [p for p in parts_dir.iterdir() if p.is_file()] if parts_dir.exists() else []
        if existing_parts:
            logger.info(f"♻️ Reusing existing {len(existing_parts)} part(s) in {parts_dir}")
            all_parts = _natural_sort([p for p in existing_parts if p.suffix.lower() in (".jpg",".jpeg",".png",".bmp",".webp",".tiff",".tif")])
        else:
            logger.info("✂️ Splitting pre-compressed images into parts ...")
            all_parts = await asyncio.to_thread(pipeline.split_all_images, Path(image_dir), parts_dir)
            if not all_parts:
                raise ValueError("No images found to process after split")

        all_parts_str = [str(p) for p in all_parts]
        original_images = [Path(p) for p in image_paths]

        # ── Detection + Cleaning on parts ────────────────────────────────
        logger.info(f"🔍 Step 1/2: Running text detection on {len(all_parts)} part(s) ...")
        await asyncio.to_thread(pipeline.run_detection, all_parts_str, str(parts_dir))

        logger.info("🧹 Step 2/2: Running pre-processing + masking + denoising + inpainting on parts ...")
        await asyncio.to_thread(pipeline.run_cleaning, all_parts_str, str(parts_dir))

        # ── Collect cleaned parts from parts_dir (exclude _work) ─────────
        # Preferred output type determines which files are the real cleaned outputs.
        # e.g. if preferred_file_type=.png and parts are .jpg, cleaned are .png
        preferred = pipeline._pcleaner_config.current_profile.general.preferred_file_type
        if preferred is None:
            # Original type preserved → cleaned suffix == part suffix
            preferred_lower = None
        else:
            if not preferred.startswith("."):
                preferred = f".{preferred}"
            preferred_lower = preferred.lower()

        candidates: list[Path] = []
        for p in parts_dir.iterdir():
            if p.is_dir():
                continue
            if p.suffix.lower() not in (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif"):
                continue
            candidates.append(p)

        if preferred_lower:
            cleaned_parts = [p for p in candidates if p.suffix.lower() == preferred_lower]
            # Fallback: if nothing matches preferred (e.g. JPEG limit forced PNG), collect any image not equal to original part names
            # Also handle case where cleaning overwrote same-suffix files → those are both original and cleaned, so candidates is correct
            if not cleaned_parts:
                logger.warning(f"⚠️ No cleaned parts with preferred ext {preferred_lower}, falling back to all images in parts/")
                cleaned_parts = candidates
        else:
            cleaned_parts = candidates

        # Natural sort for deterministic merge grouping
        cleaned_parts = _natural_sort(cleaned_parts)
        logger.info(f"📦 Collected {len(cleaned_parts)} cleaned part(s) from {parts_dir}")

        if not cleaned_parts:
            raise RuntimeError("No cleaned outputs found in parts_dir after cleaning")

        # ── Merge parts back → output_dir (001.jpg, 002.jpg …) ─────────
        logger.info(f"🧩 Merging {len(cleaned_parts)} part(s) → {output_dir} ...")
        merged_paths: list[Path] = await asyncio.to_thread(
            pipeline.merge_parts_from_disk, cleaned_parts, original_images, output_dir
        )

        if not merged_paths:
            raise RuntimeError("Merge produced no output images")

        # ── Re-discover final outputs from output_dir (sequential 001,002… ) ──
        output_paths: list[str] = []
        for p in _natural_sort([x for x in output_dir.iterdir() if x.is_file() and x.suffix.lower() in (".jpg",".jpeg",".png",".bmp",".webp",".tiff",".tif")]):
            output_paths.append(str(p.resolve()))

        if not output_paths:
            # Fallback: use merged_paths directly
            output_paths = [str(p.resolve()) for p in merged_paths]

        logger.info(f"✅ All {len(output_paths)} images cleaned successfully (merged from {len(cleaned_parts)} parts)")
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
            
            logger.info(f"⬇️ Downloading images from Google Drive...")
            download_ok = await self._retry_async(
                lambda: asyncio.to_thread(drive.download_folder, folder_id, temp_chapter_dir),
                max_retries=3, base_delay=3.0,
                label="Google Drive download"
            )
            if not download_ok:
                raise ValueError("Failed to download images from Google Drive")
            
            # flat listing — explicitly exclude _upload and any sub-directory (old: temp_chapters/<uuid>/_upload)
            image_paths = [
                os.path.join(temp_chapter_dir, f)
                for f in os.listdir(temp_chapter_dir)
                if f != "_upload"
                and not os.path.isdir(os.path.join(temp_chapter_dir, f))
                and f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp'))
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

            # ── Pre-compress stage (Upscale-bot style): temp_chapters/<uuid>/ flat → JPEG q85 max 3000  ──
            # Layout mirrors Upscale bot queue_worker.py:99-234 — single rmtree cleans temp_chapters/<uuid>/ (input flat, parts/, _work, output/)
            logger.info("🗜️ Pre-compress stage: JPEG q=85, max 3000px ...")
            _pre_pipeline = CleanPipeline()
            await asyncio.to_thread(_pre_pipeline.pre_compress_images, Path(temp_chapter_dir), 85, 3000)
            # Re-collect image_paths after compression (extension may have become .jpg, like Upscale bot)
            image_paths = [
                os.path.join(temp_chapter_dir, f)
                for f in os.listdir(temp_chapter_dir)
                if f not in ("_upload", "parts", "output", "_work")
                and not os.path.isdir(os.path.join(temp_chapter_dir, f))
                and f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff', '.tif'))
                and os.path.isfile(os.path.join(temp_chapter_dir, f))
            ]
            image_paths = sorted(image_paths, key=lambda p: _natural_sort_key(Path(p)))
            logger.info(f"📋 After pre-compress: {len(image_paths)} image(s)")
            if not image_paths:
                raise ValueError("No images found after pre-compress")
            
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
            
            logger.info(f"⬆️ Uploading {len(output_paths)} cleaned images...")
            
            # _upload sub-dir inside temp_chapters/<uuid>/_upload  (matches old worker.py:866)
            upload_dir = os.path.join(temp_chapter_dir, "_upload")
            os.makedirs(upload_dir, exist_ok=True)
            
            for output_path in output_paths:
                if os.path.exists(output_path):
                    shutil.copy2(output_path, upload_dir)
            # also copy .txt files generated during text-extraction phase (old system did this)
            txt_files = glob.glob(os.path.join(temp_chapter_dir, "*.txt"))
            for txt_file in txt_files:
                try:
                    shutil.copy2(txt_file, upload_dir)
                except Exception:
                    pass
            
            # upload_folder reads ONLY from _upload (flat), not from temp_chapter_dir directly
            upload_ok = await self._retry_async(
                lambda: asyncio.to_thread(drive.upload_folder, upload_dir, sub_folder_id),
                max_retries=3, base_delay=3.0,
                label="Google Drive upload"
            )
            
            uploaded_count = len(output_paths) + len(txt_files)
            shutil.rmtree(upload_dir, ignore_errors=True)
            # targeted cache purge for this chapter's stems (extra safety before full purge in cleanup_temp_dir)
            try:
                _stems = [os.path.splitext(os.path.basename(p))[0] for p in image_paths]
                self.processor.cleanup_cleaner_cache_for_stems(_stems)
            except Exception:
                pass

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

# Pre-load models at import time
preload_text_detector()
preload_opencv_detector()
preload_inpainting_model()
