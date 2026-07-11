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
from loguru import logger


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
    # - يشغل pcleaner كـ subprocess حقيقي
    # - ينتظر انتهاء المعالجة بالكامل
    # - يبحث عن ملفات الإخراج الفعلية (_clean / _clean_denoised / _clean_inpaint)
    # - يعيد قائمة بمسارات الصور المنظفة الجاهزة للرفع
    # =======================================================================================
    async def _clean_images(self, image_paths: list, image_dir: str) -> list:
        output_paths = []
        logger.info(f"🚀 Batch processing {len(image_paths)} images in a single pcleaner call...")

        profiles_dir = getattr(settings, "PROFILES_DIR", "pcleaner/profiles")
        profile_argument = "default"
        try:
            conf_files = glob.glob(os.path.join(profiles_dir, "*.conf"))
            if conf_files:
                profile_argument = conf_files[0]
                logger.info(f"⚙️ Profile: {profile_argument}")
        except Exception as e:
            logger.warning(f"⚠️ Profile scan failed: {e}")

        # Single batch: pcleaner clean img1 img2 ... --output_dir=... --profile=... --save-only-cleaned
        cmd = ["pcleaner", "clean"] + image_paths + [
            f"--output_dir={image_dir}",
            f"--profile={profile_argument}",
            "--save-only-cleaned",
        ]

        # Force unbuffered Python output in subprocess so progress is captured
        subprocess_env = os.environ.copy()
        subprocess_env["PYTHONUNBUFFERED"] = "1"
        # Limit BLAS threads per subprocess-worker so multiprocessing pools
        # don't exhaust the system with N_workers × 64 threads.
        subprocess_env["OPENBLAS_NUM_THREADS"] = "1"
        subprocess_env["OMP_NUM_THREADS"] = "1"
        subprocess_env["MKL_NUM_THREADS"] = "1"
        subprocess_env["NUMEXPR_NUM_THREADS"] = "1"

        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=subprocess_env,
        )

        # Stream stdout/stderr in real-time so user sees tqdm progress bars
        # Filter: skip loguru DEBUG/INFO lines and model loading noise, keep tqdm bars and errors
        _skip_prefixes = ("DEBUG", "INFO", "| INFO", "| DEBUG", "Loading weights:", "[transformers]",
                          "Key ", "---", "Notes:", "Warning: You are sending", "Saving image")
        _skip_contains = ("n_points_scale", "UNEXPECTED", "resource_tracker")
        import re
        # Loguru format: "2026-06-20 18:02:50.024 | INFO     | module:func:line - msg"
        _loguru_re = re.compile(r'^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3} \|')
        async def _stream(stream, log_fn):
            try:
                buf = ""
                while True:
                    chunk = await stream.read(1024)
                    if not chunk:
                        break
                    buf += chunk.decode(errors="ignore")
                    # Process \n-terminated lines
                    while "\n" in buf:
                        line, buf = buf.split("\n", 1)
                        if "\r" in line:
                            line = line.split("\r")[-1]
                        line = line.strip()
                        if line and not any(line.startswith(p) for p in _skip_prefixes) \
                                and not any(s in line for s in _skip_contains) \
                                and not _loguru_re.match(line):
                            log_fn(line)
                    # Process \r-updates (tqdm overwrites same line)
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

        async def _wait_process():
            return await process.wait()

        out_task = asyncio.ensure_future(_stream(process.stdout, logger.info))
        err_task = asyncio.ensure_future(_stream(process.stderr, logger.warning))
        wait_task = asyncio.ensure_future(_wait_process())

        done, pending = await asyncio.wait(
            [out_task, err_task, wait_task],
            return_when=asyncio.ALL_COMPLETED,
            timeout=600.0,
        )

        # If the process hasn't finished yet, kill it
        if wait_task not in done:
            logger.warning("⚠️ pcleaner subprocess timed out — killing")
            process.kill()
            for t in pending:
                t.cancel()
            raise RuntimeError(f"pcleaner batch timed out (timeout={timeout}s)")

        # Check exit code
        exit_code = wait_task.result()

        if exit_code != 0:
            raise RuntimeError(f"pcleaner batch failed (exit {exit_code})")

        if process.returncode != 0:
            raise RuntimeError(f"pcleaner batch failed (exit {process.returncode})")

        for image_path in image_paths:
            base_name = os.path.basename(image_path)
            stem = os.path.splitext(base_name)[0]

            # Search for any output file matching this stem (any extension, any suffix)
            all_outputs = glob.glob(os.path.join(image_dir, "**", f"*{stem}*.*"), recursive=True)
            # Filter: only actual files, exclude raw/# files and the original input files
            original_basenames = {os.path.basename(p) for p in image_paths}
            all_outputs = [
                f for f in all_outputs
                if os.path.isfile(f) and "_raw_" not in f and "#" not in f
                and os.path.basename(f) not in original_basenames
            ]

            final_output = None
            # Prefer _clean_inpainted (highest quality), then _clean_denoised, then _clean, then plain stem
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
                raise RuntimeError(f"No cleaned output found for {base_name} after batch processing")

            # Normalize to a flat path in image_dir, preserving the output extension
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
        return output_paths
            
    async def process_request(self, request_id: int, user_id: str, folder_url: str):
        """Main processing function for a request"""
        try:
            logger.info(f"🔄 Processing request {request_id} for user {user_id}")
            
            # Update status to processing
            db.update_request_status(request_id, "processing")
            
            # Extract folder ID from URL
            folder_id = gdrive.get_folder_id_from_url(folder_url)
            if not folder_id:
                raise ValueError("Invalid Google Drive folder URL")
            
            # Create temporary directory
            temp_chapter_dir = self.processor.create_temp_chapter_dir()
            
            # Download images from Google Drive
            files = gdrive.list_files_in_folder(folder_id)
            if not files:
                raise ValueError("No images found in the folder")
            
            image_paths = []
            for file in files:
                if file['name'].lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp')):
                    local_path = os.path.join(temp_chapter_dir, file['name'])
                    if gdrive.download_image(file['id'], local_path):
                        image_paths.append(local_path)
            
            if not image_paths:
                raise ValueError("Failed to download any images")
            
            # Calculate pixel count (raises ValueError if >16k height)
            pixel_count = self.processor.calculate_pixel_count(image_paths)
            credits_needed = self.processor.calculate_credits_needed(pixel_count)
            
            if credits_needed == -1:
                raise ValueError(
                    "This chapter exceeds the 500k pixel limit and cannot be processed."
                )
            
            # Check user credits
            user_credits = db.get_user_credits(user_id)
            if user_credits < credits_needed:
                raise ValueError(f"Insufficient credits. Need {credits_needed}, but user has {user_credits}")
            
            # Process images with PanelCleaner الفعلي
            output_paths = await self._clean_images(
                image_paths,
                temp_chapter_dir
            )
            
            if not output_paths:
                raise ValueError("Failed to clean images")
            
            # Guard: ensure Google Drive service is initialized before using it
            if gdrive.service is None:
                raise RuntimeError("Google Drive service is not initialized. Check OAuth credentials.")

            # 📁 1. المجلد الرئيسي الثابت (مجلدك الشخصي الكبير)
            master_folder_id = settings.GOOGLE_OUTPUT_FOLDER_ID
            logger.info(f"📂 المجلد الرئيسي للسيرفر هو: {master_folder_id}")

            # 🆕 2. إنشاء مجلد فرعي مستقل تماماً للطلب الحالي مع إعادة المحاولة
            try:
                new_folder_name = f"Cleaned_Order_{request_id}"
                logger.info(f"📁 جاري إنشاء مجلد فرعي معزول للمستخدم باسم: {new_folder_name}...")

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
                logger.info(f"✅ تم إنشاء المجلد الفرعي بنجاح! الـ ID: {sub_folder_id}")

                # 🔓 جعل الفولدر عام
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
                    logger.info(f"🔓 تم جعل رابط المجلد عاماً للعميل بنجاح.")
                except Exception as perm_err:
                    logger.warning(f"⚠️ فشل جعل المجلد عاماً: {perm_err}")

            except Exception as folder_err:
                logger.error(f"❌ فشل إنشاء المجلد الفرعي: {folder_err}")
                raise folder_err
            
# 🔄 3. رفع الصور المبيضة مع إعادة المحاولة لكل صورة
            uploaded_count = 0
            for output_path in output_paths:
                if not os.path.exists(output_path):
                    continue

                file_name = os.path.basename(output_path)
                try:
                    async def _upload_single(path=output_path, folder=sub_folder_id):
                        return gdrive.upload_file(file_path=path, folder_id=folder)

                    file_id = await self._retry_async(
                        _upload_single, max_retries=3, base_delay=2.0,
                        label=f"upload '{file_name}'"
                    )
                    if file_id:
                        uploaded_count += 1
                except Exception as upload_err:
                    logger.error(f"❌ فشل رفع الصورة {file_name} بعد 3 محاولات: {upload_err}")
            
            if uploaded_count == 0:
                raise ValueError("Failed to upload images to Google Drive")
            
            # 🔗 4. رابط المجلد الفرعي اللي هيروح للمستخدم في البوت
            output_url_result = f"https://drive.google.com/drive/folders/{sub_folder_id}"
            logger.info(f"🎯 الرابط النهائي المرسل للمستخدم: {output_url_result}")
            
            # Deduct credits
            # 🎯 الخصم الفعلي من الرصيد الحالي
            db.deduct_credits(user_id, credits_needed)
            
            # 🆕 إضافة القيمة المستهلكة إلى إجمالي النقاط المستخدمة (Total Credits Used)
            try:
                db.add_total_credits_used(user_id, credits_needed)
                logger.info(f"📊 Added {credits_needed} to total_credits_used for user {user_id}")
            except Exception as total_err:
                logger.warning(f"⚠️ Failed to update total_credits_used: {total_err}")
            
            # Update request with results
           # 📝 تحديث حالة الطلب وحفظ رابط الفولدر الفرعي الجديد مباشرة
            db.update_request_status(
                request_id,
                "completed",
                pixel_count=pixel_count,
                output_url=output_url_result,
                credits_deducted=credits_needed
            )
            
            # Cleanup temporary directory
            self.processor.cleanup_temp_dir(temp_chapter_dir)
            
            logger.info(f"✅ Request {request_id} completed successfully")
            
        except Exception as e:
            logger.error(f"❌ Error processing request {request_id}: {e}")
            db.update_request_status(request_id, "failed", error_message=f"[Request #{request_id}] {e}")
            temp_dir = locals().get("temp_chapter_dir")
            if temp_dir:
                self.processor.cleanup_temp_dir(temp_dir)
            sub_folder_id = locals().get("sub_folder_id")
            if sub_folder_id and gdrive.service:
                try:
                    gdrive.service.files().delete(fileId=sub_folder_id).execute()
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