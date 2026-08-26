"""
Inline cleaning pipeline — calls pcleaner's individual functions directly
instead of spawning subprocesses.

Models are loaded once (singleton) and reused across requests.
The pipeline mirrors the exact same steps as ``pcleaner clean`` CLI
so output is identical.
"""

import glob
import os
import re
import shutil
import threading
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, as_completed
from multiprocessing import Pool
from pathlib import Path

from loguru import logger
from PIL import Image

import app.config as cfg
import app.services.preprocessor as pp
import app.services.masker as ma
import app.services.denoiser as dn
import app.services.inpainting as ip
import app.services.image_export as ie
import app.services.output_structures as ost
import app.services.structures as st
import app.utils.helpers as hp
import app.services.model_downloader as md
from app.utils.device import cuda_available, mps_available

from app.server_config import settings

Image.MAX_IMAGE_PIXELS = 2**32


def _natural_sort_key(path: Path | str) -> list:
    """Natural sort key: splits digits vs text so e.g. 2 < 10."""
    s = path.name if isinstance(path, Path) else str(path)
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


def natural_sort(paths: list[Path]) -> list[Path]:
    return sorted(paths, key=_natural_sort_key)


# ═══════════════════════════════════════════════════════════════════════════════
# Singleton model cache — loaded once, reused across all pipeline instances
# ═══════════════════════════════════════════════════════════════════════════════

class _ModelCache:
    """Thread-safe singleton that holds heavy model objects across requests."""

    _lock = threading.Lock()
    _inpainter = None
    _inpainter_config_id = None

    @classmethod
    def get_inpainter(cls, config: cfg.Config):
        """Return the cached InpaintingModel, creating it on first call."""
        config_id = hash(str(config.current_profile.inpainter))
        with cls._lock:
            if cls._inpainter is not None and cls._inpainter_config_id == config_id:
                logger.debug("♻️ Reusing cached inpainting model")
                return cls._inpainter
            logger.info("🔧 Loading inpainting model (first call or config changed)…")
            md.ensure_inpainting_available(config)
            cls._inpainter = ip.InpaintingModel(config)
            cls._inpainter_config_id = config_id
            return cls._inpainter

    @classmethod
    def invalidate(cls):
        """Drop cached models (e.g. on config change)."""
        with cls._lock:
            cls._inpainter = None
            cls._inpainter_config_id = None


# ═══════════════════════════════════════════════════════════════════════════════
# Pipeline
# ═══════════════════════════════════════════════════════════════════════════════

class CleanPipeline:
    """Calls pcleaner's individual pipeline functions directly (no subprocess).

    Loads the config **once** at construction and exposes step-specific
    helpers matching the worker's two-phase flow:

    * ``run_detection``  — text detection only  (step 1)
    * ``run_cleaning``   — pre-process + mask + denoise + inpaint (steps 2-5)

    Heavy models (text detector, inpainter) are cached in the
    ``_ModelCache`` singleton so they are loaded only once across the
    lifetime of the server process.
    """

    def __init__(self):
        self._pcleaner_config = cfg.load_config()
        self._profile_path = self._resolve_profile_path()
        ok, err = self._pcleaner_config.load_profile(self._profile_path)
        if not ok:
            logger.warning(f"⚠️ Profile load failed ({err}), using defaults")
        logger.info(f"⚙️ CleanPipeline profile: {self._profile_path}")

    # ------------------------------------------------------------------
    # Profile helper
    # ------------------------------------------------------------------
    @staticmethod
    def _resolve_profile_path() -> str:
        profiles_dir = getattr(settings, "PROFILES_DIR", "app/profiles")
        try:
            conf_files = glob.glob(os.path.join(profiles_dir, "*.conf"))
            if conf_files:
                return conf_files[0]
        except Exception as e:
            logger.warning(f"⚠️ Profile scan failed: {e}")
        return "default"

    # ------------------------------------------------------------------
    # Step 1 — detection only (generates raw.json in cache)
    # ------------------------------------------------------------------
    def run_detection(self, image_paths: list[str], output_dir: str):
        """Run text detection AI model on the given images.

        Produces ``#raw.json`` files in ``<temp_chapter>/_work`` (per-chapter, like Upscale bot).
        Old bug: used global ``app/cache/cleaner``; now every chapter has its own ``_work`` subfolder
        with clear names (no UUID prefix collision).
        """
        logger.info(f"🔍 [inline] Running text detection on {len(image_paths)} image(s)…")
        profile = self._pcleaner_config.current_profile
        # ── Fix: all intermediates go inside temp_chapters/<uuid>/_work (organized, per-chapter) ──
        output_dir_path = Path(output_dir).resolve()
        cache_dir = output_dir_path / "_work"
        cache_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"📁 Work dir (intermediates): {cache_dir}")

        # Discover images (filter to supported types)
        paths = [Path(p) for p in image_paths]
        try:
            paths, _ = hp.discover_all_images(paths, cfg.SUPPORTED_IMG_TYPES)
        except OSError as e:
            logger.error(f"❌ Image discovery failed: {e}")
            return

        if not paths:
            logger.warning("⚠️ No images found to detect")
            return

        # Split long strips (no-op in server build)
        pass

        # GPU detection & model path
        gpu = cuda_available() or mps_available()
        model_path = self._pcleaner_config.get_model_path(gpu)

        # Run text detection with CUDA fallback
        logger.info("🤖 Running text detection AI model…")
        try:
            pp.generate_mask_data(
                paths,
                config_general=profile.general,
                config_detector=profile.text_detector,
                model_path=model_path,
                output_dir=cache_dir,
            )
        except (NotImplementedError, RuntimeError, ValueError) as e:
            estr = str(e)
            if any(kw in estr for kw in ("CUDA", "cuda", "out of memory", "kernel image", "CUDNN")):
                logger.warning("⚠️ GPU/CUDA error detected. Falling back to CPU models…")
                try:
                    cpu_model_path = self._pcleaner_config.get_model_path(False)
                    pp.generate_mask_data(
                        paths,
                        config_general=profile.general,
                        config_detector=profile.text_detector,
                        model_path=cpu_model_path,
                        output_dir=cache_dir,
                    )
                except Exception as e2:
                    logger.error(f"❌ CPU fallback also failed: {e2}")
                    raise e2
            else:
                raise

        logger.info("✅ [inline] Detection step complete")

    # ------------------------------------------------------------------
    # Step 3 — pre-processing + masking + denoising + inpainting
    # ------------------------------------------------------------------
    def run_cleaning(self, image_paths: list[str], output_dir: str):
        """Run the remaining cleaning stages on previously-detected images."""
        logger.info(f"🧹 [inline] Running pre-processing + masking + denoising + inpainting…")
        profile = self._pcleaner_config.current_profile
        # ── Fix: same per-chapter work dir as run_detection (Upscale-bot style) ──
        output_dir_path = Path(output_dir).resolve()
        cache_dir = output_dir_path / "_work"
        cache_dir.mkdir(parents=True, exist_ok=True)

        # Override skip flags based on profile config
        skip_denoising = not profile.denoiser.denoising_enabled
        skip_inpainting = not profile.inpainter.inpainting_enabled

        # ── Pre-Processing ──────────────────────────────────────────────
        logger.info("📦 Step 1/4: Pre-processing…")

        json_files = list(cache_dir.glob("*#raw.json"))
        max_workers = min(len(json_files), os.cpu_count() or 4, 8)
        if max_workers <= 1 or len(json_files) <= 2:
            for jf in json_files:
                pp.prep_json_file(
                    jf,
                    preprocessor_conf=profile.preprocessor,
                    cache_masks=False,
                )
        else:
            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                list(pool.map(
                    lambda jf: pp.prep_json_file(
                        jf,
                        preprocessor_conf=profile.preprocessor,
                        cache_masks=False,
                    ),
                    json_files,
                ))

        # ── Masking ─────────────────────────────────────────────────────
        logger.info("🎭 Step 2/4: Masking…")
        clean_json_files = list(cache_dir.glob("*#clean.json"))
        mask_data_list = [
            st.MaskerData(
                jf,
                cache_dir,
                profile.masker,
                False,   # extract_text
                False,   # show_masks (cache_masks)
                False,   # debug
                full_lama_inpainting=profile.general.full_lama_inpainting,
            )
            for jf in clean_json_files
        ]

        core_limit = profile.masker.max_threads or multiprocessing.cpu_count()
        pool_size = min(core_limit, len(mask_data_list))
        if pool_size > 1:
            with Pool(processes=pool_size) as pool:
                for _ in pool.imap(ma.mask_page, mask_data_list):
                    pass
        else:
            for md_item in mask_data_list:
                ma.mask_page(md_item)

        # ── Denoising ───────────────────────────────────────────────────
        if not skip_denoising:
            logger.info("🔊 Step 3/4: Denoising…")
            mask_json_files = list(cache_dir.glob("*#mask_data.json"))
            denoise_data_list = [
                st.DenoiserData(jf, cache_dir, profile.denoiser, False)
                for jf in mask_json_files
            ]

            core_limit = profile.denoiser.max_threads or multiprocessing.cpu_count()
            pool_size = min(core_limit, len(denoise_data_list))
            if pool_size > 1:
                with Pool(processes=pool_size) as pool:
                    for _ in pool.imap(dn.denoise_page, denoise_data_list):
                        pass
            else:
                for dd in denoise_data_list:
                    dn.denoise_page(dd)
        else:
            logger.info("⏭️ Step 3/4: Denoising disabled in profile, skipping")

        # ── Inpainting ──────────────────────────────────────────────────
        if not skip_inpainting:
            logger.info("🖌️ Step 4/4: Inpainting…")
            page_json_files = list(cache_dir.glob("*#clean.json"))
            mask_json_files = list(cache_dir.glob("*#mask_data.json"))

            # Pair page JSONs with matching mask JSONs
            zipped = []
            for pj in page_json_files:
                for mj in mask_json_files:
                    if mj.name.startswith(pj.name.split("#")[0]):
                        zipped.append((pj, mj))
                        break

            inpainter_model = _ModelCache.get_inpainter(self._pcleaner_config)
            inpaint_data_list = [
                st.InpainterData(
                    pj,
                    mj,
                    cache_dir,
                    profile.masker,
                    profile.denoiser,
                    profile.inpainter,
                    False,   # debug
                    full_lama_inpainting=profile.general.full_lama_inpainting,
                )
                for pj, mj in zipped
            ]

            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = {
                    executor.submit(ip.inpaint_page, d, inpainter_model): d
                    for d in inpaint_data_list
                }
                for f in as_completed(futures):
                    f.result()  # propagate exceptions
        else:
            logger.info("⏭️ Step 4/4: Inpainting disabled in profile, skipping")

        # ── Export ──────────────────────────────────────────────────────
        logger.info("📤 Exporting results…")
        cleaned_outputs_whitelist = [
            ost.Output.masked_output,
            ost.Output.denoised_output,
            ost.Output.inpainted_output,
        ]
        masked_outputs_whitelist = [
            ost.Output.final_mask,
            ost.Output.denoise_mask,
            ost.Output.inpainted_mask,
        ]
        text_outputs_whitelist = []

        if skip_denoising:
            if ost.Output.denoised_output in cleaned_outputs_whitelist:
                cleaned_outputs_whitelist.remove(ost.Output.denoised_output)
            if ost.Output.denoise_mask in masked_outputs_whitelist:
                masked_outputs_whitelist.remove(ost.Output.denoise_mask)
        if skip_inpainting:
            if ost.Output.inpainted_output in cleaned_outputs_whitelist:
                cleaned_outputs_whitelist.remove(ost.Output.inpainted_output)
            if ost.Output.inpainted_mask in masked_outputs_whitelist:
                masked_outputs_whitelist.remove(ost.Output.inpainted_mask)

        export_targets = ie.discover_viable_outputs(
            cache_dir, cleaned_outputs_whitelist, masked_outputs_whitelist, text_outputs_whitelist
        )

        # merge_export_splits is a no-op in server build (no split/merge)

        export_data = [
            (
                target.original_path,
                target.export_path,
                cache_dir,
                target.uuid,
                target.outputs,
                output_dir_path,   # ← write exports directly to chapter folder
                profile.general.preferred_file_type,
                profile.general.preferred_mask_file_type,
                profile.denoiser.denoising_enabled,
                profile.general.layered_export,
            )
            for target in export_targets
        ]

        core_limit = profile.general.max_threads_export or multiprocessing.cpu_count()
        # Clamp to cpu count but use ProcessPool for CPU-bound PIL save (bypasses GIL, uses all 20 cores)
        pool_size = min(core_limit, len(export_data), multiprocessing.cpu_count())
        if pool_size > 1:
            try:
                from concurrent.futures import ProcessPoolExecutor
                with ProcessPoolExecutor(max_workers=pool_size) as executor:
                    futures = [executor.submit(ie.copy_to_output_batched, d) for d in export_data]
                    for f in as_completed(futures):
                        f.result()
            except Exception as proc_err:
                logger.warning(f"ProcessPool failed ({proc_err}), falling back to ThreadPool")
                with ThreadPoolExecutor(max_workers=pool_size) as executor:
                    futures = [executor.submit(ie.copy_to_output_batched, d) for d in export_data]
                    for f in as_completed(futures):
                        f.result()
        else:
            for ed in export_data:
                ie.copy_to_output_batched(ed)

        logger.info("✅ [inline] Cleaning step complete")

    # ------------------------------------------------------------------
    # Pre-compress — مقتبس من Upscale bot image_pipeline.py
    # يقرأ كل الصور في input_dir، يعمل thumbnail 3000، يحول RGBA/P إلى RGB
    # بخلفية بيضاء، يحفظ JPEG quality 85 بنفس الاسم لكن بامتداد .jpg
    # ------------------------------------------------------------------
    def pre_compress_images(
        self, input_dir: Path, quality: int = 85, max_dimension: int = 3000
    ) -> list[Path]:
        """Pre-compress images to JPEG (quality=85, max 3000px) for faster VRAM.

        Scans ``input_dir`` shallow (flat, no subdirs) for supported images,
        converts RGBA/P → RGB on white, thumbnails to ``max_dimension``,
        saves as ``.jpg`` with ``quality``.  Original files with different
        suffix are removed to avoid duplicates.  ThreadPool-accelerated.
        """
        input_dir = Path(input_dir).resolve()
        if not input_dir.exists():
            logger.warning(f"⚠️ pre_compress: input_dir not found: {input_dir}")
            return []

        # Shallow discovery — only flat files, exclude subdirs like parts/output/_work
        candidates: list[Path] = []
        for p in input_dir.iterdir():
            if p.is_dir():
                continue
            if p.suffix.lower() not in [s.lower() for s in cfg.SUPPORTED_IMG_TYPES]:
                continue
            if not p.is_file():
                continue
            candidates.append(p)

        if not candidates:
            logger.info("ℹ️ pre_compress: no images found to compress")
            return []

        logger.info(f"🗜️ Pre-compressing {len(candidates)} image(s) → JPEG q={quality}, max={max_dimension}px ...")

        def _compress_one(src: Path) -> Path:
            dst = src.with_suffix(".jpg")
            try:
                with Image.open(src) as im:
                    dpi = im.info.get("dpi")
                    # Handle transparency / palette
                    if im.mode in ("RGBA", "LA", "PA"):
                        bg = Image.new("RGB", im.size, (255, 255, 255))
                        if im.mode == "PA":
                            im = im.convert("RGBA")
                        # Use alpha channel as mask
                        alpha = im.split()[-1]
                        bg.paste(im, mask=alpha)
                        im = bg
                    elif im.mode == "P":
                        if "transparency" in im.info:
                            im = im.convert("RGBA")
                            bg = Image.new("RGB", im.size, (255, 255, 255))
                            bg.paste(im, mask=im.split()[-1])
                            im = bg
                        else:
                            im = im.convert("RGB")
                    elif im.mode != "RGB":
                        im = im.convert("RGB")

                    # Thumbnail preserves aspect ratio
                    im.thumbnail((max_dimension, max_dimension), Image.LANCZOS)

                    save_kwargs: dict = {"quality": quality, "optimize": True, "progressive": True}
                    if dpi:
                        save_kwargs["dpi"] = dpi
                    im.save(dst, "JPEG", **save_kwargs)

                # Remove original if suffix changed (e.g. .png → .jpg) to avoid duplicate stems
                if dst.resolve() != src.resolve():
                    try:
                        src.unlink()
                    except Exception:
                        pass
                return dst
            except Exception as e:
                logger.warning(f"⚠️ pre_compress failed for {src.name}: {e}")
                return src

        max_workers = min(len(candidates), os.cpu_count() or 4, 8)
        results: list[Path] = []
        if max_workers <= 1:
            for c in candidates:
                results.append(_compress_one(c))
        else:
            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                future_to_src = {pool.submit(_compress_one, src): src for src in candidates}
                for fut in as_completed(future_to_src):
                    try:
                        results.append(fut.result())
                    except Exception as e:
                        src = future_to_src[fut]
                        logger.warning(f"⚠️ pre_compress thread failed for {src.name}: {e}")
                        results.append(src)

        results = natural_sort(results)
        logger.info(f"✅ Pre-compress done: {len(results)} file(s)")
        return results

    # ------------------------------------------------------------------
    # Split — مقتبس من pcleaner/main.py القديم + Upscale bot
    # لكل صورة استدعِ image_ops.calculate_best_splits ثم قص عبر crop
    # ------------------------------------------------------------------
    def split_all_images(self, input_dir: Path, parts_dir: Path) -> list[Path]:
        """Split long-strip images into parts using profile's split settings.

        For each image in ``input_dir`` calls ``image_ops.calculate_best_splits``
        with values from ``profile.general`` (preferred_split_height=2000,
        split_tolerance_margin=500, split_long_strips, long_strip_aspect_ratio).
        No-split images are copied directly to ``parts_dir``; split ones are
        cropped per segment and saved as ``{stem}_part{idx:03d}{suffix}`` with
        DPI preserved. ThreadPool-accelerated and natural-sorted.
        """
        import app.services.image_ops as ops

        input_dir = Path(input_dir).resolve()
        parts_dir = Path(parts_dir).resolve()
        parts_dir.mkdir(parents=True, exist_ok=True)

        profile = self._pcleaner_config.current_profile
        preferred = int(profile.general.preferred_split_height or 2000)
        tolerance = int(profile.general.split_tolerance_margin or 500)
        do_split = bool(profile.general.split_long_strips)
        aspect = float(profile.general.long_strip_aspect_ratio or 0.33)

        # Shallow discovery of input images (after pre_compress they are .jpg)
        candidates: list[Path] = []
        for p in input_dir.iterdir():
            if p.is_dir():
                continue
            if p.suffix.lower() not in [s.lower() for s in cfg.SUPPORTED_IMG_TYPES]:
                continue
            candidates.append(p)
        candidates = natural_sort(candidates)

        if not candidates:
            logger.info("ℹ️ split_all: no images to split")
            return []

        logger.info(
            f"✂️ Splitting {len(candidates)} image(s) "
            f"(preferred={preferred}, tol={tolerance}, split={do_split}, aspect={aspect}) ..."
        )

        all_parts: list[Path] = []
        # We collect splits first (fast, single-threaded calc), then crop in threadpool
        # calculate_best_splits opens image, so we keep it sequential to avoid too many open files
        split_jobs: list[tuple[Path, list[int]]] = []
        for img_path in candidates:
            try:
                splits = ops.calculate_best_splits(img_path, preferred, tolerance, do_split, aspect)
            except Exception as e:
                logger.warning(f"⚠️ calculate_best_splits failed for {img_path.name}: {e}")
                splits = []
            split_jobs.append((img_path, splits))

        def _process_one(job: tuple[Path, list[int]]) -> list[Path]:
            src, splits = job
            suffix = src.suffix  # preserve original suffix (now .jpg after pre_compress)
            stem = src.stem
            if not splits:
                # No split → copy directly
                dst = parts_dir / src.name
                try:
                    if src.resolve() != dst.resolve():
                        shutil.copy2(src, dst)
                    else:
                        dst = src  # already in parts_dir? shouldn't happen since input_dir != parts_dir
                    return [dst]
                except Exception as e:
                    logger.warning(f"⚠️ copy without split failed for {src.name}: {e}")
                    return [src]

            # Split into segments via crop
            try:
                with Image.open(src) as im:
                    dpi = im.info.get("dpi")
                    w, h = im.size
                    y1 = 0
                    out_paths: list[Path] = []
                    # Build y coords: splits + [h]
                    y_coords = list(splits) + [h]
                    for idx, y2 in enumerate(y_coords):
                        # Crop segment
                        segment = im.crop((0, y1, w, y2))
                        part_name = f"{stem}_part{idx:03d}{suffix}"
                        dst = parts_dir / part_name
                        save_kwargs: dict = {}
                        if dpi:
                            save_kwargs["dpi"] = dpi
                        # Preserve format based on suffix
                        fmt = suffix.lstrip(".").upper()
                        if fmt == "JPG":
                            fmt = "JPEG"
                        # For JPEG, use quality 92 to avoid re-compression artifacts before cleaning
                        if fmt in ("JPEG", "JPG"):
                            save_kwargs.update({"quality": 92, "optimize": True})
                        segment.save(dst, fmt if fmt != "JPEG" else "JPEG", **save_kwargs)
                        out_paths.append(dst)
                        y1 = y2
                    return out_paths
            except Exception as e:
                logger.warning(f"⚠️ split crop failed for {src.name}: {e}")
                # Fallback: copy original
                try:
                    dst = parts_dir / src.name
                    shutil.copy2(src, dst)
                    return [dst]
                except Exception:
                    return []

        max_workers = min(len(split_jobs), os.cpu_count() or 4, 8)
        if max_workers <= 1 or len(split_jobs) <= 2:
            for job in split_jobs:
                all_parts.extend(_process_one(job))
        else:
            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                futures = {pool.submit(_process_one, job): job[0] for job in split_jobs}
                # Collect in submission order then sort naturally at the end
                tmp_results: list[Path] = []
                for fut in as_completed(futures):
                    try:
                        tmp_results.extend(fut.result())
                    except Exception as e:
                        logger.warning(f"⚠️ split thread failed: {e}")
                all_parts = tmp_results

        all_parts = natural_sort(all_parts)
        logger.info(f"✅ Split done: {len(candidates)} → {len(all_parts)} part(s)")
        return all_parts

    # ------------------------------------------------------------------
    # Merge — مقتبس من Upscale bot + pcleaner image_export.stitch_images
    # جمع حسب stem، رتب الأجزاء و ادمج عبر stitch_images أو يدوي
    # ------------------------------------------------------------------
    def merge_parts_from_disk(
        self, cleaned_parts: list[Path], original_images: list[Path], output_dir: Path
    ) -> list[Path]:
        """Merge cleaned parts back into full images (disk-based, Upscale-bot style).

        Groups ``cleaned_parts`` by original stem (stripping ``_part###``),
        sorts each group naturally, stitches via ``image_ops.stitch_images``
        or manual ``Image.new + paste`` fallback with JPEG 65500px limit
        handling, writes to ``output_dir``, then renames sequentially
        ``001.jpg, 002.jpg`` etc. in natural order of ``original_images``.
        """
        import app.services.image_ops as ops

        output_dir = Path(output_dir).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)

        if not cleaned_parts:
            logger.warning("⚠️ merge: no cleaned_parts to merge")
            return []

        # Group by stem: strip _part### suffix
        part_re = re.compile(r"^(.*)_part\d+$")
        grouped: dict[str, list[Path]] = {}
        # Also track which original_stems have single-file (no split) case
        for p in cleaned_parts:
            stem = p.stem
            m = part_re.match(stem)
            key = m.group(1) if m else stem
            grouped.setdefault(key, []).append(p)

        # Sort each group's parts naturally
        for k in grouped:
            grouped[k] = natural_sort(grouped[k])

        # Determine order: sort keys by natural order of original_images stems
        # original_images may be post-pre_compress (.jpg) paths; extract stems
        orig_stem_order = [Path(o).stem for o in original_images]
        # Create a ranking map for natural order fallback
        def _rank(key: str) -> int:
            try:
                return orig_stem_order.index(key)
            except ValueError:
                # If cleaned key not in original (e.g. after ext change), try fuzzy
                return 10_000 + natural_sort([Path(key)])[0].name.__hash__() % 1000

        sorted_keys = sorted(grouped.keys(), key=lambda k: (_rank(k), _natural_sort_key(k)))

        merged_paths: list[Path] = []
        # Determine preferred output extension from config or first part
        profile = self._pcleaner_config.current_profile
        preferred_ext = profile.general.preferred_file_type or ".png"
        # Ensure leading dot
        if not preferred_ext.startswith("."):
            preferred_ext = f".{preferred_ext}"
        # For merge we use preferred_ext, but if it's empty fallback to .jpg
        if not preferred_ext:
            preferred_ext = ".png"

        for key in sorted_keys:
            parts = grouped[key]
            if not parts:
                continue
            # Output filename before sequential rename
            # Use preferred_ext for final, but if parts are .png and pref is .png keep it
            merged_name = f"{key}{preferred_ext}"
            merged_path = output_dir / merged_name

            try:
                if len(parts) == 1 and part_re.match(parts[0].stem) is None:
                    # No-split image: just copy the single cleaned file
                    src = parts[0]
                    # Ensure correct extension: if preferred_ext differs, convert via save
                    if src.suffix.lower() != preferred_ext.lower():
                        with Image.open(src) as im:
                            dpi = im.info.get("dpi")
                            save_kwargs: dict = {}
                            if dpi:
                                save_kwargs["dpi"] = dpi
                            fmt = preferred_ext.lstrip(".").upper()
                            if fmt == "JPG":
                                fmt = "JPEG"
                            if fmt == "JPEG":
                                if im.mode in ("RGBA", "LA", "P"):
                                    bg = Image.new("RGB", im.size, (255, 255, 255))
                                    if im.mode == "P":
                                        im = im.convert("RGBA")
                                    if im.mode in ("RGBA", "LA"):
                                        bg.paste(im, mask=im.split()[-1])
                                        im = bg
                                    else:
                                        im = im.convert("RGB")
                                elif im.mode != "RGB":
                                    im = im.convert("RGB")
                                save_kwargs.update({"quality": 92, "optimize": True})
                            im.save(merged_path, fmt, **save_kwargs)
                    else:
                        shutil.copy2(src, merged_path)
                else:
                    # Multi-part → stitch
                    # Try using image_ops.stitch_images first
                    try:
                        # Need to handle JPEG 65500 limit: pre-check total height
                        # Quick check: sum heights
                        total_h = 0
                        max_w = 0
                        for pp in parts:
                            with Image.open(pp) as im:
                                w, h = im.size
                                max_w = max(max_w, w)
                                total_h += h
                        # If JPEG and height exceeds limit, switch to PNG
                        if preferred_ext.lower() in (".jpg", ".jpeg") and total_h > 65500:
                            logger.warning(f"⚠️ Merged height {total_h} exceeds JPEG limit → saving as PNG for {key}")
                            merged_path = merged_path.with_suffix(".png")
                        ops.stitch_images(parts, merged_path)
                    except Exception as stitch_err:
                        logger.warning(f"⚠️ stitch_images failed for {key} ({stitch_err}), falling back to manual stitch")
                        # Manual stitch
                        images = [Image.open(p) for p in parts]
                        widths, heights = zip(*(im.size for im in images))
                        total_w = max(widths)
                        total_h = sum(heights)
                        # JPEG limit handling for manual too
                        if merged_path.suffix.lower() in (".jpg", ".jpeg") and total_h > 65500:
                            merged_path = merged_path.with_suffix(".png")
                        # Determine mode
                        modes = [im.mode for im in images]
                        # Hierarchy for greatest mode
                        hierarchy = ["1", "L", "P", "LA", "RGB", "RGBA", "CMYK", "YCbCr"]
                        greatest = max(modes, key=lambda m: hierarchy.index(m) if m in hierarchy else 99)
                        stitched = Image.new(greatest, (total_w, total_h))
                        y_off = 0
                        for im in images:
                            stitched.paste(im, (0, y_off))
                            y_off += im.size[1]
                            im.close()
                        # Save with dpi from first image if available
                        try:
                            dpi = images[0].info.get("dpi") if images else None
                        except Exception:
                            dpi = None
                        save_kwargs = {}
                        if dpi:
                            save_kwargs["dpi"] = dpi
                        if merged_path.suffix.lower() in (".jpg", ".jpeg"):
                            if stitched.mode in ("RGBA", "LA", "P"):
                                bg = Image.new("RGB", stitched.size, (255, 255, 255))
                                if stitched.mode == "P":
                                    stitched = stitched.convert("RGBA")
                                if stitched.mode in ("RGBA", "LA"):
                                    bg.paste(stitched, mask=stitched.split()[-1])
                                    stitched = bg
                                else:
                                    stitched = stitched.convert("RGB")
                            save_kwargs.update({"quality": 92, "optimize": True})
                        stitched.save(merged_path, **save_kwargs)
                        for im in images:
                            try:
                                im.close()
                            except Exception:
                                pass
                merged_paths.append(merged_path)
            except Exception as e:
                logger.error(f"❌ merge failed for {key}: {e}")
                continue

        # Sequential rename: 001.ext, 002.ext etc. in natural order
        merged_paths = natural_sort(merged_paths)
        final_paths: list[Path] = []
        # Group by extension handling: if after JPEG→PNG switch we have mixed exts, keep each file's own ext for sequential name
        for idx, src in enumerate(merged_paths):
            ext = src.suffix  # keep actual suffix (may be .png after fallback)
            dst = output_dir / f"{idx+1:03d}{ext}"
            try:
                if src.resolve() != dst.resolve():
                    if dst.exists():
                        dst.unlink()
                    src.rename(dst)
                final_paths.append(dst)
            except Exception as e:
                logger.warning(f"⚠️ sequential rename failed {src.name} → {dst.name}: {e}")
                final_paths.append(src)

        final_paths = natural_sort(final_paths)
        logger.info(f"✅ Merge done: {len(cleaned_parts)} part(s) → {len(final_paths)} final image(s)")
        return final_paths

    # ------------------------------------------------------------------
    # Single-call full pipeline (all steps at once)
    # ------------------------------------------------------------------
    def run_full(self, image_paths: list[str], output_dir: str) -> list[str]:
        """Run the complete pipeline (detection → export) and return output paths."""
        logger.info(f"🚀 [inline] Full pipeline for {len(image_paths)} image(s)…")
        self.run_detection(image_paths, output_dir)
        self.run_cleaning(image_paths, output_dir)
        return self._collect_outputs(image_paths, output_dir)

    # ------------------------------------------------------------------
    # Collect output paths after cleaning
    # ------------------------------------------------------------------
    @staticmethod
    def _collect_outputs(image_paths: list[str], image_dir: str) -> list[str]:
        """Locate the cleaned output files and normalise their names."""
        output_paths: list[str] = []

        for image_path in image_paths:
            base_name = os.path.basename(image_path)
            stem = os.path.splitext(base_name)[0]

            all_outputs = glob.glob(
                os.path.join(image_dir, "**", f"*{stem}*.*"), recursive=True
            )
            original_basenames = {os.path.basename(p) for p in image_paths}
            all_outputs = [
                f
                for f in all_outputs
                if os.path.isfile(f)
                and "_raw_" not in f
                and "#" not in f
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
                if os.path.exists(direct_output) and os.path.abspath(
                    direct_output
                ) != os.path.abspath(image_path):
                    final_output = direct_output

            if not final_output or not os.path.exists(final_output):
                logger.error(f"❌ No cleaned output found for {base_name}")
                continue

            output_ext = os.path.splitext(final_output)[1]
            base_final = os.path.splitext(base_name)[0] + output_ext
            new_fixed_path = os.path.join(image_dir, base_final)
            try:
                if os.path.abspath(final_output) != os.path.abspath(new_fixed_path):
                    if os.path.exists(new_fixed_path):
                        os.remove(new_fixed_path)
                    os.rename(final_output, new_fixed_path)
                elif "_clean" in os.path.basename(final_output):
                    os.rename(final_output, new_fixed_path)
                final_output = new_fixed_path
            except Exception as move_err:
                logger.warning(f"⚠️ Failed to normalise output for {base_name}: {move_err}")

            output_paths.append(final_output)

        return output_paths

    # ------------------------------------------------------------------
    # Cache directory accessor
    # ------------------------------------------------------------------
    @property
    def cache_dir(self) -> Path | None:
        try:
            return self._pcleaner_config.get_cleaner_cache_dir()
        except Exception:
            return None
