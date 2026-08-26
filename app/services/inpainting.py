import numpy as np
import cv2
from PIL import Image
from loguru import logger
import app.config as cfg
import app.services.model_downloader as md
import app.services.output_structures as ost
import app.services.structures as st
import app.services.image_ops as ops
from app.utils.device import gpu_available as _gpu_avail


class _SimpleLama:
    """Minimal LaMa inpainting wrapper using simple-lama-inpainting. Thread-safe lazy load."""

    def __init__(self, model_path=None):
        self._model = None
        self._lock = __import__('threading').Lock()
        self._model_path = model_path

    def _load_model(self):
        if self._model is not None:
            return
        with self._lock:
            if self._model is not None:
                return
            # Respect the sandbox path (like old pcleaner did via LAMA_MODEL env)
            if self._model_path is not None:
                try:
                    from pathlib import Path as _P
                    p = _P(self._model_path)
                    if p.is_file():
                        import os as _os
                        _os.environ["LAMA_MODEL"] = str(p)
                except Exception:
                    pass
            try:
                from simple_lama_inpainting import SimpleLama as _SimpleLama
                self._model = _SimpleLama()
            except ImportError:
                raise RuntimeError(
                    "simple-lama-inpainting package not installed. "
                    "Install it with: pip install simple-lama-inpainting"
                )

    def __call__(self, image: Image.Image, mask: Image.Image) -> Image.Image:
        self._load_model()
        result = self._model(image, mask)
        return result


class InpaintingModel:
    def __init__(self, config: cfg.Config) -> None:
        md.ensure_inpainting_available(config)

        self.model_path = md.get_inpainting_model_path(config)
        if not self.model_path.is_file():
            raise FileNotFoundError(f"Inpainting model not found after ensure: {self.model_path}")

        # Make simple-lama use our sandbox file (prevents downloading big-lama.pt to hub cache)
        import os as _os
        _os.environ["LAMA_MODEL"] = str(self.model_path)

        if _gpu_avail():
            logger.info("Inpainting: Using GPU acceleration")
        else:
            logger.warning("Inpainting: GPU not available, using CPU (slower)")

        self.simple_lama = _SimpleLama(model_path=self.model_path)
        # Eager load (not lazy) so 4-thread pool doesn't trigger 4 parallel downloads
        self.simple_lama._load_model()

        model_name = self.model_path.name
        logger.info(f"Inpainting Model: {model_name} | Device: {'GPU' if _gpu_avail() else 'CPU'} | Path: {self.model_path}")

    def __call__(self, image: Image, mask: Image) -> Image:
        inpainted_image = self.simple_lama(image, mask)
        if inpainted_image.size != image.size:
            width, height = image.size
            inpainted_image = inpainted_image.crop((0, 0, width, height))
        return inpainted_image


def inpaint_page(i_data: st.InpainterData, model: InpaintingModel) -> st.InpaintingAnalytic:
    m_conf = i_data.masker_config
    d_conf = i_data.denoiser_config
    i_conf = i_data.inpainter_config

    page_data = st.PageData.from_json(i_data.page_data_json_path.read_text(encoding="utf-8"))
    mask_data = st.MaskData.from_json(i_data.mask_data_json_path.read_text(encoding="utf-8"))

    # Load images
    original_image = Image.open(mask_data.original_path).convert("RGB")
    exact_mask_pil = Image.open(page_data.mask_path)
    mask_pil = Image.open(mask_data.mask_path)

    h_cache, w_cache = mask_pil.size[1], mask_pil.size[0]
    orig_size = original_image.size
    cache_size = mask_pil.size
    original_path: Path = mask_data.original_path
    path_gen = ost.OutputPathGenerator(original_path, i_data.cache_dir, i_data.page_data_json_path)

    # Convert masks to numpy uint8 arrays (0 or 255)
    exact_np = np.asarray(exact_mask_pil.convert("1"), dtype=np.uint8) * 255
    mask_np = np.asarray(mask_pil.convert("1"), dtype=np.uint8) * 255

    # Pre-create combined mask arrays at cached resolution
    combined_inpaint = np.zeros((h_cache, w_cache), dtype=np.uint8)
    combined_isolation = np.zeros((h_cache, w_cache), dtype=np.uint8)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    analytics_thicknesses: list[int] = []
    has_any_box = False

    for box, deviation, failed, thickness in mask_data.boxes_with_stats:
        src = exact_np if failed else mask_np
        bh = box.y2 - box.y1
        bw = box.x2 - box.x1
        if bh <= 0 or bw <= 0:
            continue

        box_mask = src[box.y1:box.y2, box.x1:box.x2].copy()

        # Apply minimum thickness growth regardless of failure status.
        if m_conf.min_mask_thickness > 0:
            box_mask = cv2.dilate(box_mask, kernel, iterations=m_conf.min_mask_thickness)

        growth = i_conf.min_inpainting_radius
        if deviation > 0:
            growth += int(deviation * i_conf.inpainting_radius_multiplier)
            growth = min(growth, i_conf.max_inpainting_radius)

        iso_growth = growth + i_conf.inpainting_isolation_radius
        has_any_box = True
        analytics_thicknesses.append(growth)

        pad = iso_growth
        p_y1 = max(0, box.y1 - pad)
        p_y2 = min(h_cache, box.y2 + pad)
        p_x1 = max(0, box.x1 - pad)
        p_x2 = min(w_cache, box.x2 + pad)
        ph = p_y2 - p_y1
        pw = p_x2 - p_x1

        padded = np.zeros((ph, pw), dtype=np.uint8)
        m_y1 = box.y1 - p_y1
        m_x1 = box.x1 - p_x1
        padded[m_y1:m_y1 + bh, m_x1:m_x1 + bw] = box_mask

        inp_grown = cv2.dilate(padded, kernel, iterations=growth)
        np.maximum(combined_inpaint[p_y1:p_y2, p_x1:p_x2], inp_grown, out=combined_inpaint[p_y1:p_y2, p_x1:p_x2])

        iso_grown = cv2.dilate(padded, kernel, iterations=iso_growth)
        np.maximum(combined_isolation[p_y1:p_y2, p_x1:p_x2], iso_grown, out=combined_isolation[p_y1:p_y2, p_x1:p_x2])

    if not has_any_box:
        inpainted_image = original_image
        inpainted_image.save(path_gen.inpainting)
        original_image.save(path_gen.clean_inpainted)
        return st.InpaintingAnalytic(tuple(analytics_thicknesses), original_path)

    # Convert numpy masks to PIL
    combined_inpaint_pil = Image.fromarray(combined_inpaint, mode="L")
    combined_isolation_pil = Image.fromarray(combined_isolation, mode="L")

    # ── Mask Conditioning ──────────────────────────────────────────────────────
    # Send a BINARY mask to LaMa (thresholded, not faded).
    # LaMa was trained on binary masks; soft masks cause artifacts (black blocks).
    combined_inpaint_binary = combined_inpaint_pil.point(lambda x: 255 if x > 127 else 0)

    # Resize binary mask to original size (NEAREST to preserve hard edges)
    binary_mask_for_lama = combined_inpaint_binary.resize(orig_size, resample=Image.NEAREST)
    inpainted_image = model(original_image, binary_mask_for_lama)

    # ── Alpha Masking / Compositing ────────────────────────────────────────────
    # Build a faded alpha mask for smooth blending of the inpainted result.
    if i_conf.inpainting_fade_radius:
        fade_mask = combined_inpaint_binary.copy()
        fade_mask = ops.fade_mask_edges(fade_mask, i_conf.inpainting_fade_radius)
    else:
        fade_mask = combined_inpaint_binary.copy()
    fade_mask = fade_mask.resize(orig_size, resample=Image.BILINEAR)

    # Isolation mask: clip to binary isolation area (binary, no fade)
    if orig_size != cache_size:
        combined_isolation_pil = combined_isolation_pil.resize(orig_size, resample=Image.NEAREST)

    # Final alpha mask: fade_mask, but only within the isolation area
    # (outside isolation = fully transparent, inside = faded values)
    isolation_arr = np.array(combined_isolation_pil, dtype=np.uint8)
    fade_arr = np.array(fade_mask, dtype=np.uint8)
    final_alpha = np.where(isolation_arr > 127, fade_arr, 0).astype(np.uint8)
    final_alpha_pil = Image.fromarray(final_alpha, mode="L")

    # Build cleaned_image: start with original, add the median-color mask overlay
    cleaned_image = original_image.convert("RGBA")
    if orig_size != cache_size:
        mask_pil_resized = mask_pil.resize(orig_size, resample=Image.NEAREST)
    else:
        mask_pil_resized = mask_pil
    cleaned_image.paste(mask_pil_resized, (0, 0), mask_pil_resized)
    if d_conf.denoising_enabled and path_gen.noise_mask.is_file():
        noise_mask = Image.open(path_gen.noise_mask)
        cleaned_image.paste(noise_mask, (0, 0), noise_mask)

    # Composite inpainted over cleaned using the faded alpha mask
    inpainted_rgba = inpainted_image.convert("RGBA")
    inpainted_rgba.putalpha(final_alpha_pil)
    cleaned_image.alpha_composite(inpainted_rgba)
    cleaned_image.putalpha(255)

    inpainted_image.save(path_gen.inpainting)
    cleaned_image.save(path_gen.clean_inpainted)

    return st.InpaintingAnalytic(tuple(analytics_thicknesses), original_path)
