"""
Bubble-aware text detector using comic-text-bubble-detector (RT-DETRv2).
Replaces the legacy 3-headed comictextdetector.pt with a modern object detector
trained on 11k manga/webtoon/manhua images for better accuracy and fewer false positives.
"""
import numpy as np
import cv2
import torch
from loguru import logger

from .utils.textblock import TextBlock, examine_textblk

# ---------------------------------------------------------------------------
# HuggingFace model identifiers
# ---------------------------------------------------------------------------
HF_MODEL_ID = "ogkalu/comic-text-and-bubble-detector"

MODEL_LABELS = {0: "bubble", 1: "text_bubble", 2: "text_free"}
TEXT_CLASS_IDS = {1, 2}  # only produce masks for actual text


class BubbleAwareTextDetector:
    """
    Drop-in replacement for pcleaner's legacy TextDetector.

    Uses the RT-DETRv2-based comic-text-bubble-detector from HuggingFace.
    Trained on 11k manga / webtoon / manhua images.

    - Detects speech bubbles AND text regions.
    - Only masks text (ignores empty bubbles -> zero false positives on people/objects).
    - Output format identical to legacy TextDetector: (mask, mask_refined, blk_list).
    - Accepts ``model_dir`` to persist the HuggingFace cache to a fixed local path
      (e.g. ``pcleaner/models/text_detector/``) so the model is downloaded once
      and reused across restarts.

    Attributes
    ----------
    device : str
    half : bool
    conf_thresh : float
    model_dir : Path | None
    """

    def __init__(
        self,
        model_path=None,
        input_size=640,
        device="cpu",
        half=False,
        conf_thresh=0.35,
        model_dir=None,
    ):
        self.device = device
        self.half = half and device != "cpu"
        self.conf_thresh = conf_thresh
        self.input_size = input_size
        self.model_dir = model_dir

        from transformers import RTDetrV2ForObjectDetection, RTDetrImageProcessor

        # If model_dir is given, use it as the HuggingFace cache directory,
        # so the model is stored in a fixed sandboxed path (not ~/.cache/huggingface/).
        cache_kw = {"cache_dir": str(model_dir)} if model_dir else {}

        logger.info(
            f"Loading bubble-aware detector from {HF_MODEL_ID} "
            f"{' (cache: ' + str(model_dir) + ')' if model_dir else ''} ..."
        )
        self.processor = RTDetrImageProcessor.from_pretrained(HF_MODEL_ID, **cache_kw)
        self.model = RTDetrV2ForObjectDetection.from_pretrained(HF_MODEL_ID, **cache_kw)
        self.model.to(device)
        self.model.eval()
        if self.half:
            self.model = self.model.half()
        logger.info("Bubble-aware detector loaded.")

    @torch.no_grad()
    def __call__(self, img, refine_mode=None, keep_undetected_mask=False):
        """
        Run detection.

        Parameters
        ----------
        img : np.ndarray  (H, W, 3)  BGR  (OpenCV convention)
        refine_mode : ignored (kept for API compat)
        keep_undetected_mask : ignored

        Returns
        -------
        mask : np.ndarray  (H, W)  uint8  binary mask (0 / 255)
        mask_refined : np.ndarray  (H, W)  uint8  same as mask
        blk_list : list[TextBlock]
        """
        im_h, im_w = img.shape[:2]

        # ---- 1. convert BGR → RGB & run processor ---------------------------
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        inputs = self.processor(images=rgb, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        if self.half:
            inputs["pixel_values"] = inputs["pixel_values"].half()

        outputs = self.model(**inputs)

        # ---- 2. post-process results ----------------------------------------
        target_size = torch.tensor([[im_h, im_w]], device=self.device)
        results = self.processor.post_process_object_detection(
            outputs, target_sizes=target_size, threshold=self.conf_thresh
        )[0]

        boxes = results["boxes"].cpu().numpy()   # (N, 4)  xyxy in original coords
        scores = results["scores"].cpu().numpy()
        labels = results["labels"].cpu().numpy()

        # Keep only text classes
        text_mask = np.isin(labels, list(TEXT_CLASS_IDS))
        boxes = boxes[text_mask]
        scores = scores[text_mask]
        labels = labels[text_mask]

        # ---- 3. build binary mask from detected boxes -----------------------
        mask = np.zeros((im_h, im_w), dtype=np.uint8)
        for box in boxes:
            x1, y1, x2, y2 = box.astype(np.int32)
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(im_w - 1, x2)
            y2 = min(im_h - 1, y2)
            if x2 > x1 and y2 > y1:
                mask[y1:y2, x1:x2] = 255

        # ---- 4. create TextBlock list (one per detected box) ----------------
        blk_list = []
        for box, label_id in zip(boxes, labels):
            x1, y1, x2, y2 = box.astype(np.int32)
            # Build a 4-corner polygon that matches the box
            poly = np.array(
                [[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.int32
            )
            lang = "eng" if label_id == 1 else "unknown"

            blk = TextBlock(
                xyxy=[x1, y1, x2, y2],
                lines=[poly.tolist()],
                language=lang,
            )
            examine_textblk(blk, im_w, im_h, sort=False)
            blk_list.append(blk)

        # Return mask twice for API compat with legacy refine_mask
        return mask, mask, blk_list