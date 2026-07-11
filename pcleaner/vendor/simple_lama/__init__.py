"""
SimpleLaMa wrapper with FP16 + torch.compile support.
Loads the big-lama TorchScript model directly on GPU.
"""
import os
import time
import torch
import numpy as np
from PIL import Image


class SimpleLama:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_path = os.environ.get("LAMA_MODEL")
        if not model_path:
            raise RuntimeError("LAMA_MODEL environment variable not set")
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"LaMa model not found: {model_path}")

        t0 = time.time()
        print(f"Loading LaMa model from {model_path} to {self.device}")

        # Load directly on target device (not CPU first)
        self.model = torch.jit.load(model_path, map_location=self.device)
        self.model.eval()

        # Force FP32 — the LaMa architecture (heavy dilated convs) frequently overflows
        # FP16 on RTX 50-series hardware, producing NaN→black blocks in the output.
        self.model = self.model.float()
        self.use_half = False

        self._warmup()

        t1 = time.time()
        print(f"LaMa ready (device={self.device}) — load took {t1-t0:.1f}s")
        self._logged_device = False

    def _warmup(self, size=256):
        """Run one dummy inference to warm up CUDA."""
        try:
            dummy_img = torch.randn(1, 3, size, size, device=self.device, dtype=torch.float)
            dummy_mask = torch.zeros(1, 1, size, size, device=self.device, dtype=torch.float)
            self.model(dummy_img, dummy_mask)
        except Exception:
            pass

    @torch.no_grad()
    def __call__(self, image: Image.Image, mask: Image.Image) -> Image.Image:
        if not self._logged_device:
            print(
                f"  GPU check: model weight device={next(self.model.parameters()).device}, "
                f"CUDA avail={torch.cuda.is_available()}"
            )
            self._logged_device = True

        img_np = np.array(image.convert("RGB"), dtype=np.float32) / 255.0
        img_t = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).to(self.device)

        mask_np = np.array(mask.convert("L"), dtype=np.float32) / 255.0
        mask_t = torch.from_numpy(mask_np).unsqueeze(0).unsqueeze(0).to(self.device)
        mask_t = (mask_t > 0.5).float()

        h, w = img_t.shape[2:]
        pad_h = (8 - h % 8) % 8
        pad_w = (8 - w % 8) % 8
        if pad_h or pad_w:
            img_t = torch.nn.functional.pad(img_t, (0, pad_w, 0, pad_h), mode="reflect")
            mask_t = torch.nn.functional.pad(mask_t, (0, pad_w, 0, pad_h), mode="reflect")

        inpainted_t = self.model(img_t, mask_t)

        if pad_h or pad_w:
            inpainted_t = inpainted_t[:, :, :h, :w]

        inpainted_np = inpainted_t.squeeze(0).permute(1, 2, 0).cpu().numpy()
        inpainted_np = np.clip(inpainted_np * 255, 0, 255).astype(np.uint8)
        return Image.fromarray(inpainted_np, mode="RGB")
