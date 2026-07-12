"""
مُصنف الفقاعات باستخدام CLIP embeddings + nearest neighbor.

بدلاً من القواعد الثابتة (نسبة العرض للطول، الحواف، إلخ)،
يستخدم هذا المُصنف نموذج CLIP لمقارنة الفقاعات مع أمثلة مرجعية.

كيف يعمل:
1. يُحمّل أمثلة الفقاعات من bubble_examples/
2. يستخرج embedding لكل مثال باستخدام CLIP
3. عند تصنيف فقاعة جديدة: يستخرج embedding لها ويقارنه بالأمثلة
4. يُرجع أقرب نوع مع درجة الثقة
"""

from __future__ import annotations

import json
from enum import Enum
from pathlib import Path
from dataclasses import dataclass, field

import cv2
import numpy as np
from loguru import logger
from PIL import Image

# مسار مجلد الأمثلة
EXAMPLES_DIR = Path(__file__).parent / "bubble_examples"
CACHE_FILE = Path(__file__).parent / "bubble_embeddings_cache.json"


class BubbleType(Enum):
    """أنواع الفقاعات المكتشفة."""
    SPEECH = "SPEECH"
    SHOUT = "SHOUT"
    THOUGHT = "THOUGHT"
    BOX = "BOX"
    CONNECTED = "CONNECTED"
    SYSTEM = "SYSTEM"
    ON_TEXT = "OT"
    SIDE_TEXT = "ST"
    UNKNOWN = "UNKNOWN"


# خريطة الأنواع → الرموز
SYMBOL_MAP: dict[BubbleType, str] = {
    BubbleType.SPEECH: '""',
    BubbleType.SHOUT: "::",
    BubbleType.THOUGHT: "()",
    BubbleType.BOX: "[]",
    BubbleType.CONNECTED: "//",
    BubbleType.SYSTEM: "<>",
    BubbleType.ON_TEXT: "OT:",
    BubbleType.SIDE_TEXT: "ST:",
    BubbleType.UNKNOWN: "",
}


@dataclass
class BubbleClassification:
    """نتيجة تصنيف فقاعة واحدة."""
    bubble_type: BubbleType
    confidence: float
    best_match_path: str = ""  # مسار أقرب مثال مطابق

    @property
    def symbol(self) -> str:
        return SYMBOL_MAP.get(self.bubble_type, "")


class BubbleClassifier:
    """
    مُصنف الفقاعات باستخدام CLIP embeddings.

    يستخدم نموذج CLIP (Contrastive Language-Image Pre-training)
    لاستخراج embeddings من صور الفقاعات الم.reference والم未知ة.
    """

    def __init__(self, examples_dir: Path | str | None = None):
        """
        تهيئة مُصنف الفقاعات.

        :param examples_dir: مسار مجلد الأمثلة المرجعية
        """
        self.examples_dir = Path(examples_dir) if examples_dir else EXAMPLES_DIR
        self._model = None
        self._processor = None
        self._embeddings_cache: dict[str, list[float]] = {}
        self._examples_loaded = False

    def _load_model(self):
        """تحميل نموذج CLIP بشكل كسول."""
        if self._model is not None:
            return

        try:
            import torch
            from transformers import CLIPModel, CLIPProcessor

            model_id = "openai/clip-vit-base-patch32"
            logger.info(f"🔧 Loading CLIP model: {model_id}")

            self._processor = CLIPProcessor.from_pretrained(model_id)
            self._model = CLIPModel.from_pretrained(model_id)
            self._model.eval()

            # نقل النموذج إلى GPU إذا متاح
            if torch.cuda.is_available():
                self._model = self._model.cuda()
                logger.info("🎮 CLIP model loaded on GPU")
            else:
                logger.info("💻 CLIP model loaded on CPU")

        except ImportError as e:
            logger.error(f"❌ transformers/CLIP not installed: {e}")
            raise
        except Exception as e:
            logger.error(f"❌ Failed to load CLIP model: {e}")
            raise

    def _load_examples(self):
        """تحميل الأمثلة المرجعية واستخراج embeddings."""
        if self._examples_loaded:
            return

        # محاولة تحميل الكاش أولاً
        if self._load_cache():
            self._examples_loaded = True
            return

        self._load_model()
        self._load_examples_from_disk()
        self._save_cache()
        self._examples_loaded = True

    def _load_cache(self) -> bool:
        """تحميل embeddings من الكاش."""
        try:
            if not CACHE_FILE.exists():
                return False

            with open(CACHE_FILE, "r", encoding="utf-8") as f:
                self._embeddings_cache = json.load(f)

            if self._embeddings_cache:
                logger.info(f"📦 Loaded {len(self._embeddings_cache)} cached embeddings")
                return True
            return False
        except Exception as e:
            logger.warning(f"⚠️ Failed to load embeddings cache: {e}")
            return False

    def _save_cache(self):
        """حفظ embeddings في الكاش."""
        try:
            with open(CACHE_FILE, "w", encoding="utf-8") as f:
                json.dump(self._embeddings_cache, f, ensure_ascii=False)
            logger.info(f"💾 Saved {len(self._embeddings_cache)} embeddings to cache")
        except Exception as e:
            logger.warning(f"⚠️ Failed to save embeddings cache: {e}")

    def _load_examples_from_disk(self):
        """تحميل جميع أمثلة الفقاعات واستخراج embeddings."""
        import torch

        self._embeddings_cache = {}

        for bubble_type_dir in self.examples_dir.iterdir():
            if not bubble_type_dir.is_dir():
                continue

            type_name = bubble_type_dir.name.lower()
            if type_name not in [t.value.lower() for t in BubbleType]:
                # تخطي مجلدات غير معروفة
                continue

            for img_path in bubble_type_dir.glob("*.png"):
                try:
                    embedding = self._extract_embedding(img_path)
                    if embedding is not None:
                        cache_key = f"{type_name}/{img_path.name}"
                        self._embeddings_cache[cache_key] = embedding.tolist()
                        logger.debug(f"✅ Embedded: {cache_key}")
                except Exception as e:
                    logger.warning(f"⚠️ Failed to embed {img_path}: {e}")

            for img_path in bubble_type_dir.glob("*.jpg"):
                try:
                    embedding = self._extract_embedding(img_path)
                    if embedding is not None:
                        cache_key = f"{type_name}/{img_path.name}"
                        self._embeddings_cache[cache_key] = embedding.tolist()
                        logger.debug(f"✅ Embedded: {cache_key}")
                except Exception as e:
                    logger.warning(f"⚠️ Failed to embed {img_path}: {e}")

        logger.info(f"📚 Loaded {len(self._embeddings_cache)} example embeddings")

    def _extract_embedding(self, image_path: Path | str) -> np.ndarray | None:
        """استخراج embedding من صورة."""
        import torch

        self._load_model()

        try:
            img = Image.open(image_path).convert("RGB")
            inputs = self._processor(images=img, return_tensors="pt")

            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self._model.get_image_features(**inputs)

            # تطبيع الـ embedding
            embedding = outputs / outputs.norm(p=2, dim=-1, keepdim=True)
            return embedding.cpu().numpy().flatten()
        except Exception as e:
            logger.error(f"❌ Error extracting embedding from {image_path}: {e}")
            return None

    def classify_bubble(
        self,
        image_crop: Image.Image | np.ndarray,
        bubble_type_hint: str | None = None,
    ) -> BubbleClassification:
        """
        تصنيف فقاعة باستخدام CLIP embeddings.

        :param image_crop: صورة الفقاعة المقصوصة (PIL Image أو numpy array)
        :param bubble_type_hint: تلميح لنوع الفقاعة من detector (اختياري)
        :return: تصنيف الفقاعة مع درجة الثقة
        """
        # إذا كان فيه تلميح من الـ detector مباشرة
        if bubble_type_hint:
            hint_type = self._resolve_hint(bubble_type_hint)
            if hint_type:
                return BubbleClassification(
                    bubble_type=hint_type,
                    confidence=0.95,
                    best_match_path=f"hint:{bubble_type_hint}"
                )

        # تحميل الأمثلة
        self._load_examples()

        if not self._embeddings_cache:
            logger.warning("⚠️ No example embeddings available, using default SPEECH")
            return BubbleClassification(BubbleType.SPEECH, 0.5, "no_examples")

        # استخراج embedding للفقاعة المجهولة
        if isinstance(image_crop, np.ndarray):
            image_crop = Image.fromarray(cv2.cvtColor(image_crop, cv2.COLOR_BGR2RGB))

        query_embedding = self._extract_embedding(image_crop)
        if query_embedding is None:
            return BubbleClassification(BubbleType.SPEECH, 0.5, "embedding_failed")

        # حساب التشابه مع جميع الأمثلة
        best_type = BubbleType.SPEECH
        best_similarity = -1.0
        best_match_path = ""

        for cache_key, emb_list in self._embeddings_cache.items():
            ref_embedding = np.array(emb_list)
            similarity = float(np.dot(query_embedding, ref_embedding))

            if similarity > best_similarity:
                best_similarity = similarity
                best_match_path = cache_key
                # استخراج نوع الفقاعة من مسار المثال
                type_name = cache_key.split("/")[0]
                try:
                    best_type = BubbleType(type_name.upper())
                except ValueError:
                    best_type = BubbleType.SPEECH

        # تحويل التشابه إلى درجة ثقة (0-1)
        # cosine similarity range: -1 to 1, convert to 0-1
        confidence = max(0.0, min(1.0, (best_similarity + 1.0) / 2.0))

        # إذا كانت الثقة منخفضة جداً، نرجع SPEECH كافتراضي
        if confidence < 0.6:
            return BubbleClassification(BubbleType.SPEECH, 0.5, best_match_path)

        return BubbleClassification(
            bubble_type=best_type,
            confidence=confidence,
            best_match_path=best_match_path
        )

    def _resolve_hint(self, hint: str) -> BubbleType | None:
        """تحويل تلميح الـ detector إلى BubbleType."""
        hint_lower = hint.lower().strip()
        if hint_lower == "text_bubble":
            return BubbleType.SPEECH
        elif hint_lower == "text_free":
            return BubbleType.ON_TEXT
        elif hint_lower == "bubble":
            return None  # فقاعة فارغة بدون نص — نحتاج تحليل شكل
        return None

    def add_example(
        self,
        image_path: Path | str,
        bubble_type: BubbleType,
    ):
        """
        إضافة مثال جديد وتحديث الكاش.

        :param image_path: مسار صورة الفقاعة
        :param bubble_type: نوع الفقاعة
        """
        self._load_model()

        embedding = self._extract_embedding(image_path)
        if embedding is not None:
            cache_key = f"{bubble_type.value.lower()}/{Path(image_path).name}"
            self._embeddings_cache[cache_key] = embedding.tolist()
            self._save_cache()
            logger.info(f"➕ Added example: {cache_key}")
        else:
            logger.error(f"❌ Failed to add example: {image_path}")

    def clear_cache(self):
        """مسح الكاش وإعادة بنائه."""
        self._embeddings_cache = {}
        if CACHE_FILE.exists():
            CACHE_FILE.unlink()
        self._examples_loaded = False
        logger.info("🗑️ Embeddings cache cleared")


def create_bubble_classifier(
    examples_dir: Path | str | None = None,
) -> BubbleClassifier:
    """دالة مصنع لإنشاء BubbleClassifier."""
    return BubbleClassifier(examples_dir=examples_dir)
