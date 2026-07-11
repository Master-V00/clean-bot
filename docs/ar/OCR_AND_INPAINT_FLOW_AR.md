# سير عمل OCR و Inpainting: من النص الخام إلى الصورة النظيفة

## نظرة عامة

هذا المستند يشرح بالتفصيل الخطوتين الأساسيتين في PanelCleaner: اكتشاف النص (OCR) وإزالة النص مع ملء الخلفية (Inpainting).

```
الصورة الأصلية
    ↓
OCR (masker.py + ctd_interface.py)
    ├── Manga-OCR (للنصوص اليابانية/العربية)
    └── PaddleOCR (للنصوص العامة)
    ↓
توليد القناع (mask)
    ↓
Inpainting (inpainting.py + LaMa model)
    ↓
الصورة النظيفة
```

---

## 1. محركات OCR

**الملف الرئيسي**: `src/pcleaner/ocr/masker.py` (800+ سطر)
**الملف المساعد**: `src/pcleaner/ocr/ctd_interface.py` (لـ PaddleOCR)

### 1.1 Manga-OCR

| الخاصية | القيمة |
|---|---|
| **النموذج** | `manga-ocr` (مبني على Transformer) |
| **الحجم** | ~300 MB |
| **اللغة** | ياباني، صيني، عربي (متعدد اللغات) |
| **الجهاز** | CPU / GPU / MPS |
| **الدقة** | عالية جداً للنصوص اليابانية |
| **السرعة** | متوسطة (1-2 ثانية لكل صورة على GPU) |

**التشغيل**:
```python
# masker.py
from manga_ocr import MangaOcr
from pcleaner.runtime.device import torch_device_str

ocr = MangaOcr(device=torch_device_str())
text = ocr(img_np)  # إرجاع النص المكتشف
```

### 1.2 PaddleOCR

| الخاصية | القيمة |
|---|---|
| **النموذج** | PaddleOCR (PP-OCRv4) |
| **الحجم** | ~100 MB |
| **اللغة** | صيني، إنجليزي، عربي (80+ لغة) |
| **الجهاز** | CPU / GPU (MPS غير مدعوم) |
| **الدقة** | جيدة جداً للنصوص العامة |
| **السرعة** | سريعة (0.5-1 ثانية لكل صورة على GPU) |

**التشغيل**:
```python
# ctd_interface.py
from paddleocr import PaddleOCR
from pcleaner.runtime.device import torch_device_str

ocr = PaddleOCR(use_angle_cls=True, lang='ar', use_gpu=cuda_available())
results = ocr.ocr(img_path)  # إرجاع المربعات + النصوص
```

### 1.3 اختيار محرك OCR

يتحكم `Config.ocr.engine` في المحرك المستخدم:

```python
# config.py
class OCRConfig:
    engine: str = "manga_ocr"  # أو "paddle"
    detect_device: str = "auto"  # أو "cuda:0" أو "cpu"
    detect: str = "balanced"  # "light" | "balanced" | "full"
```

**متى تستخدم أي محرك**:

| المحتوى | المحرك الموصى به |
|---|---|
| مانجا يابانية | Manga-OCR |
| مانجا صينية/عربية | PaddleOCR (يدعم اللغة) |
| نصوص عامة (إنجليزية) | كلا المحركين |
| دقة عالية مطلوبة | Manga-OCR |
| سرعة عالية مطلوبة | PaddleOCR |
| GPU غير متاح | PaddleOCR (أسرع على CPU) |

---

## 2. خطوات OCR التفصيلية: `masker.py`

### `detect_text(data, masker, config)` (سطر 100+)

```
أدخل: الصورة + إعدادات OCR
  ↓
1. تحويل الصورة إلى تدرج رمادي (grayscale)
  ↓
2. تجزئة الصورة إلى بنود (panels) — تحليل التخطيط
  ├── اكتشاف الحدود الرأسية/الأفقية
  └── تحديد مناطق البالونات (balloons)
  ↓
3. لكل بالون:
  ├── اقتصاص منطقة البالون
  ├── تطبيق preprocessing (تباين، سطوع)
  ├── تشغيل OCR على المنطقة
  ├── تسجيل النص + الموقع (x, y, w, h)
  └── تخزين في TextBlock
  ↓
4. دمج النصوص المتجاورة (اختياري)
  ↓
5. تجميع النتائج في بنية TextData
  ↓
أخرج: TextData (يحتوي على جميع النصوص والمواقع)
```

### `generate_mask(data, masker, config)` (سطر 300+)

```
أدخل: TextData (نتائج OCR)
  ↓
1. إنشاء صورة فارغة (نفس حجم الصورة الأصلية)
  ↓
2. لكل TextBlock:
  ├── رسم مستطيل أبيض (255) في موقع النص
  └── تطبيق توسيع (dilate) للحصول على هامش
  ↓
3. تطبيق تنعيم الحواف (Gaussian blur خفيف)
  ↓
4. حفظ القناع كصورة PNG
  ↓
أخرج: صورة قناع (أبيض = مناطق النص، أسود = الخلفية)
```

### تصور القناع:

```
الصورة الأصلية:          القناع (mask):
┌──────────────────┐    ┌──────────────────┐
│                  │    │                  │
│  مرحباً بالعالم  │    │  ██████████████  │
│                  │    │                  │
│  How are you?    │    │  ████████████    │
│                  │    │                  │
└──────────────────┘    └──────────────────┘
(نص على خلفية)         (أبيض على أسود)
```

---

## 3. Inpainting: `inpainting.py`

**الملف**: `src/pcleaner/inpainting.py` (300+ سطر)

### النموذج: LaMa (Large Mask Inpainting)

| الخاصية | القيمة |
|---|---|
| **النموذج** | `big-lama.pt` |
| **الحجم** | ~700 MB |
| **المعمارية** | Fast Fourier Convolution + ResNet |
| **الجهاز** | CPU / GPU / MPS |
| **دقة الإدخال** | قابلة للتكوين (512-4096 px) |
| **ملف النموذج** | `~/.cache/pcleaner/models/big-lama.pt` |

### `inpaint_masked(data, config)` (سطر 50+)

```
أدخل: الصورة الأصلية + القناع + الإعدادات
  ↓
1. تحميل نموذج LaMa إلى الجهاز (torch_device_str)
  ↓
2. تجهيز الصورة:
  ├── تغيير الحجم إلى inpainting_resolution
  ├── تحويل إلى Tensor
  └── نقل إلى الجهاز (GPU/CPU)
  ↓
3. تجهيز القناع:
  ├── تغيير الحجم ليطابق الصورة
  ├── عكس القناع (1 - mask) للمنطقة المراد ملؤها
  └── تحويل إلى Tensor + نقل إلى الجهاز
  ↓
4. (اختياري) تقسيم إلى بلاطات (tiles):
  ├── إذا كانت الصورة أكبر من tile_size
  ├── معالجة كل بلاطة على حدة
  └── دمج النتائج
  ↓
5. تشغيل النموذج:
  ├── forward pass (لا حاجة للـ gradients)
  └── التقاط النتيجة
  ↓
6. معالجة لاحقة:
  ├── نقل النتيجة إلى CPU
  ├── تحويل Tensor → NumPy → PIL Image
  └── تغيير الحجم إلى الأصلي
  ↓
أخرج: الصورة النظيفة (بدون نصوص)
```

### التقسيم إلى بلاطات (Tiling):

عندما تكون الصورة أكبر من `tile_size`، يتم تقسيم المعالجة:

```
┌──────────────────────┐
│        الصورة        │
│  ┌────┬────┬────┐   │
│  │ T1 │ T2 │ T3 │   │
│  ├────┼────┼────┤   │
│  │ T4 │ T5 │ T6 │   │
│  ├────┼────┼────┤   │
│  │ T7 │ T8 │ T9 │   │
│  └────┴────┴────┘   │
└──────────────────────┘
كل بلاطة تُعالج بشكل منفصل على GPU
ثم يتم دمجها مع تداخل (overlap) لتجنب الحواف
```

**إعدادات Tiling**:

```python
config.inpainting.tile_size = 1024  # حجم البلاطة بالبكسل
config.inpainting.tile_overlap = 64 # تداخل البلاطات
config.inpainting.inpainting_resolution = 2048  # دقة المعالجة
```

---

## 4. التكامل بين OCR و Inpainting: `helpers.py`

**الملف**: `src/pcleaner/helpers.py`

### `process_chapter(data, config, ...)` (سطر 200+)

هذه الوظيفة تنسق بين OCR و Inpainting لفصل كامل:

```
process_chapter(chapter_path, config)
  ↓
1. قراءة جميع صور الفصل
  ↓
2. لكل صورة:
  ├── detect_text() ← OCR ← masker.py
  ├── save_ocr_data() ← حفظ نتائج OCR إلى JSON
  ├── generate_mask() ← mask ← masker.py
  ├── save_mask() ← حفظ القناع (اختياري، للتصحيح)
  └── inpaint_masked() ← inpainting ← inpainting.py
      └── save_clean() ← حفظ الصورة النظيفة
  ↓
3. حفظ تقرير العملية
```

---

## 5. إعدادات OCR و Inpainting

### إعدادات OCR:

```ini
[ocr]
engine = manga_ocr          # manga_ocr | paddle
detect = balanced           # light | balanced | full
detect_device = auto        # auto | cuda:0 | cpu
ocr_resolution = 2048       # دقة OCR
```

### إعدادات Inpainting:

```ini
[inpainting]
inpainting_resolution = 2048  # دقة inpainting
use_cuda = true               # استخدام GPU للـ inpainting
tile_size = 1024              # حجم البلاطة
tile_overlap = 64             # تداخل البلاطات
```

### التأثير على الأداء:

| `detect` | الجودة | السرعة | الاستخدام |
|---|---|---|---|
| `light` | منخفضة | سريعة | معاينة سريعة |
| `balanced` | متوسطة | متوسطة | الاستخدام اليومي |
| `full` | عالية | بطيئة | النتائج النهائية |

| `inpainting_resolution` | الجودة | الذاكرة | السرعة |
|---|---|---|---|
| 1024 | متوسطة | قليلة | سريعة |
| 2048 | عالية | متوسطة | متوسطة |
| 4096 | ممتازة | عالية (8+ GB) | بطيئة |

---

## 6. سير عمل متعدد المراحل — مثال كامل

```bash
# الخطوة 1: OCR فقط (لرؤية النتائج)
pcleaner ocr ./manga_chapter/ --output-dir ./ocr_output/

# الخطوة 2: فحص نتائج OCR
# → ./ocr_output/chapter_001/ocr_data.json

# الخطوة 3: تشغيل inpainting مع إعدادات مخصصة
pcleaner inpaint ./manga_chapter/ \
  --ocr-engine manga_ocr \
  --inpainting-resolution 2048 \
  --tile-size 1024 \
  --device cuda:0

# أو: سير العمل الكامل مرة واحدة
pcleaner clean ./manga_chapter/ --output-dir ./clean_chapter/
```

---

## 7. بنية بيانات OCR

```python
# masker.py
@dataclass
class TextBlock:
    text: str           # النص المكتشف
    x: int              # إحداثي X
    y: int              # إحداثي Y
    w: int              # العرض
    h: int              # الارتفاع
    confidence: float   # درجة الثقة (0-1)

@dataclass
class TextData:
    blocks: list[TextBlock]  # جميع الكتل النصية
    image_size: tuple        # (width, height)
    ocr_engine: str          # المحرك المستخدم
```

---

## 8. أدوات التصحيح

```bash
# حفظ الأقنعة مع النتائج
pcleaner clean input/ --save-masks

# حفظ نتائج OCR كـ JSON
pcleaner ocr input/ --save-json

# تصحيح OCR (رؤية مربعات النص)
pcleaner ocr input/ --debug --visualize

# اختبار inpainting بسرعة
pcleaner inpaint input/single_page.png --inpainting-resolution 1024
```

---

## 9. ملخص

```
OCR Pipeline (masker.py + ctd_interface.py)
  ├── Manga-OCR: للنصوص اليابانية/العربية (GPU/CPU)
  ├── PaddleOCR: للنصوص العامة (GPU/CPU, MPS غير مدعوم)
  ├── detect_text() → TextData (المواقع + النصوص)
  └── generate_mask() → صورة قناع (أبيض/أسود)

Inpainting Pipeline (inpainting.py)
  ├── LaMa model (big-lama.pt, ~700MB)
  ├── inpaint_masked() → صورة نظيفة
  ├── يدعم Tiling للصور الكبيرة
  └── يدعم GPU/CPU/MPS

التنسيق (helpers.py)
  └── process_chapter() → يربط OCR + Inpainting لفصل كامل
```

**الملفات الرئيسية**:
- `src/pcleaner/ocr/masker.py:100-500` — OCR + mask
- `src/pcleaner/ocr/ctd_interface.py:30-150` — PaddleOCR interface
- `src/pcleaner/inpainting.py:50-250` — Inpainting
- `src/pcleaner/helpers.py:200-350` — التنسيق بين OCR و Inpainting
