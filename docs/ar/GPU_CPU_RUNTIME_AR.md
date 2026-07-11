# بيئة التشغيل: GPU و CPU — آلية الكشف والاختيار والتبديل

## نظرة عامة

يشرح هذا المستند كيفية اكتشاف PanelCleaner للأجهزة المتاحة (GPU/CPU)، وكيفية اختيار الجهاز المناسب، وكيفية التبديل بينهما ديناميكياً.

```
مبدأ أساسي: كل مسارات اكتشاف الجهاز تتمركز في ملف واحد:
→ src/pcleaner/runtime/device.py
```

---

## 1. ملف `device.py`: المصدر الوحيد للحقيقة

**الملف**: `src/pcleaner/runtime/device.py` (156 سطر)

يحتوي هذا الملف على 4 دوال رئيسية:

### `cuda_available() → bool`
```python
def cuda_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False
```

| القيمة | المعنى |
|---|---|
| `True` | NVIDIA GPU + CUDA toolkit + PyTorch CUDA متاحون |
| `False` | GPU غير متاح (أو PyTorch مثبت بدون CUDA) |

**الأسباب الشائعة لـ `False`**:
- PyTorch مثبت بإصدار CPU فقط (`pip install torch` بدون CUDA)
- NVIDIA driver غير مثبت
- CUDA toolkit غير مثبت
- لا يوجد GPU متوافق

### `mps_available() → bool`
```python
def mps_available() -> bool:
    try:
        import torch
        return torch.backends.mps.is_available()
    except (ImportError, AttributeError):
        return False
```

| القيمة | المعنى |
|---|---|
| `True` | Apple Silicon (M1/M2/M3) + PyTorch MPS متاح |
| `False` | غير متاح (نظام Windows/Linux أو PyTorch قديم) |

### `device_kind() → str`
```python
def device_kind() -> str:
    if cuda_available():
        return "GPU"
    elif mps_available():
        return "MPS"
    return "CPU"
```

القيم الممكنة: `"GPU"`, `"MPS"`, `"CPU"`

### `device_name() → str`
```python
def device_name() -> str:
    if cuda_available():
        import torch
        return torch.cuda.get_device_name(0)
    elif mps_available():
        return "Apple Silicon (MPS)"
    return "CPU"
```

القيم الممكنة:
- `"NVIDIA GeForce RTX 3090"` (مثال)
- `"Apple Silicon (MPS)"`
- `"CPU"`

### `torch_device_str() → str`
```python
def torch_device_str() -> str:
    if cuda_available():
        return "cuda"
    elif mps_available():
        return "mps"
    return "cpu"
```

هذه الدالة تُستخدم مباشرة في:
- `inpainting.py` — لتحميل نموذج LaMa
- `ocr/masker.py` — لتحميل نموذج OCR
- `model_downloader.py` — لاختيار مسار النموذج

---

## 2. أين يُستخدم اكتشاف الجهاز؟

| الملف | السطر | الاستخدام |
|---|---|---|
| `device.py` | الكل | المصدر الوحيد للدوال |
| `inpainting.py` | ~50 | `torch_device_str()` لتحميل النموذج |
| `ocr/masker.py` | ~10 | `torch_device_str()` لتحميل OCR |
| `ocr/ctd_interface.py` | ~30 | `torch_device_str()` لـ PaddleOCR |
| `model_downloader.py` | ~200 | `cuda_available()` لاختيار النموذج |
| `gui/processing.py` | ~50 | `cuda_available()` (مع try/except) |
| `gui/mainwindow_driver.py` | ~30 | `cuda_available()` (مع try/except) |
| `gui/ctd_interface_gui.py` | ~15 | `cuda_available()` (مع try/except) |
| `gui/memory_watcher.py` | ~10 | `cuda_available()` (مع try/except) |
| `gui/model_downloader_driver.py` | ~10 | `cuda_available()` (مع try/except) |
| `main.py` (خادم API) | ~20 | `device_kind()` لنقطة `/device` |

---

## 3. آلية الاختيار التلقائي للجهاز

عند بدء أي عملية (CLI أو GUI):

```
1. استدعاء device_kind()
2. إذا GPU متاح → استخدم CUDA لجميع النماذج
3. إذا MPS متاح → استخدم MPS (فقط Apple Silicon)
4. وإلا → استخدم CPU
```

### أولوية الجهاز:
```
GPU (CUDA) > MPS (Apple Silicon) > CPU
```

### تجاوز الاختيار التلقائي:

**CLI**:
```bash
# استخدام CPU رغماً عن وجود GPU
pcleaner clean input/ --device cpu

# استخدام GPU محدد
pcleaner clean input/ --device cuda:0

# استخدام MPS (Apple Silicon)
pcleaner clean input/ --device mps
```

**GUI**:
- في نافذة الإعدادات: قائمة منسدلة لاختيار الجهاز
- إعدادات متقدمة: `ocr_device` منفصل عن `inpaint_device`

**الخادم (Server)**:
- يستخدم الجهاز المتاح تلقائياً
- يمكن تعيينه عبر متغير البيئة `PANELCLEANER_DEVICE=cpu`

---

## 4. توزيع النماذج على الأجهزة

| النموذج | GPU | CPU | MPS |
|---|---|---|---|
| LaMa (inpainting) | ✅ ممتاز | ⚠️ بطيء جداً | ✅ جيد |
| Manga-OCR | ✅ جيد | ✅ متوسط | ✅ جيد |
| PaddleOCR | ✅ جيد | ✅ متوسط | ❌ غير مدعوم |

### استهلاك الذاكرة:

| النموذج | GPU RAM (~) | CPU RAM (~) |
|---|---|---|
| LaMa (2048px) | 2-4 GB | 4-8 GB |
| LaMa (1024px) | 1-2 GB | 2-4 GB |
| Manga-OCR | ~500 MB | ~1 GB |
| PaddleOCR | ~1 GB | ~2 GB |

---

## 5. التبديل الديناميكي أثناء التشغيل

يدعم PanelCleaner تبديل الجهاز لكل خطوة على حدة:

```python
# إعدادات OCR
config.ocr.detect_device = "cuda:0"    # GPU لـ OCR
config.ocr.detect_device = "cpu"        # CPU لـ OCR

# إعدادات Inpainting
config.inpainting.device = "cuda:0"     # GPU لـ inpainting
config.inpainting.device = "cpu"         # CPU لـ inpainting
```

هذا مفيد عندما:
- GPU له ذاكرة محدودة → OCR على GPU، inpainting على CPU
- OCR engine لا يدعم GPU → force CPU لـ OCR
- تريد توفير GPU للتطبيقات الأخرى → CPU فقط

---

## 6. حالة `torch` غير مثبت

جميع ملفات GUI تستخدم استيراداً محروساً لـ `torch`:

```python
# gui/processing.py
try:
    from pcleaner.runtime.device import cuda_available
except ImportError:
    def cuda_available() -> bool:
        return False
```

هذا يضمن أن GUI يمكنه العمل حتى بدون PyTorK (لأغراض تصدير الإعدادات أو إدارة الملفات فقط).

---

## 7. التحقق من الجهاز — أوامر سريعة

```bash
# معلومات كاملة عن الجهاز
python -c "from pcleaner.runtime.device import *; print(f'CUDA: {cuda_available()}, MPS: {mps_available()}, KIND: {device_kind()}, NAME: {device_name()}, TORCH_DEVICE: {torch_device_str()}')"

# نموذج إخراج مع GPU
# CUDA: True, MPS: False, KIND: GPU, NAME: NVIDIA GeForce RTX 3090, TORCH_DEVICE: cuda

# نموذج إخراج بدون GPU
# CUDA: False, MPS: False, KIND: CPU, NAME: CPU, TORCH_DEVICE: cpu
```

---

## 8. المقارنة: GPU vs CPU — الأداء

| العملية | GPU (RTX 3090) | CPU (i9-13900K) | السرعة النسبية |
|---|---|---|---|
| OCR (صفحة واحدة) | 0.5-1 ثانية | 2-5 ثوانٍ | 4-5x أسرع |
| Inpainting (2048px) | 2-4 ثوانٍ | 30-60 ثانية | 10-15x أسرع |
| Inpainting (1024px) | 0.5-1 ثانية | 5-15 ثانية | 8-10x أسرع |
| Manga-OCR (صفحة) | 1-2 ثانية | 3-8 ثوانٍ | 3-4x أسرع |

---

## 9. ملخص

```
device.py (المصدر الوحيد)
  ├── cuda_available()    → bool
  ├── mps_available()     → bool
  ├── device_kind()       → "GPU" | "MPS" | "CPU"
  ├── device_name()       → str (اسم الجهاز)
  └── torch_device_str()  → "cuda" | "mps" | "cpu"
         ↓
    تُستخدم في:
      ├── inpainting.py (LaMa model)
      ├── ocr/masker.py (OCR models)
      ├── ocr/ctd_interface.py (PaddleOCR)
      ├── model_downloader.py (model selection)
      └── gui/*.py (مع try/except ImportError)
```

**قاعدة مهمة**: لا تستخدم `torch.cuda.is_available()` مباشرة في أي مكان خارج `device.py`. استخدم دوال `device.py` حصراً.
