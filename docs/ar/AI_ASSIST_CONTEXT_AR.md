# سياق الذكاء الاصطناعي: دليل المطور الكامل لـ PanelCleaner

## نظرة عامة

هذا المستند مخصص لمطوري الذكاء الاصطناعي (AI) أو المطورين الجدد الذين يحتاجون إلى فهم كامل لمشروع PanelCleaner. يغطي هذا المستند بنية المشروع، القرارات المعمارية الرئيسية، الملفات الأكثر أهمية، والمزالق الشائعة.

---

## 1. ما هو PanelCleaner؟

PanelCleaner هو تطبيق Python لاكتشاف النصوص في صور المانجا والكوميكس وإزالتها باستخدام تقنيات OCR و Inpainting (ملء الصور بالذكاء الاصطناعي).

**الوظيفة الأساسية**: `pcleaner clean input/ -o output/` ← يكتشف النصوص ويزيلها من جميع الصور في مجلد الإدخال.

---

## 2. بنية المشروع — المستوى العالي

```
PanelCleaner/
├── main.py                  # خادم FastAPI
├── pyproject.toml           # بناء المشروع + إدارة التبعيات (المصدر الوحيد)
├── requirements.txt         # نسخة مطابقة لـ pyproject.toml (للتوافق القديم)
│
├── pcleaner/                # المكتبة الأساسية
│   ├── __init__.py
│   ├── main.py              # CLI: argparse + توزيع الأوامر
│   ├── helpers.py           # تنسيق pipeline (process_chapter)
│   ├── inpainting.py        # LaMa model → إزالة النصوص
│   ├── model_downloader.py  # تنزيل النماذج والتحقق منها
│   ├── config.py            # تكوين المستخدم
│   │
│   ├── ocr/                 # نظام OCR
│   │   ├── __init__.py
│   │   ├── masker.py        # OCR + توليد الأقنعة
│   │   └── ctd_interface.py # واجهة PaddleOCR
│   │
│   ├── gui/                 # واجهة المستخدم الرسومية (PyQt6)
│   │   ├── mainwindow_driver.py   # النافذة الرئيسية (2429 سطر)
│   │   ├── processing.py          # معالجة الخلفية في GUI
│   │   ├── ctd_interface_gui.py   # واجهة OCR للـ GUI
│   │   ├── memory_watcher.py      # مراقب ذاكرة GPU
│   │   └── model_downloader_driver.py  # شريط تقدم التحميل
│   │
│   └── runtime/             # بيئة التشغيل
│       └── device.py        # ⭐ مصدر واحد لاكتشاف GPU/CPU
│
├── server/                  # خادم API (FastAPI)
│   ├── api.py               # مسارات API
│   ├── worker.py            # معالجة الخلفية
│   ├── database.py          # Supabase
│   ├── google_drive.py      # Google Drive رفع
│   ├── cleanup.py           # تنظيف المهام منتهية الصلاحية
│   ├── celery_app.py        # Celery (اختياري)
│   └── config.py            # إعدادات الخادم
│
├── Dockerfile               # بناء CPU
├── Dockerfile.server        # بناء CPU + GPU
├── docker-compose.yml       # تنسيق الحاويات
│
├── tests/                   # اختبارات
│   └── test_imports.py      # اختبارات الاستيراد
│
├── translations/            # ترجمات
│   └── ar/                  # ترجمة عربية (قيد التطوير)
│
├── docs/                    # التوثيق
│   └── ar/                  # التوثيق العربي (مُولد)
│
├── credentials/             # شهادات API (Supabase, Google Drive)
├── icons/                   # أيقونات التطبيق
├── ui_files/                # ملفات واجهة Qt Designer (.ui)
├── flatpak/                 # بناء Flatpak (Linux)
└── media/                   # وسائط (شعار، فيديوهات)
```

---

## 3. القرارات المعمارية الرئيسية

### 3.1 مصدر واحد لاكتشاف الجهاز

**المشكلة**: كل ملف كان يستخدم `torch.cuda.is_available()` مباشرة، مما أدى إلى 10+ نسخ من نفس المنطق.

**القرار**: إنشاء `pcleaner/runtime/device.py` يحتوي على 5 دوال فقط ولا شيء آخر:
- `cuda_available()`
- `mps_available()`
- `device_kind()`
- `device_name()`
- `torch_device_str()`

**القاعدة**: لا تكتب `torch.cuda.is_available()` في أي مكان خارج `device.py`.

### 3.2 استيراد محروس لـ PyTorch في GUI

**المشكلة**: استيراد `torch` في ملفات GUI يؤدي إلى فشل الاستيراد إذا كان PyTorch غير مثبت.

**القرار**: جميع ملفات الـ GUI تستخدم:
```python
try:
    from pcleaner.runtime.device import cuda_available
except ImportError:
    def cuda_available() -> bool:
        return False
```

### 3.3 `pyproject.toml` هو مصدر الحقيقة الوحيد

**المشكلة**: وجود `requirements.txt` منفصل يؤدي إلى تناقضات.

**القرار**:
- `pyproject.toml` يحدد جميع التبعيات مع المجموعات (extras): `core`, `gui`, `server`, `ocr`, `ocr-extra`, `paddle`, `gpu`, `inpaint`, `full`, `all`
- `requirements.txt` هو مجرد نسخة من `pyproject.toml` للتوافق مع أدوات قديمة
- تثبيت دقيق: `pip install .[gui,ocr,gpu]`

### 3.4 Python 3.12 + إزالة `datetime.utcnow()`

تم إزالة جميع استخدامات `datetime.utcnow()` واستبدالها بـ `datetime.now(timezone.utc)` لتجنب التحذيرات في Python 3.12.

### 3.5 CLI و GUI و Server — ثلاث واجهات لنفس المحرك

```
CLI (main.py:pcleaner clean) ─────────┐
                                       ├── helpers.py (pipeline)
GUI (mainwindow_driver.py) ───────────┤       │
                                       │       ├── masker.py (OCR)
Server (api.py → worker.py) ──────────┘       ├── inpainting.py
                                               └── model_downloader.py
```

كل واجهة تستخدم نفس دوال المعالجة الأساسية.

---

## 4. الملفات الأكثر أهمية (للمطور الجديد)

| الملف | لماذا هو مهم | عدد الأسطر |
|---|---|---|
| `pcleaner/main.py` | نقطة دخول CLI، فهم الأوامر الفرعية | 1119 |
| `pcleaner/runtime/device.py` | اكتشاف GPU/CPU، المصدر الوحيد | 156 |
| `pcleaner/ocr/masker.py` | OCR + توليد القناع، ثاني أهم ملف | 800+ |
| `pcleaner/inpainting.py` | نموذج LaMa وإزالة النصوص | 300+ |
| `pcleaner/model_downloader.py` | تنزيل النماذج والتحقق من SHA256 | 400+ |
| `pcleaner/helpers.py` | تنسيق pipeline كاملة | 200+ |
| `pcleaner/config.py` | هيكل إعدادات المستخدم | - |
| `gui/mainwindow_driver.py` | نافذة GUI الرئيسية | 2429 |
| `gui/processing.py` | معالجة الخلفية في GUI | 1031 |
| `server/api.py` | جميع مسارات API | 130+ |
| `server/worker.py` | معالج الخلفية للخادم | 150+ |
| `server/database.py` | التفاعل مع Supabase | 80+ |
| `pyproject.toml` | إدارة التبعيات والإصدارات | - |
| `Dockerfile.server` | بناء CPU/GPU متعدد المراحل | 50+ |

---

## 5. فهم دورة حياة الأمر `clean`

```
$ pcleaner clean input_dir -o output_dir
  ↓
main.py:launch_clean()                     ← argparse + Config
  ↓
device_kind() + torch_device_str()         ← device.py
  ↓
ModelDownloader.setup()                    ← تحميل النماذج إذا لزم الأمر
  ↓
parse_chapters(input_dir)                  ← تقسيم الإدخال إلى فصول
  ↓
لكل فصل:
  process_chapter(chapter, config)         ← helpers.py
    ├── لكل صورة:
    │   detect_text(image, masker, config) ← masker.py (OCR)
    │   generate_mask(data, masker, config)← masker.py (mask)
    │   inpaint_masked(data, config)       ← inpainting.py (LaMa)
    │   save_clean(image, output_path)     ← حفظ النتيجة
    └── save_report(chapter, data)         ← تقرير
  ↓
طباعة ملخص النتائج
```

---

## 6. نظام OCR — ما تحتاج معرفته

### محركان متاحان:

| المحرك | اللغة | الجهاز | الدقة | السرعة |
|---|---|---|---|---|
| Manga-OCR | ياباني، عربي، متعدد اللغات | GPU/CPU/MPS | عالية | متوسطة |
| PaddleOCR | 80+ لغة (صيني، عربي، إنجليزي) | GPU/CPU | جيدة | سريعة |

### الإعدادات الرئيسية:

```python
config.ocr.engine = "manga_ocr"  # أو "paddle"
config.ocr.detect = "balanced"   # "light" | "balanced" | "full"
config.ocr.detect_device = "auto"  # "cuda:0" | "cpu"
```

### بنية البيانات:

```python
@dataclass
class TextBlock:
    text: str       # النص
    x: int          # إحداثي
    y: int
    w: int          # العرض
    h: int          # الارتفاع
    confidence: float  # 0.0 - 1.0
```

---

## 7. نظام Inpainting — ما تحتاج معرفته

### النموذج: LaMa (big-lama.pt)
- حجمه: ~700 MB
- يعمل على GPU/CPU/MPS
- يدعم التقسيم إلى بلاطات (tiling) للصور الكبيرة

### إعدادات رئيسية:

```python
config.inpainting.inpainting_resolution = 2048  # 1024-4096
config.inpainting.tile_size = 1024              # تقسيم الصورة
config.inpainting.tile_overlap = 64
```

### الدوال الأساسية:

```python
inpaint_masked(data: TextData, config: Config) → PIL.Image
```

المدخلات: الصورة الأصلية + القناع → المخرجات: الصورة النظيفة

---

## 8. المزالق الشائعة (Pitfalls)

1. **GPU لا يعمل رغم وجود بطاقة رسوم**: سبب شائع هو تثبيت PyTorch CPU. تأكد من:
   ```bash
   python -c "import torch; print(torch.cuda.is_available())"  # → True
   ```

2. **خطأ `int() base 10: '2.6'`**: إصدار PaddleOCR غير متوافق. استخدم `pcleaner[ocr-extra]`.

3. **GUI لا يفتح**: PyQt6 غير مثبت. استخدم `pip install pcleaner[gui]`.

4. **OOM (نفاد الذاكرة)**: الصورة كبيرة جداً. قلل `inpainting_resolution` أو استخدم `tile_size`.

5. **الخادم لا يستجيب**: تحقق من أن الخادم يعمل (`curl localhost:8000/health`) وأن متغيرات البيئة مضبوطة.

6. **Docker GPU لا يعمل**: NVIDIA Container Toolkit غير مثبت. على Windows: `winget install Nvidia.CUDA`.

7. **دالة `inc_progress` مفقودة**: في `gui/ctd_interface_gui.py` — هذه مشكلة موجودة مسبقاً. استخدم `mainwindow_driver.py` للإصدارات الجديدة.

8. **تحذير `datetime.utcnow()`: أستخدم Python 3.12. تم إصلاحه بالفعل — استخدم `datetime.now(timezone.utc)`.

---

## 9. كيفية إضافة ميزة جديدة

### خطوات:

1. **افهم أين تنتمي الميزة**:
   - CLI: أضف `subcommand` في `pcleaner/main.py`
   - محرك OCR: أضف دالة في `pcleaner/ocr/`
   - Inpainting: عدّل `pcleaner/inpainting.py`
   - API: أضف route في `server/api.py` + منطق في `server/worker.py`
   - GUI: أضف widget في `gui/mainwindow_driver.py`

2. **اتبع القواعد**:
   - استخدم دوال `device.py` لاكتشاف الجهاز — لا `torch.cuda.is_available()` مباشرة
   - أضف `try/except ImportError` حول استيراد `torch` في ملفات GUI
   - أضف التبعية إلى مجموعة (extras) مناسبة في `pyproject.toml`
   - حدّث `requirements.txt` فقط إذا لزم الأمر (للتوافق)

3. **التحقق من التغييرات**:
   ```bash
   python -m compileall src/pcleaner/               # تحقق من عدم وجود أخطاء syntax
   pip check                                         # تحقق من تعارض الإصدارات
   python -c "from pcleaner.runtime.device import device_name; print(device_name())"  # تحقق من اكتشاف الجهاز
   ```

---

## 10. خريطة التبعيات (Dependency Map)

```
PanelCleaner
├── Core (ضروري دائماً)
│   ├── torch              # PyTorch
│   ├── Pillow             # PIL لمعالجة الصور
│   ├── numpy              # المصفوفات
│   └── requests           # HTTP requests
│
├── GUI (اختياري: extra "gui")
│   └── PyQt6              # واجهة المستخدم
│
├── Server (اختياري: extra "server")
│   ├── fastapi            # إطار عمل API
│   ├── uvicorn            # خادم ASGI
│   ├── supabase           # قاعدة البيانات
│   └── google-api-python-client  # Google Drive
│
├── OCR (اختياري: extra "ocr")
│   ├── manga-ocr          # OCR للمانجا
│   └── paddleocr          # OCR العام (extra: "paddle")
│
└── Inpainting (جزء من core)
    └── big-lama.pt (نموذج خارجي، ~700 MB)
```

---

## 11. المؤشرات المستقبلية (TODO)

- [ ] Docker build/run verification — لا يوجد Docker daemon في بيئة الاختبار
- [ ] GPU runtime verification — لا يوجد NVIDIA GPU في بيئة الاختبار
- [ ] `tests/test_imports.py` يفشل لأن `INTERNAL_MODELS_DIR.exists()` يكون `False` في clone جديد
- [ ] `gui/ctd_interface_gui.py`: دالة `inc_progress` مفقودة (خطأ موجود مسبقاً)
- [ ] اختبارات OCR و Inpainting متكاملة

---

## 12. اختصارات سريعة

```bash
# تشغيل الاختبارات
python -m compileall src/pcleaner/

# التحقق من التبعيات
pip check

# تشغيل المعالجة الكاملة
pcleaner clean input/ -o output/

# تشغيل الخادم
uvicorn main:app --reload

# تشغيل GUI
pcleaner gui

# تنزيل النماذج يدوياً
pcleaner model download

# معلومات الجهاز
python -c "from pcleaner.runtime.device import *; print(f'{device_kind()}: {device_name()}')"
```

---

## 13. المراجع

- **ملفات التوثيق العربي**: `docs/ar/*.md`
- **GitHub**: `https://github.com/anomalyco/PanelCleaner-CUDA-build-masterV`
- **Docker Hub**: `pcleaner/server:cpu`, `pcleaner/server:gpu`
- **نموذج LaMa**: `https://github.com/saic-mdal/lama`
- **Manga-OCR**: `https://github.com/kha-white/manga-ocr`
- **PaddleOCR**: `https://github.com/PaddlePaddle/PaddleOCR`
