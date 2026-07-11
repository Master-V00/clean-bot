# دورة حياة الطلب: سير العمل الكامل من CLI إلى الإخراج

## نظرة عامة

يشرح هذا المستند الرحلة الكاملة لأي أمر يتم تشغيله في PanelCleaner، بدءًا من استدعاء CLI وحتى كتابة النتيجة النهائية.

```
CLI (main.py) → dispatcher → device detection → model loader → OCR/masking → inpainting → output
```

---

## 1. نقطة الدخول: `main.py` (سطر 1–1119)

يوجد نقطة دخول CLI في `src/pcleaner/main.py`. وهي مسؤولة عن:

- **تحليل الوسائط** (`argparse`): قراءة الأمر الفرعي (subcommand) والوسائط المرافقة
- **إنشاء كائن `Config`**: من ملف التكوين أو القيم الافتراضية
- **التوزيع على المعالج المناسب**: اختيار الوظيفة الصحيحة للتنفيذ

### الأوامر الفرعية المتاحة:

| الأمر الفرعي | الوظيفة المعالجة | الوصف |
|---|---|---|
| `clean` | `launch_clean()` | سير العمل الكامل (OCR + inpainting) |
| `gui` | `launch_gui()` | تشغيل الواجهة الرسومية (PyQt6) |
| `server` | `launch_server()` | تشغيل خادم API (FastAPI + Uvicorn) |
| `model` | `launch_model()` | إدارة تحميل النماذج |
| `cache` | `launch_cache()` | إدارة ذاكرة التخزين المؤقت |
| `mask` | `launch_mask()` | إنشاء الأقنعة فقط (بدون inpainting) |
| `hash` | `launch_hash()` | حساب بصمات الملفات |
| `ocr` | `launch_ocr()` | تشغيل OCR فقط |
| `inpaint` | `launch_inpaint()` | تشغيل inpainting فقط |
| `launch` | `launch_launch()` | قائمة تفاعلية (اختيار الوضع) |

### `launch_clean()` — سير العمل الكامل

هذه هي الوظيفة الأساسية التي تشغل pipeline التنظيف بالكامل. مسارها في `main.py:620`:

```
launch_clean()
  ↓
detect_device()          ← device.py
  ↓
ModelDownloader.setup()  ← model_downloader.py
  ↓
parse_chapters()         ← تجزئة الإدخال إلى فصول
  ↓
process_chapter()        ← helpers.py
  ├── detect_text()      ← masker.py (OCR)
  ├── generate_mask()    ← masker.py (توليد القناع)
  └── inpaint_masked()   ← inpainting.py
  ↓
حفظ الصور النظيفة إلى مجلد الإخراج
```

---

## 2. اكتشاف الجهاز: `device.py` (سطر 1–156)

قبل أي عملية، يتم اكتشاف الجهاز المتاح. هذه هي الخطوة الحاسمة التي تحدد ما إذا كان سيتم استخدام GPU أو CPU.

**الملف**: `src/pcleaner/runtime/device.py`

```python
cuda_available()   → bool  # هل CUDA متاح؟
mps_available()    → bool  # هل MPS (Apple Silicon) متاح؟
device_kind()      → str   # "GPU" أو "MPS" أو "CPU"
device_name()      → str   # اسم الجهاز مثل "NVIDIA RTX 3090"
torch_device_str() → str   # "cuda" أو "mps" أو "cpu"
```

هذه الدوال تُستخدم في **كل مكان** في المشروع. لا يوجد أي كود يكتشف الجهاز مباشرة عبر `torch.cuda.is_available()` خارج هذا الملف.

---

## 3. تحميل النماذج: `model_downloader.py` (سطر 1–400+)

**الملف**: `src/pcleaner/model_downloader.py`

يقوم `ModelDownloader` بتنزيل النماذج التالية عند الحاجة:

| النموذج | الاستخدام | الحجم التقريبي |
|---|---|---|
| LaMa (big-lama.pt) | Inpainting (ملء المناطق الممسوحة) | ~700 MB |
| Manga-OCR | OCR للنصوص اليابانية/العربية | ~300 MB |
| PaddleOCR | OCR للنصوص العامة | ~100 MB |

**آلية التحميل**:
1. البحث في `INTERNAL_MODELS_DIR` أو المسار المخصص (`~/.cache/pcleaner/models/`)
2. إذا لم يوجد → تنزيل من URL محدد في `MODELS`
3. فحص التكامل باستخدام SHA256
4. التحميل إلى الجهاز المناسب (`torch_device_str`)

---

## 4. مرحلة OCR: `masker.py` (سطر 1–800+)

**الملف**: `src/pcleaner/ocr/masker.py`

### `detect_text(data, ...)` — سطر 100+
- يقوم بتقسيم النص إلى بنود (panels) وبالونات (balloons)
- يدعم اثنين من محركات OCR:
  - **manga_ocr**: أفضل للنصوص اليابانية
  - **PaddleOCR**: الأفضل للنصوص العامة والعربية
- إعدادات `detect_device` في `Config.ocr` للاختيار بين السرعة والدقة

### `generate_mask(data, masker, ...)` — سطر 300+
- يحول نتائج OCR إلى صورة قناع (mask)
- كل منطقة نص مكتشفة تُملأ باللون الأبيض على خلفية سوداء
- القناع يُستخدم لاحقًا في inpainting لتحديد المناطق المراد إزالتها

---

## 5. مرحلة Inpainting: `inpainting.py` (سطر 1–300+)

**الملف**: `src/pcleaner/inpainting.py`

### `inpaint_masked(data, config, ...)`
- تحميل نموذج LaMa إلى الجهاز المناسب
- تمرير الصورة الأصلية + القناع إلى النموذج
- النموذج يملأ المناطق البيضاء في القناع بمحتوى متناسق مع الخلفية
- حفظ النتيجة في مجلد الإخراج

**خيارات inpainting**:
- `inpainting_resolution`: دقة المعالجة (1024px، 2048px، إلخ)
- `use_cuda`: استخدام GPU إذا متاح
- `tile_size`: تجزئة الصورة لتقليل استهلاك الذاكرة

---

## 6. الإخراج: حفظ النتائج

بعد الانتهاء من inpainting:

1. **الصور النظيفة**: حفظ الصور الأصلية مع إزالة النصوص في `{output_dir}/{chapter_name}/`
2. **ملفات JSON**: حفظ بيانات التعريف (المناطق المكتشفة، الإعدادات المستخدمة)
3. **السجلات**: حفظ سجل العملية في `{output_dir}/pcleaner.log`

```
{output_dir}/
├── chapter_001/
│   ├── page_001_clean.png
│   ├── page_002_clean.png
│   └── ...
├── chapter_002/
│   └── ...
├── ocr_data.json
└── pcleaner.log
```

---

## 7. الوضع الرسومي (GUI): `mainwindow_driver.py` + `processing.py`

عند استخدام `pcleaner gui`، يتم تشغيل:

1. **نافذة PyQt6 الرئيسية**: `mainwindow_driver.py` (2429 سطر)
2. **معالج الخلفية**: `processing.py` (1031 سطر)
3. **مراقب الذاكرة**: `memory_watcher.py` (اختياري)
4. **محمل النماذج**: `model_downloader_driver.py` (شريط التقدم)

**سير عمل GUI**:
```
User selects input folder
  ↓
Config window opens (ocr_mode, model selection, device)
  ↓
Processing starts → background thread (QThread)
  ↓
Progress updates via signals (% complete, current page)
  ↓
Output folder opens when complete
```

---

## 8. خادم API: `server/` directory

**الملف**: `main.py` (خادم FastAPI)

```
POST /process          → worker.py (معالجة غير متزامنة)
GET  /status/{task_id} → التحقق من حالة المهمة
GET  /health           → التحقق من صحة الخادم
GET  /ready            → التحقق من جاهزية النماذج
GET  /device           → معلومات الجهاز الحالي
POST /settings/generate-key → API key
```

**سير عمل الخادم**:
```
طلب وارد → التحقق من API key → إنشاء task_id → تسليم إلى Celery worker
                                                      ↓
                                             worker يقوم بـ:
                                             1. تنزيل الصورة من URL
                                             2. OCR → mask → inpaint
                                             3. رفع النتيجة إلى Supabase
                                             4. تحديث حالة المهمة
```

---

## ملخص: خريطة التدفق الكاملة للطلب

```
pcleaner clean input/
  ↓
main.py → launch_clean()
  ↓
detect_device() [device.py]
  ├── cuda_available()  ← torch.cuda.is_available()
  ├── mps_available()   ← torch.backends.mps.is_available()
  └── torch_device_str() ← "cuda" | "mps" | "cpu"
  ↓
ModelDownloader.setup() [model_downloader.py]
  ├── تحميل LaMA (~700MB)
  ├── تحميل OCR model (~300MB)
  └── تحميل إلى device
  ↓
parse_chapters() [helpers.py]
  └── تجزئة المجلدات إلى فصول
  ↓
 لكل فصل:
  ├── detect_text() [masker.py]
  │   └── OCR → بنود → بالونات → نصوص
  ├── generate_mask() [masker.py]
  │   └── نصوص → صورة قناع (أبيض/أسود)
  └── inpaint_masked() [inpainting.py]
      └── صورة + قناع → LaMa → صورة نظيفة
  ↓
حفظ الإخراج إلى {output_dir}/{chapter}/
```

**الملفات الرئيسية**:
- `src/pcleaner/main.py:620` — `launch_clean()` بداية pipeline
- `src/pcleaner/runtime/device.py:1-156` — الكشف عن الجهاز
- `src/pcleaner/model_downloader.py` — تحميل النماذج
- `src/pcleaner/ocr/masker.py` — OCR + توليد القناع
- `src/pcleaner/inpainting.py` — Inpainting
- `src/pcleaner/helpers.py` — التنسيق بين المكونات
