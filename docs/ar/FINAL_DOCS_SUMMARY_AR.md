# فهرس التوثيق العربي الشامل — PanelCleaner

## نظرة عامة

هذا المستند هو فهرس وملخص لجميع ملفات التوثيق العربي المتوفرة في مجلد `docs/ar/`. استخدم هذا الدليل للعثور على المعلومات التي تحتاجها بسرعة.

---

## قائمة الملفات

| # | الملف | المحتوى | الصفحات المقابلة في الكود |
|---|---|---|---|
| 1 | `FULL_FILE_TREE_AR.md` | شجرة الملفات الكاملة | المشروع بأكمله |
| 2 | `PROJECT_ARCHITECTURE_AR.md` | بنية المشروع العامة والعلاقات بين المكونات | `pyproject.toml`, `pcleaner/__init__.py` |
| 3 | `STARTUP_SEQUENCE_AR.md` | تسلسل بدء التشغيل لكل وضع (CLI/GUI/Server) | `pcleaner/main.py`, `gui/mainwindow_driver.py`, `main.py` |
| 4 | `RUNTIME_FLOW_AR.md` | دورة حياة الطلب الكاملة من CLI إلى الإخراج | `pcleaner/main.py:620`, `helpers.py`, `masker.py`, `inpainting.py` |
| 5 | `ERROR_DEBUGGING_MAP_AR.md` | خريطة كاملة لجميع أنواع الأخطاء وطرق حلها | جميع الملفات (أخطاء الاستيراد، CUDA، OCR، الخادم، GUI، Docker) |
| 6 | `GPU_CPU_RUNTIME_AR.md` | آلية الكشف عن GPU/CPU والاختيار والتبديل الديناميكي | `pcleaner/runtime/device.py:1-156` |
| 7 | `SERVER_FLOW_AR.md` | بنية الخادم، المسارات، المعالجة الخلفية، Supabase، Google Drive | `server/api.py`, `server/worker.py`, `server/database.py`, `server/cleanup.py` |
| 8 | `OCR_AND_INPAINT_FLOW_AR.md` | سير عمل OCR و Inpainting بالتفصيل | `pcleaner/ocr/masker.py`, `pcleaner/ocr/ctd_interface.py`, `pcleaner/inpainting.py` |
| 9 | `DOCKER_RUNTIME_AR.md` | بنية Docker والبناء والنشر متعدد المراحل | `Dockerfile`, `Dockerfile.server`, `docker-compose.yml` |
| 10 | `AI_ASSIST_CONTEXT_AR.md` | سياق كامل لمطوري AI: بنية، قرارات، مزالق، مراجع | جميع الملفات — نظرة شاملة |

---

## خلاصة كل ملف

### 1. `FULL_FILE_TREE_AR.md`
شجرة الملفات الكاملة للمشروع مع شرح كل ملف ودوره. يغطي ~150 ملفاً عبر المجلدات: `pcleaner/`, `server/`, `gui/`, `tests/`, `docs/`, `translations/`, `credentials/`, `icons/`، إلخ.

### 2. `PROJECT_ARCHITECTURE_AR.md`
يشرح بنية المشروع على مستوى عالٍ: كيف تتفاعل المكونات (CLI ↔ helpers ↔ OCR ↔ inpainting)، نظام المجموعات (extras) في `pyproject.toml`، وخريطة التبعيات.

### 3. `STARTUP_SEQUENCE_AR.md`
يشرح ما يحدث عند تشغيل كل وضع:
- `pcleaner clean input/` → `launch_clean()` → device detection → model loading → processing
- `pcleaner gui` → `launch_gui()` → PyQt6 main window → config → processing thread
- `pcleaner server` → `launch_server()` → uvicorn → FastAPI → routes
- `pcleaner model download` → تنزيل النماذج والتحقق من SHA256

### 4. `RUNTIME_FLOW_AR.md`
يشرح دورة حياة الطلب الكاملة خطوة بخطوة:
```
CLI → dispatcher → device → model loader → OCR → mask → inpainting → output
```
مع الإشارة إلى أرقام الأسطر في كل ملف.

### 5. `ERROR_DEBUGGING_MAP_AR.md`
دليل تصحيح الأخطاء الشامل مقسم حسب النوع:
- استيراد (ImportError)
- CUDA/GPU (CUDA out of memory, Torch not compiled with CUDA)
- OCR (PaddleOCR, Manga-OCR)
- Inpainting (Input type mismatch, MemoryError)
- الخادم (422, 500, Supabase, Google Drive, Celery)
- GUI (PyQt6, تجميد، inc_progress)
- Docker (NVIDIA Container Toolkit, healthcheck)
- Syntax (indentation, utcnow)
- الذاكرة (OOM, SWAP)

### 6. `GPU_CPU_RUNTIME_AR.md`
يشرح كيفية عمل `device.py` بالكامل:
- الدوال الخمس: `cuda_available()`, `mps_available()`, `device_kind()`, `device_name()`, `torch_device_str()`
- أين تُستخدم كل دالة
- آلية الاختيار التلقائي (GPU > MPS > CPU)
- التجاوز اليدوي (`--device cpu`)
- استهلاك الذاكرة لكل نموذج
- مقارنة أداء GPU vs CPU
- الاستيراد المحروس في ملفات GUI

### 7. `SERVER_FLOW_AR.md`
يشرح خادم API بالكامل:
- المسارات: `GET /health`, `GET /ready`, `GET /device`, `POST /process`, `GET /status/{id}`, `POST /settings/generate-key`
- `worker.py`: تنزيل الصورة → OCR → mask → inpainting → رفع إلى Supabase/Google Drive
- `database.py`: Supabase (إنشاء مهمة، تحديث حالة، رفع ملف)
- `google_drive.py`: OAuth 2.0، رفع النتائج
- `cleanup.py`: تنظيف دوري كل ساعة

### 8. `OCR_AND_INPAINT_FLOW_AR.md`
يشرح بالتفصيل:
- محركات OCR: Manga-OCR (Transformer، ~300MB) و PaddleOCR (PP-OCRv4، ~100MB)
- `masker.py`: `detect_text()` — تجزئة الصورة → بالونات → OCR → TextData
- `masker.py`: `generate_mask()` — TextData → صورة قناع (أبيض/أسود)
- `inpainting.py`: LaMa model (~700MB، FFC-ResNet) → ملء المناطق البيضاء
- Tiling: تقسيم الصور الكبيرة إلى بلاطات مع تداخل
- إعدادات: `detect`, `inpainting_resolution`, `tile_size`

### 9. `DOCKER_RUNTIME_AR.md`
يشرح:
- `Dockerfile`: بناء CPU (Python 3.12-slim، تبعيات النظام، HEALTHCHECK)
- `Dockerfile.server`: بناء متعدد المراحل (base → cpu → gpu)
- `docker-compose.yml`: تنسيق الحاويات مع متغيرات البيئة
- أوامر البناء والتشغيل لكل سيناريو (CLI، خادم إنتاج، تصحيح)
- أحجام الحاويات التقريبية: base ~300MB، cpu ~2.5GB، gpu ~4.5GB
- استكشاف الأخطاء وإصلاحها

### 10. `AI_ASSIST_CONTEXT_AR.md`
دليل المطور الكامل:
- ما هو PanelCleaner
- بنية المشروع (مستوى عالٍ)
- القرارات المعمارية (مصدر واحد للجهاز، استيراد محروس، pyproject.toml كمصدر وحيد)
- الملفات الأكثر أهمية (مع عدد الأسطر)
- دورة حياة الأمر `clean`
- نظام OCR و Inpainting
- المزالق الشائعة (8+ مشكلة معروفة)
- كيفية إضافة ميزة جديدة
- خريطة التبعيات
- المؤشرات المستقبلية
- اختصارات سريعة

---

## خريطة التنقل السريع

| أريد معرفة... | اقرأ الملف |
|---|---|
| هيكل المجلدات الكامل | `FULL_FILE_TREE_AR.md` |
| كيف يعمل المشروع بشكل عام | `PROJECT_ARCHITECTURE_AR.md` |
| ماذا يحدث عند تشغيل الأمر | `STARTUP_SEQUENCE_AR.md` |
| مسار المعالجة الكامل خطوة بخطوة | `RUNTIME_FLOW_AR.md` |
| كيفية حل خطأ معين | `ERROR_DEBUGGING_MAP_AR.md` |
| كيف يعمل اكتشاف GPU/CPU | `GPU_CPU_RUNTIME_AR.md` |
| كيف يعمل خادم API | `SERVER_FLOW_AR.md` |
| كيف يعمل OCR و Inpainting | `OCR_AND_INPAINT_FLOW_AR.md` |
| كيفية بناء Docker ونشره | `DOCKER_RUNTIME_AR.md` |
| كل ما يحتاجه مطور AI جديد | `AI_ASSIST_CONTEXT_AR.md` |

---

## إحصائيات التوثيق

| المقياس | القيمة |
|---|---|
| عدد ملفات التوثيق | 10 |
| إجمالي المواضيع المغطاة | 15+ (بدء التشغيل، CLI، GUI، Server، OCR، Inpainting، GPU/CPU، Docker، الأخطاء، إلخ) |
| عدد الملفات المصدر المشار إليها | 50+ |
| عدد سطور الكود المغطاة | ~8000+ سطر |
| اللغات | العربية (النص)، الإنجليزية (أسماء الدوال والمتغيرات) |

---

## صيانة التوثيق

عند إجراء تغييرات على الكود، يجب تحديث الملفات التالية:

| نوع التغيير | الملفات التي يجب تحديثها |
|---|---|
| إضافة أمر CLI جديد | `RUNTIME_FLOW_AR.md`, `AI_ASSIST_CONTEXT_AR.md` |
| إضافة مسار API جديد | `SERVER_FLOW_AR.md` |
| تغيير اكتشاف الجهاز | `GPU_CPU_RUNTIME_AR.md` |
| تغيير OCR/Inpainting | `OCR_AND_INPAINT_FLOW_AR.md` |
| تغيير Docker | `DOCKER_RUNTIME_AR.md` |
| إضافة ملف/مجلد جديد | `FULL_FILE_TREE_AR.md` |
| إضافة خطأ جديد محتمل | `ERROR_DEBUGGING_MAP_AR.md` |
| تغيير بنية المشروع | `PROJECT_ARCHITECTURE_AR.md` |
| أي تغيير كبير | `AI_ASSIST_CONTEXT_AR.md` |

---

## جهات الاتصال والمراجع

- **المشروع**: PanelCleaner-CUDA-build-masterV
- **المسار المحلي**: `C:\Users\National\Desktop\45\PanelCleaner-CUDA-build-masterV`
- **التوثيق العربي**: `C:\Users\National\Desktop\45\PanelCleaner-CUDA-build-masterV\docs\ar\`
- **GitHub**: `https://github.com/anomalyco/PanelCleaner-CUDA-build-masterV`

---

*آخر تحديث: 16 يونيو 2026*
