# خريطة الأخطاء والتصحيح: دليل شامل لجميع أنواع الأخطاء

## نظرة عامة

هذا المستند يغطي جميع أنواع الأخطاء التي قد تحدث في PanelCleaner، مع أسبابها المحتملة وطرق حلها. مقسم حسب المكونات.

---

## 1. أخطاء الاستيراد (Import Errors)

### `ImportError: No module named 'torch'`

| التفاصيل | القيمة |
|---|---|
| **السبب** | لم يتم تثبيت PyTorch |
| **المكان** | أي ملف يستورد torch |
| **الرسالة** | `ModuleNotFoundError: No module named 'torch'` |
| **الحل** | `pip install torch` أو تثبيت النسخة المناسبة: `pip install torch --index-url https://download.pytorch.org/whl/cu118` |

### `ImportError: cannot import name 'cuda_available' from 'pcleaner.runtime.device'`

| التفاصيل | القيمة |
|---|---|
| **السبب** | الملف `device.py` غير موجود أو قديم |
| **المكان** | `device.py` يحوي الدوال الأساسية للكشف عن GPU |
| **الحل** | تأكد من وجود الملف في `src/pcleaner/runtime/device.py` |

### خطأ استيراد PyQt6

```python
try:
    from PyQt6.QtCore import QThread, pyqtSignal
except ImportError:
    print("GUI غير متاح: من فضلك ثبّت PyQt6")
```

| **السبب** | PyQt6 غير مثبت |
|---|---|
| **الحل** | `pip install pcleaner[gui]` أو `pip install PyQt6` |

---

## 2. أخطاء CUDA / GPU

### `RuntimeError: CUDA out of memory`

| التفاصيل | القيمة |
|---|---|
| **السبب** | ذاكرة GPU غير كافية (OOM) |
| **المكان** | `inpainting.py` عند تحميل الصورة إلى GPU |
| **الرسالة** | `torch.cuda.OutOfMemoryError: CUDA out of memory` |
| **الحل** | |
| 1. | تقليل `inpainting_resolution` (مثلاً من 2048 إلى 1024) |
| 2. | تفعيل `tile_size` لتقسيم الصورة |
| 3. | استخدام `--device cpu` للرجوع إلى CPU |
| 4. | إغلاق تطبيقات أخرى تستخدم GPU |

### `AssertionError: Torch not compiled with CUDA enabled`

| التفاصيل | القيمة |
|---|---|
| **السبب** | تثبيت PyTorch لوحدة المعالجة المركزية فقط |
| **المكان** | `device.py:cuda_available()` |
| **الرسالة** | `torch.cuda.is_available()` تُرجع `False` |
| **الحل** | `pip uninstall torch; pip install torch --index-url https://download.pytorch.org/whl/cu118` |

### `NVIDIA-SMI has failed because it couldn't communicate with the NVIDIA driver`

| التفاصيل | القيمة |
|---|---|
| **السبب** | مشكلة في تعريف NVIDIA (driver) |
| **المكان** | النظام (خارج Python) |
| **الحل** | إعادة تثبيت تعريف NVIDIA driver من https://www.nvidia.com/drivers |

### تصحيح أخطاء GPU

```bash
# التحقق من توفر CUDA
python -c "import torch; print(torch.cuda.is_available())"

# التحقق من إصدار CUDA
python -c "import torch; print(torch.version.cuda)"

# التحقق من اسم بطاقة الرسوم
python -c "import torch; print(torch.cuda.get_device_name(0))"

# التحقق من الذاكرة المتوفرة
python -c "import torch; print(torch.cuda.get_device_properties(0).total_memory / 1024**3, 'GB')"

# استخدام أداة device.py المدمجة
python -c "from pcleaner.runtime.device import device_name, device_kind, cuda_available; print(device_name(), device_kind(), cuda_available())"
```

---

## 3. أخطاء OCR

### PaddleOCR: `ValueError: invalid literal for int() with base 10`

| التفاصيل | القيمة |
|---|---|
| **السبب** | إصدار غير متوافق من PaddleOCR/PaddlePaddle |
| **المكان** | `masker.py` أو `ctd_interface.py` |
| **الرسالة** | `ValueError: invalid literal for int() with base 10: '2.6'` |
| **الحل** | تثبيت الإصدار المناسب: `pip install pcleaner[ocr-extra]` |

### Manga-OCR: `FileNotFoundError`

| التفاصيل | القيمة |
|---|---|
| **السبب** | نموذج manga-ocr لم يتم تنزيله أو تلف |
| **المكان** | `model_downloader.py` |
| **الحل** | `pcleaner model download` أو `python -m pcleaner model download` |

### OCR بطيء جداً

| **السبب** | استخدام CPU بدلاً من GPU لـ OCR |
|---|---|
| **الحل** | 1. استخدام `--ocr-device cuda:0` في CLI |
| | 2. تفعيل GPU في إعدادات GUI |
| | 3. استخدام `detect: light` لتسريع OCR على حساب الدقة |

### تصحيح أخطاء OCR

```bash
# اختبار OCR فقط (بدون inpainting)
pcleaner ocr input/ --output-dir ocr_test/

# اختبار OCR مع GPU
pcleaner ocr input/ --ocr-device cuda:0

# رؤية نتائج OCR مفصلة
pcleaner clean input/ --debug
```

---

## 4. أخطاء Inpainting

### `RuntimeError: Input type (torch.FloatTensor) and weight type ...`

| التفاصيل | القيمة |
|---|---|
| **السبب** | عدم تطابق أنواع البيانات بين النموذج والمدخلات |
| **المكان** | `inpainting.py` |
| **الحل** | نموذج LaMa تالف أو إصدار PyTorch غير متوافق. أعد تحميله: `pcleaner model download --force` |

### `MemoryError: Unable to allocate ...`

| **السبب** | الصورة كبيرة جداً للذاكرة المتاحة |
|---|---|
| **المكان** | `inpainting.py:inpaint_masked()` |
| **الحل** | استخدم `tile_size=512` في الإعدادات أو قم بتقليل دقة inpainting |

### Inpainting ينتج نتائج مشوشة (blurry)

| **السبب** | دقة inpainting منخفضة جداً |
|---|---|
| **المكان** | `inpainting.py` |
| **الحل** | زيادة `inpainting_resolution` إلى 2048 أو 4096 واستخدام GPU |

---

## 5. أخطاء الخادم (Server)

### `ModuleNotFoundError: No module named 'uvicorn'`

| **السبب** | لم يتم تثبيت dependencies الخادم |
|---|---|
| **الحل** | `pip install pcleaner[server]` |

### FastAPI: `422 Unprocessable Entity`

| **السبب** | طلب API غير صحيح (حقول مفقودة أو غير صالحة) |
|---|---|
| **المكان** | `server/api.py` |
| **الحل** | تحقق من توثيق API في `server/api.py:routes` |

### `500 Internal Server Error` في `/process`

| **السبب** | خطأ في worker أثناء المعالجة |
|---|---|
| **المكان** | `server/worker.py` |
| **الحل** | 1. تحقق من سجلات الخادم (`docker logs` أو `server/logs/`) |
| | 2. تأكد من تحميل النماذج (`GET /ready`) |
| | 3. تحقق من اتصال Supabase (`server/database.py`) |

### مشكلة Supabase: `HTTPError: 401 Unauthorized`

| **السبب** | انتهاء صلاحية Supabase token أو API key غير صحيح |
|---|---|
| **المكان** | `server/database.py` |
| **الحل** | تحديث متغيرات البيئة `SUPABASE_URL` و `SUPABASE_KEY` |

### مشكلة Google Drive: `google.auth.exceptions.RefreshError`

| **السبب** | انتهاء صلاحية OAuth token |
|---|---|
| **المكان** | `server/google_drive.py` |
| **الحل** | حذف `token.pickle` وإعادة المصادقة |

### Celery worker: `ConnectionError: Connection refused`

| **السبب** | Redis/Broker غير قيد التشغيل |
|---|---|
| **المكان** | `server/celery_app.py` |
| **الحل** | `docker compose up redis` أو تثبيت وتشغيل Redis محلياً |

---

## 6. أخطاء الواجهة الرسومية (GUI)

### `qt.qpa.plugin: Could not load the Qt platform plugin "windows"`

| **السبب** | مشكلة في بيئة Qt أو عدم وجود منصة عرض |
|---|---|
| **الحل** | `pip install PyQt6` أو إعادة تثبيت PyQt6 |

### GUI يتجمد (freeze) أثناء المعالجة

| **السبب** | المعالجة تتم في thread الرئيسي بدلاً من QtWorker |
|---|---|
| **المكان** | `gui/processing.py` |
| **الحل** | تأكد من استخدام `QThread` للمعالجة في الخلفية (انظر `gui/processing.py:ProcessingThread`) |

### `AttributeError: 'ProcessingWindow' object has no attribute 'inc_progress'`

| **السبب** | خطأ موجود مسبقاً في `ctd_interface_gui.py` — دالة `inc_progress` مفقودة |
|---|---|
| **المكان** | `gui/ctd_interface_gui.py` (قديم، استخدام مفضل: `mainwindow_driver.py`) |
| **الحل** | استخدم `mainwindow_driver.py` بدلاً من `ctd_interface_gui.py` للإصدارات الجديدة |

### مؤشر التقدم لا يعمل

| **السبب** | إشارات (signals) PyQt6 غير متصلة بشكل صحيح |
|---|---|
| **المكان** | `gui/processing.py` |
| **الحل** | تحقق من اتصال إشارات `progress_updated` و `page_done` في `mainwindow_driver.py` |

---

## 7. أخطاء Docker

### `docker: command not found`

| **السبب** | Docker غير مثبت |
|---|---|
| **الحل** | ثبّت Docker Desktop من https://www.docker.com/products/docker-desktop/ |

### `nvidia-smi: command not found` في حاوية GPU

| **السبب** | NVIDIA Container Toolkit غير مثبت |
|---|---|
| **المكان** | `Dockerfile.server:gpu` |
| **الحل** | `winget install Nvidia.CUDA` وعمل Restart |

### `The container didn't respond to the health check`

| **السبب** | فحص الصحة (healthcheck) في `Dockerfile.server` فشل |
|---|---|
| **المكان** | `Dockerfile` — سطر `HEALTHCHECK` |
| **الحل** | تحقق من `docker compose ps` لمعرفة الحالة، ثم `docker compose logs` |

### بناء Docker يفشل مع `pip install` يعلق

| **السبب** | ذاكرة منخفضة أو اتصال إنترنت بطيء |
|---|---|
| **الحل** | أضف `--timeout=1000` إلى pip: `pip install --timeout=1000 ...` |

---

## 8. أخطاء Compile / Syntax

### `SyntaxError: 'return' outside function`

| **السبب** | مسافة بادئة (indentation) غير صحيحة |
|---|---|
| **المكان** | `ocr/masker.py` (حدث في الإصدارات السابقة) |
| **الحل** | تأكد من أن `return` داخل دالة وليس على مستوى الملف |

### `DeprecationWarning: datetime.utcnow() is deprecated`

| **السبب** | Python 3.12 أزال `datetime.utcnow()` |
|---|---|
| **المكان** | `server/database.py` |
| **الحل** | استخدم `datetime.now(timezone.utc)` بدلاً من `datetime.utcnow()` |

### خطأ في مطابقة الإصدارات: `pip check` يعرض تعارضات

```bash
pip check
# ابحث عن "incompatible" أو "conflict"
```

| **السبب** | تعارضات بين إصدارات المكتبات (خاصة PaddlePaddle/PyTorch/Torchvision) |
|---|---|
| **المكان** | `pyproject.toml` أو `requirements.txt` |
| **الحل** | استخدم التثبيت المستهدف: `pip install pcleaner[ocr]` بدلاً من الكل مرة واحدة |

---

## 9. أخطاء الذاكرة والأداء

### `Killed` (على Linux) أو توقف العملية فجأة

| **السبب** | OOM Killer قام بقتل العملية (نفاد ذاكرة RAM) |
|---|---|
| **الحل** | 1. استخدم `--tile-size 512` |
| | 2. قلل `--inpainting-resolution` |
| | 3. أغلق التطبيقات الأخرى |

### بطء شديد في أول تشغيل

| **السبب** | تنزيل النماذج (LaMa ~700MB + OCR ~300MB) |
|---|---|
| **الحل** | اتصال إنترنت سريع مطلوب للتشغيل الأول. بعد ذلك، النماذج تُخبأ في `~/.cache/pcleaner/models/` |

### `pagefile.sys` أو `swap` يمتلئ

| **السبب** | استخدام الذاكرة الافتراضية عند نفاد RAM |
|---|---|
| **المكان** | النظام |
| **الحل** | زيادة حجم SWAP/paging file أو إضافة RAM |

---

## 10. جدول سرعة التشخيص

| المشكلة | أسرع طريقة للتشخيص |
|---|---|
| GPU لا يعمل | `python -c "from pcleaner.runtime.device import device_name; print(device_name())"` |
| نموذج تالف | `pcleaner model download --force` |
| OCR لا يعمل | `pcleaner ocr test.png --debug` |
| خادم لا يستجيب | `curl http://localhost:8000/health` |
| ذاكرة منخفضة | `nvidia-smi` (GPU) أو Task Manager (RAM) |
| تعارض إصدارات | `pip check` |
| خطأ في الترجمة | `python -m compileall src/pcleaner/` |
| PyTorch بدون CUDA | `python -c "import torch; print(torch.cuda.is_available())"` |

---

## روابط سريعة للملفات

| الملف | الغرض |
|---|---|
| `src/pcleaner/runtime/device.py` | كشف GPU/CPU |
| `src/pcleaner/model_downloader.py` | تنزيل النماذج والتحقق منها |
| `src/pcleaner/ocr/masker.py` | OCR وتوليد الأقنعة |
| `src/pcleaner/inpainting.py` | Inpainting باستخدام LaMa |
| `server/worker.py` | معالجة الخادم الخلفية |
| `server/database.py` | اتصال Supabase |
| `server/google_drive.py` | رفع Google Drive |
| `gui/mainwindow_driver.py` | واجهة المستخدم الرسومية |
