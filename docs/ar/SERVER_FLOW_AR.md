# سير عمل الخادم (Server): البنية، المسارات، المعالجة الخلفية

## نظرة عامة

يشرح هذا المستند كيفية عمل خادم PanelCleaner API: المسارات (routes)، معالجة المهام في الخلفية (worker)، التفاعل مع Supabase و Google Drive، وتنظيف الملفات منتهية الصلاحية.

```
المستخدم → FastAPI (main.py) → Worker (worker.py) → Supabase + Google Drive
                                    ↓
                              OCR → Inpainting → رفع النتيجة
```

---

## 1. بنية مجلد الخادم

```
server/
├── __init__.py          # تصدير create_app()
├── api.py               # جميع مسارات API
├── worker.py            # معالجة الخلفية (OCR + Inpainting)
├── database.py          # التفاعل مع Supabase
├── google_drive.py      # رفع إلى Google Drive
├── cleanup.py           # تنظيف المهام منتهية الصلاحية
├── celery_app.py        # تكوين Celery (اختياري)
└── config.py            # إعدادات الخادم
```

---

## 2. نقطة الدخول: `main.py` (جذر المشروع)

**الملف**: `main.py` (في جذر المشروع، وليس داخل `pcleaner/`)

```python
from fastapi import FastAPI
from server.api import router

app = FastAPI(title="PanelCleaner API")
app.mount("/api/v1", router)
```

**نقطة البداية**:
```bash
uvicorn main:app --host 0.0.0.0 --port 8000

# أو عبر CLI:
pcleaner server --host 0.0.0.0 --port 8000
```

---

## 3. مسارات API: `server/api.py`

### `GET /health`

| الحقل | القيمة |
|---|---|
| **الغرض** | التحقق من أن الخادم قيد التشغيل |
| **الاستجابة** | `{"status": "healthy", "timestamp": "..."}` |
| **الرمز** | `api.py:~30` |
| **الاستخدام** | فحص الصحة في Docker HEALTHCHECK |

### `GET /ready`

| الحقل | القيمة |
|---|---|
| **الغرض** | التحقق من تحميل جميع النماذج |
| **الاستجابة** | `{"ready": true, "models": {"lama": true, "ocr": true}}` |
| **الرمز** | `api.py:~40` |
| **الاستخدام** | التأكد من أن الخادم جاهز لاستقبال طلبات المعالجة |

### `GET /device`

| الحقل | القيمة |
|---|---|
| **الغرض** | عرض معلومات الجهاز الحالي |
| **الاستجابة** | `{"device": "GPU", "name": "NVIDIA RTX 3090", "cuda": true, "mps": false}` |
| **الرمز** | `api.py:~50` |
| **المصدر** | يستدعي `device_kind()` و `device_name()` من `pcleaner.runtime.device` |

### `POST /process`

| الحقل | القيمة |
|---|---|
| **الغرض** | إرسال مهمة معالجة جديدة |
| **المُدخل** | `{"image_url": "...", "options": {...}}` |
| **الاستجابة** | `{"task_id": "uuid", "status": "queued"}` |
| **الرمز** | `api.py:~70` |
| **المصادقة** | مطلوب API key (عبر `Authorization: Bearer <key>`) |

**سير عمل `POST /process`**:
```
1. التحقق من API key
2. التحقق من صحة URL الصورة
3. إنشاء task_id (UUID)
4. إضافة المهمة إلى قائمة الانتظار (queue)
5. إرجاع task_id للمستخدم
6. معالجة الخلفية تبدأ فوراً (أو عبر Celery)
```

### `GET /status/{task_id}`

| الحقل | القيمة |
|---|---|
| **الغرض** | التحقق من حالة مهمة محددة |
| **الاستجابة** | `{"task_id": "...", "status": "processing|completed|failed", "progress": 45}` |
| **الرمز** | `api.py:~100` |
| **الاستخدام** | يستعلم العميل (client) عن حالة المهمة كل بضع ثوانٍ (polling) |

### `POST /settings/generate-key`

| الحقل | القيمة |
|---|---|
| **الغرض** | إنشاء API key جديد |
| **الاستجابة** | `{"api_key": "..."}` |
| **الرمز** | `api.py:~120` |
| **ملاحظة** | يتطلب صلاحيات إدارية |

---

## 4. معالج الخلفية: `server/worker.py`

**الملف**: `server/worker.py` (150+ سطر)

### `process_task(task_id, image_url, options)`

هذه هي الوظيفة الأساسية التي تقوم بمعالجة المهمة:

```
process_task(task_id, image_url, options)
  ↓
1. تنزيل الصورة من image_url
  ↓
2. التحقق من أن الصورة صالحة
  ↓
3. OCR (detect_text من masker.py)
  ↓
4. توليد القناع (generate_mask من masker.py)
  ↓
5. Inpainting (inpaint_masked من inpainting.py)
  ↓
6. حفظ النتيجة مؤقتاً
  ↓
7. رفع النتيجة إلى Supabase
  ↓
8. (اختياري) رفع إلى Google Drive
  ↓
9. تحديث حالة المهمة إلى "completed"
  ↓
10. حذف الملفات المؤقتة
```

### إدارة التزامن (Concurrency)

| الطريقة | الوصف | الملف |
|---|---|---|
| **مباشر (sync)** | معالجة مهمة واحدة في كل مرة | `worker.py` |
| **Celery** | معالجة عدة مهام بالتوازي (يتطلب Redis/RabbitMQ) | `celery_app.py` |
| **ThreadPool** | معالجة متزامنة داخل نفس العملية | `worker.py` (اختياري) |

### إدارة الفشل:

```python
try:
    process_task(task_id, image_url, options)
except OCRException as e:
    update_status(task_id, "failed", error=f"OCR فشل: {e}")
except InpaintingException as e:
    update_status(task_id, "failed", error=f"Inpainting فشل: {e}")
except Exception as e:
    update_status(task_id, "failed", error=f"خطأ غير متوقع: {e}")
```

---

## 5. قاعدة البيانات: `server/database.py`

**الملف**: `server/database.py` (80+ سطر)

### `SupabaseClient`

```python
class SupabaseClient:
    def __init__(self, url: str, key: str):
        self.client = create_client(url, key)
```

**متغيرات البيئة المطلوبة**:
- `SUPABASE_URL` — رابط مشروع Supabase
- `SUPABASE_KEY` — مفتاح API (service_role key للمهام الإدارية)

**العمليات**:
| العملية | الوظيفة | التوقيع |
|---|---|---|
| إنشاء مهمة | `create_task(task_id, status, ...)` | إدراج سجل جديد في جدول `tasks` |
| تحديث الحالة | `update_task(task_id, status, ...)` | تحديث سجل موجود |
| الحصول على مهمة | `get_task(task_id)` | استعلام حسب task_id |
| حذف مهمة | `delete_task(task_id)` | حذف بعد انتهاء الصلاحية |
| رفع ملف | `upload_file(task_id, file_bytes, filename)` | رفع إلى Supabase Storage |

**نموذج بيانات الجدول `tasks`**:
```sql
CREATE TABLE tasks (
    id UUID PRIMARY KEY,
    status TEXT DEFAULT 'queued',
    progress INTEGER DEFAULT 0,
    image_url TEXT,
    result_url TEXT,
    error TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    expires_at TIMESTAMPTZ DEFAULT NOW() + INTERVAL '24 hours'
);
```

---

## 6. Google Drive: `server/google_drive.py`

**الملف**: `server/google_drive.py` (100+ سطر)

### `GoogleDriveUploader`

```python
class GoogleDriveUploader:
    def __init__(self, credentials_path: str):
        self.service = build("drive", "v3", credentials=...)
```

**المصادقة**:
- تستخدم OAuth 2.0 (تطبيق سطح المكتب)
- تحفظ `token.pickle` محلياً لتجنب إعادة المصادقة
- إذا انتهت صلاحية token: `RefreshError` → حذف `token.pickle` وإعادة التشغيل

**الرفع**:
```python
def upload_result(task_id: str, file_path: str, folder_id: str = None) -> str:
    # رفع الملف إلى Google Drive
    # إرجاع رابط المشاركة
```

**المتغيرات البيئية**:
- `GOOGLE_DRIVE_CREDENTIALS` — مسار ملف credentials.json
- `GOOGLE_DRIVE_FOLDER_ID` — معرف المجلد المستهدف

---

## 7. التنظيف: `server/cleanup.py`

**الملف**: `server/cleanup.py` (40+ سطر)

### `cleanup_expired_tasks()`

- تعمل كـ scheduled task (cron job أو APScheduler)
- تحذف المهام التي انتهت صلاحيتها (24 ساعة)
- تحذف الملفات المرتبطة من Supabase Storage
- تحذف الملفات المؤقتة من الخادم

**جدولة التنظيف**:
```python
# في main.py
from server.cleanup import cleanup_expired_tasks
import asyncio

async def periodic_cleanup():
    while True:
        await asyncio.sleep(3600)  # كل ساعة
        cleanup_expired_tasks()
```

---

## 8. تكامل Docker مع الخادم

في `Dockerfile.server`، يتم تشغيل الخادم عبر:

```dockerfile
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

**متغيرات البيئة في Docker**:
```yaml
# docker-compose.yml
services:
  server:
    environment:
      - SUPABASE_URL=${SUPABASE_URL}
      - SUPABASE_KEY=${SUPABASE_KEY}
      - PANELCLEANER_DEVICE=cpu  # أو cuda
      - GOOGLE_DRIVE_CREDENTIALS=/app/credentials/google_drive.json
    ports:
      - "8000:8000"
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
```

---

## 9. سير عمل الطلب الكامل — مثال

```bash
# 1. العميل يرسل طلب معالجة
curl -X POST http://localhost:8000/api/v1/process \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"image_url": "https://example.com/manga/page001.png", "options": {"ocr_engine": "manga_ocr"}}'

# 2. الاستجابة: task_id
# {"task_id": "abc-123-def", "status": "queued"}

# 3. العميل يستعلم عن الحالة كل 5 ثوانٍ
curl http://localhost:8000/api/v1/status/abc-123-def

# 4. عندما يكتمل:
# {"task_id": "abc-123-def", "status": "completed", "result_url": "https://supabase.storage/page001_clean.png"}

# 5. (اختياري) العميل يحمل النتيجة من رابط Supabase
```

---

## 10. بنية قاعدة الكود للخادم

```
main.py (FastAPI app)
  └── server/api.py (router)
        ├── GET  /health          ← التحقق من الصحة
        ├── GET  /ready           ← التحقق من النماذج
        ├── GET  /device          ← معلومات الجهاز
        ├── POST /process         ← إنشاء مهمة
        │     └── worker.py       ← معالجة الخلفية
        │           ├── masker.py      ← OCR
        │           ├── inpainting.py  ← Inpainting
        │           ├── database.py    ← Supabase
        │           └── google_drive.py ← رفع
        ├── GET  /status/{id}     ← حالة المهمة
        └── POST /settings/generate-key  ← API key
              └── cleanup.py      ← تنظيف دوري
```

**ملفات رئيسية**:
- `main.py:1-30` — إنشاء تطبيق FastAPI
- `server/api.py:30-130` — جميع المسارات
- `server/worker.py:1-150` — معالج الخلفية
- `server/database.py:1-80` — Supabase
- `server/cleanup.py:1-40` — التنظيف
