# Docker: البنية، البناء، والنشر

## نظرة عامة

يشرح هذا المستند بنية Docker الخاصة بـ PanelCleaner، وكيفية بناء الحاويات لكل من CPU و GPU، والنشر باستخدام docker-compose.

```
المشروع
├── Dockerfile              # بناء CPU (للـ CLI + GUI)
├── Dockerfile.server       # بناء متعدد المراحل (CPU + GPU)
├── docker-compose.yml      # تنسيق الحاويات
├── .dockerignore           # استثناءات
└── credentials/            # الشهادات (Supabase, Google Drive)
```

---

## 1. `Dockerfile` — حاوية CPU الأساسية

**الملف**: `Dockerfile` (40+ سطر)

### خطوات البناء:

```dockerfile
# 1. القاعدة: Python 3.12 slim
FROM python:3.12-slim AS base

# 2. تثبيت النظام
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# 3. نسخ التبعيات
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. نسخ الكود
COPY . /app
WORKDIR /app

# 5. تثبيت المشروع
RUN pip install --no-cache-dir -e .

# 6. نقطة الدخول
ENTRYPOINT ["pcleaner"]
CMD ["--help"]

# 7. فحص الصحة
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python -c "from pcleaner.runtime.device import device_kind; exit(0)"

# 8. المنفذ
EXPOSE 8000
```

### مثال الاستخدام:

```bash
# بناء الحاوية
docker build -t pcleaner:latest .

# تشغيل أمر clean
docker run --rm -v D:/manga:/data pcleaner:latest clean /data/input -o /data/output

# تشغيل الخادم
docker run --rm -p 8000:8000 pcleaner:latest server --host 0.0.0.0 --port 8000

# الدخول التفاعلي
docker run --rm -it --entrypoint bash pcleaner:latest
```

| الوسيط | الشرح |
|---|---|
| `--rm` | حذف الحاوية بعد الانتهاء |
| `-v D:/manga:/data` | ربط مجلد محلي بمجلد الحاوية |
| `-p 8000:8000` | تعريض المنفذ 8000 |
| `--entrypoint bash` | استبدال نقطة الدخول بـ bash |

---

## 2. `Dockerfile.server` — بناء متعدد المراحل (CPU + GPU)

**الملف**: `Dockerfile.server` (50+ سطر)

### المراحل (Stages):

```
Dockerfile.server
  ├── base (مرحلة القاعدة)
  │   ├── Python 3.12-slim
  │   └── تثبيت تبعيات النظام
  │
  ├── cpu (مرحلة CPU)
  │   ├── FROM base
  │   ├── pip install torch (CPU version)
  │   └── CMD uvicorn main:app
  │
  └── gpu (مرحلة GPU)
      ├── FROM base
      ├── pip uninstall torch (CPU)
      ├── pip install torch (CUDA 11.8)
      └── CMD uvicorn main:app
```

### خطوات بناء GPU:

```dockerfile
FROM base AS gpu
# إزالة PyTorch الخاص بـ CPU
RUN pip uninstall -y torch torchvision torchaudio
# تثبيت PyTorch مع CUDA 11.8
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
# تثبيت بقية dependencies
RUN pip install --no-cache-dir -e ".[server,gpu]"
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### البناء والاستخدام:

```bash
# بناء حاوية CPU
docker build -f Dockerfile.server --target=cpu -t pcleaner-server:cpu .

# بناء حاوية GPU
docker build -f Dockerfile.server --target=gpu -t pcleaner-server:gpu .

# تشغيل حاوية GPU (يتطلب NVIDIA Container Toolkit)
docker run --rm --gpus all \
  -p 8000:8000 \
  -e SUPABASE_URL=... \
  -e SUPABASE_KEY=... \
  pcleaner-server:gpu
```

**متطلبات GPU في Docker**:

1. تثبيت NVIDIA Container Toolkit:
   ```bash
   # Windows: تثبيت Docker Desktop مع دعم WSL2
   # ثم تثبيت CUDA على Windows
   winget install Nvidia.CUDA
   ```

2. تشغيل Docker مع `--gpus all`:
   ```bash
   docker run --gpus all --rm nvidia/cuda:12.0-base nvidia-smi
   ```

---

## 3. `docker-compose.yml` — تنسيق الحاويات

**الملف**: `docker-compose.yml` (30+ سطر)

```yaml
services:
  server:
    build:
      context: .
      dockerfile: Dockerfile.server
      target: ${TARGET:-cpu}  # cpu أو gpu
    ports:
      - "8000:8000"
    environment:
      - SUPABASE_URL=${SUPABASE_URL}
      - SUPABASE_KEY=${SUPABASE_KEY}
      - PANELCLEANER_DEVICE=${DEVICE:-cpu}
      - GOOGLE_DRIVE_CREDENTIALS=/app/credentials/google_drive.json
    volumes:
      - ./credentials:/app/credentials:ro
      - model_cache:/root/.cache/pcleaner/models
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s

volumes:
  model_cache:
```

### أوامر docker-compose:

```bash
# بناء وتشغيل مع CPU
$env:TARGET="cpu"
docker compose up --build -d

# بناء وتشغيل مع GPU
$env:TARGET="gpu"
$env:DEVICE="cuda"
docker compose up --build -d

# إيقاف
docker compose down

# عرض السجلات
docker compose logs -f

# التحقق من الحالة
docker compose ps
```

---

## 4. متغيرات البيئة (Environment Variables)

| المتغير | الوصف | إلزامي | مثال |
|---|---|---|---|
| `SUPABASE_URL` | رابط Supabase المشروع | للخادم فقط | `https://YOUR_PROJECT.supabase.co` |
| `SUPABASE_KEY` | مفتاح Supabase API | للخادم فقط | `eyJ...` |
| `PANELCLEANER_DEVICE` | جهاز المعالجة | لا | `cpu` / `cuda` |
| `GOOGLE_DRIVE_CREDENTIALS` | مسار شهادة Google Drive | لا | `/app/credentials/drive.json` |
| `TARGET` | مرحلة البناء (cpu/gpu) | لا | `cpu` |
| `DEVICE` | جهاز Docker Compose | لا | `cpu` / `cuda` |

---

## 5. `credentials/` — الشهادات

```
credentials/
├── google_drive.json     # شهادة Google Drive OAuth
├── supabase.json         # إعدادات Supabase (اختياري)
└── README.md             # تعليمات إعداد الشهادات
```

**ملاحظة**: لا تقم بمشاركة أو رفع ملفات الشهادات إلى git. تم إضافتها إلى `.gitignore`.

---

## 6. `.dockerignore`

```
__pycache__/
*.pyc
.git/
.gitignore
*.md
tests/
temp_chapters/
logs/
credentials/*.json  # استثناء: لا تنسخ الشهادات الفعلية
!credentials/README.md
```

---

## 7. أحجام الحاويات التقريبية

| المرحلة | الحجم التقريبي | المحتوى |
|---|---|---|
| `base` | ~300 MB | Python 3.12-slim + تبعيات النظام |
| `cpu` | ~2.5 GB | + PyTorch CPU + نموذج LaMa |
| `gpu` | ~4.5 GB | + PyTorch CUDA + نموذج LaMa |

---

## 8. سير عمل Docker النموذجي

### سيناريو 1: معالجة محلية عبر CLI

```bash
# 1. بناء الحاوية
docker build -t pcleaner:latest .

# 2. تشغيل المعالجة
docker run --rm -v D:/manga/chapter1:/input -v D:/manga/output:/output pcleaner:latest clean /input -o /output

# 3. النتائج في D:/manga/output/
```

### سيناريو 2: خادم إنتاج

```powershell
# 1. إعداد المتغيرات
$env:SUPABASE_URL = "https://YOUR_PROJECT.supabase.co"
$env:SUPABASE_KEY = "eyJ..."
$env:TARGET = "gpu"
$env:DEVICE = "cuda"

# 2. بناء وتشغيل
docker compose up --build -d

# 3. التحقق
curl http://localhost:8000/health
# → {"status": "healthy"}

curl http://localhost:8000/ready
# → {"ready": true, "models": {"lama": true, "ocr": true}}

# 4. إرسال طلب معالجة
curl -X POST http://localhost:8000/api/v1/process \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"image_url": "https://example.com/page.png"}'
```

### سيناريو 3: تصحيح الأخطاء في الحاوية

```bash
# تشغيل bash داخل الحاوية
docker run --rm -it --entrypoint bash pcleaner:latest

# داخل الحاوية:
pcleaner --help
python -c "from pcleaner.runtime.device import device_name; print(device_name())"
pip check
python -m compileall src/pcleaner/
```

---

## 9. استكشاف أخطاء Docker وإصلاحها

| المشكلة | السبب | الحل |
|---|---|---|
| `nvidia-smi` لا يعمل في الحاوية | NVIDIA Container Toolkit غير مثبت | `winget install Nvidia.CUDA` |
| بناء GPU يفشل مع `pip install torch` | انتهاء المهلة (timeout) | أضف `--timeout=1000` |
| الحاوية لا تستجيب على `:8000` | المنفذ غير مكشوف أو الخادم لم يبدأ | تحقق من `docker compose ps` |
| `curl: command not found` في HEALTHCHECK | curl غير مثبت في الصورة الأساسية | استخدم `python -c "..."` بدلاً من curl |
| `Cache directory model_cache` يمتلئ | مساحة القرص غير كافية | نظف `docker system prune` أو احذف `model_cache` يدوياً |
| `Permission denied` على مجلد credentials | صلاحيات الملفات | تأكد من أن ملفات credentials قابلة للقراءة (`chmod 644`) |

---

## 10. ملخص

```
Dockerfile (CPU CLI)
  └── python:3.12-slim → pip install → pcleaner → ENTRYPOINT pcleaner

Dockerfile.server (متعدد المراحل)
  ├── base ← python:3.12-slim + system deps
  ├── cpu ← pip install torch (CPU) → CMD uvicorn
  └── gpu ← pip uninstall torch; pip install torch (CUDA) → CMD uvicorn

docker-compose.yml
  └── server service
        ├── build: Dockerfile.server [cpu|gpu]
        ├── ports: 8000
        ├── env: SUPABASE_URL, SUPABASE_KEY, DEVICE
        ├── volumes: credentials, model_cache
        └── healthcheck: /health endpoint
```

**الملفات الرئيسية**:
- `Dockerfile:1-40` — بناء CPU
- `Dockerfile.server:1-50` — بناء متعدد المراحل (CPU/GPU)
- `docker-compose.yml:1-35` — تنسيق الحاويات
