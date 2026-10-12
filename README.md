<div align="center">

# 🧹 CLEAN BOT

### منظف صفحات المانجا بالذكاء الاصطناعي — AI Manga Panel Cleaner

[![License](https://img.shields.io/badge/License-MIT-green)](./LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-server-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Discord](https://img.shields.io/badge/Discord-bot-5865F2?logo=discord&logoColor=white)](https://discord.js.org/)
[![Docker](https://img.shields.io/badge/Docker-ready-2496ED?logo=docker&logoColor=white)](https://www.docker.com/)

**بوت ديسكورد + سيرفر تنظيف صور المانجا (إزالة النصوص وترميم الخلفية) مع نظام نقاط وتخزين سحابي.**

**Discord bot + manga panel-cleaning server (text removal & inpainting) with credits and cloud storage.**

</div>

---

<div align="center">

## 🌍 اختار اللغة — Choose Language

### [🇪🇬 العربية](#-النسخة-العربية) &nbsp;&nbsp;|&nbsp;&nbsp; [🇬🇧 English](#-english-version)

</div>

---

<a id="-النسخة-العربية"></a>

<div dir="rtl">

## 🇪🇬 النسخة العربية

### 📖 الفكرة

**Clean Bot** منظومة متكاملة لتنظيف صفحات المانجا: المستخدم يبعت الصورة لبوت الديسكورد (`/clean`)، السيرفر (FastAPI) يشيل النصوص ويرمم الخلفية بالذكاء الاصطناعي، والنتيجة ترجع خاص مع حذف تلقائي بعد 20 دقيقة. نظام نقاط واشتراكات عبر Supabase وتخزين Google Drive.

### ✨ المميزات

- 🤖 أوامر ديسكورد: `/clean` `/balance` `/profile` `/history` `/ping` + إدارة النقاط
- 🖼️ إزالة نصوص + inpainting بعدة موديلات (LaMa) وكواشف (Hi-SAM / CTD / YOLO)
- 👁️ فحص تلقائي للطلبات المكتملة كل 10 ثوانٍ وتسليم خاص
- 🗑️ تنظيف تلقائي للملفات المنتهية
- 🐳 دعم Docker (معالج + كارت شاشة)
- 📜 ملف `New Text Document.txt` فيه كل أوامر التشغيل مرتبة + `notes/` لدليل التجهيز

### 🧩 المعمارية

```
Discord (/clean) → FastAPI /api → Task Worker → Inpainting/OCR models
                        ↓
              Supabase (credits) + Google Drive (in/out)
```

### 🛠️ التقنيات

`Python 3.10+` • `FastAPI` • `PyTorch` • `OpenCV` • `Node.js + discord.js` • `Supabase` • `Redis` • `Docker`

### 🚀 التشغيل (5 خطوات)

```bash
# 1. ثبّت الحزم
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
cd discord-bot && npm i && cd ..

# 2. جهّز البيئة: انسخ .env.example إلى .env واملأ القيم
#    + حط credentials/google-service-account.json

# 3. شغّل
chmod +x start.sh stop.sh && ./start.sh

# 4. السيرفر على http://localhost:10100 والبوت يشتغل معه

# 5. النشر على السيرفر
powershell -ExecutionPolicy Bypass -File .\sync_to_server.ps1
```

التفاصيل الكاملة في `New Text Document.txt`.

### 📁 الهيكل

```
clean-bot/
├── app/                 # سيرفر FastAPI (api • workers • services)
├── discord-bot/         # بوت الديسكورد (index.js)
├── main.py              # نقطة تشغيل السيرفر
├── start.sh / stop.sh   # تشغيل وإيقاف
├── sync_to_server.ps1   # النشر على السيرفر
├── New Text Document.txt# دفتر أوامر التشغيل
├── notes/               # أدلة وشروحات
├── .env.example         # قالب المتغيرات
└── requirements.txt
```

<details>
<summary><b>🕰️ تاريخ المشروع</b></summary>

| المرحلة | التاريخ | الوصف |
|---|---|---|
| `v00` | 2026-07-11 | لقطة PanelCleaner الأصلية مع البوت والمستندات |
| `v01` | 2026-07-12 | إعداد السيرفر + pcleaner + docker + ملاحظات التشغيل |
| restructure | 2026-08-26 | إعادة هيكلة شاملة لسيرفر app مع سكربتات التشغيل والنشر |

</details>

</div>

---

<a id="-english-version"></a>

## 🇬🇧 English Version

### 📖 Overview

**Clean Bot** is a full manga panel-cleaning pipeline: users send pages to the Discord bot (`/clean`), the FastAPI server removes text and inpaints backgrounds with AI, results are delivered via DM with auto-delete after 20 minutes. Credits via Supabase, storage on Google Drive.

### ✨ Features

- 🤖 Discord commands: `/clean` `/balance` `/profile` `/history` `/ping` + points admin
- 🖼️ Text removal + inpainting (LaMa variants, Hi-SAM / CTD / YOLO detectors)
- 👁️ 10s polling for completed jobs with DM delivery
- 🗑️ Automatic expiry cleanup
- 🐳 Docker support (CPU + GPU)
- 📜 `New Text Document.txt` runbook + `notes/` guides

### 🧩 Architecture

```
Discord (/clean) → FastAPI /api → Task Worker → Inpainting/OCR models
                        ↓
              Supabase (credits) + Google Drive (in/out)
```

### 🛠️ Stack

`Python 3.10+` • `FastAPI` • `PyTorch` • `OpenCV` • `Node.js + discord.js` • `Supabase` • `Redis` • `Docker`

### 🚀 Run (5 steps)

```bash
# 1. Install
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
cd discord-bot && npm i && cd ..

# 2. Env: copy .env.example to .env and fill values
#    + place credentials/google-service-account.json

# 3. Start
chmod +x start.sh stop.sh && ./start.sh

# 4. API at http://localhost:10100, bot starts alongside

# 5. Deploy to server
powershell -ExecutionPolicy Bypass -File .\sync_to_server.ps1
```

Full details in `New Text Document.txt`.

### 📁 Structure

```
clean-bot/
├── app/                 # FastAPI server (api • workers • services)
├── discord-bot/         # Discord bot (index.js)
├── main.py              # Server entry point
├── start.sh / stop.sh   # Start & stop
├── sync_to_server.ps1   # Server deploy
├── New Text Document.txt# Runbook
├── notes/               # Guides
├── .env.example         # Env template
└── requirements.txt
```

<details>
<summary><b>🕰️ Project history</b></summary>

| Stage | Date | Description |
|---|---|---|
| `v00` | 2026-07-11 | Upstream PanelCleaner snapshot with bot and docs |
| `v01` | 2026-07-12 | Server setup with pcleaner, docker and run notes |
| restructure | 2026-08-26 | Full restructure to app server with run/deploy scripts |

</details>

---

<div align="center">

### [⬆️ back to top / العودة للأعلى](#-clean-bot)

MIT License.

</div>
