#!/bin/bash

# Start script for Clean Bot (PanelCleaner Server)
# Smart CUDA detection + auto setup + interactive menu

set -e

# ── Progress / Spinner helpers ──
# Spinner shows while a background job runs (so you know it's not frozen)
_spinner_pid=""
show_spinner() {
    local pid=$1; local msg="$2"
    local spin='⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏'; local i=0
    while kill -0 "$pid" 2>/dev/null; do
        printf "\r  [%s] %s..." "${spin:i++%${#spin}:1}" "$msg"
        sleep 0.12
    done
    printf "\r  ✓ %s                    \n" "$msg"
}
run_with_spinner() {
    local msg="$1"; shift
    "$@" &
    local pid=$!
    show_spinner "$pid" "$msg"
    wait "$pid"
    return $?
}
# Simple progress bar for pip (pip has its own bar, we just ensure it's visible)
pip_install() {
    echo "  → pip install $*"
    pip install --progress-bar on "$@" 2>&1 | while IFS= read -r line; do echo "    $line"; done
    return ${PIPESTATUS[0]}
}

echo "========================================="
echo "   Clean Bot - Setup & Launch"
echo "========================================="

# Check prerequisites
for cmd in python3 node npm; do
    if ! command -v $cmd &> /dev/null; then
        echo "Error: $cmd is not installed"
        exit 1
    fi
done

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "[1/9] Creating Python virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# ============================================
# Smart CUDA detection + PyTorch install
# ============================================
echo "[2/9] Detecting GPU and CUDA version..."

detect_cuda_version() {
    if command -v nvidia-smi &> /dev/null; then
        CUDA_VER=$(nvidia-smi | grep -oP 'CUDA Version:\s+\K[0-9]+\.[0-9]+' | head -1)
        if [ -n "$CUDA_VER" ]; then echo "$CUDA_VER"; return; fi
    fi
    if command -v nvcc &> /dev/null; then
        CUDA_VER=$(nvcc --version | grep -oP 'release \K[0-9]+\.[0-9]+')
        if [ -n "$CUDA_VER" ]; then echo "$CUDA_VER"; return; fi
    fi
    echo ""
}

detect_gpu_name() {
    if command -v nvidia-smi &> /dev/null; then
        nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1
    fi
}

GPU_NAME=$(detect_gpu_name)
CUDA_VERSION=$(detect_cuda_version)

echo "  GPU: ${GPU_NAME:-Unknown}"
echo "  CUDA: ${CUDA_VERSION:-Unknown}"

get_torch_index_url() {
    local cuda_major=$(echo "$1" | cut -d. -f1)
    local cuda_minor=$(echo "$1" | cut -d. -f2)

    if [ -z "$1" ]; then echo "https://download.pytorch.org/whl/cpu"; return; fi

    if [ "$cuda_major" -ge 13 ]; then
        echo "https://download.pytorch.org/whl/cu126"
    elif [ "$cuda_major" -ge 12 ]; then
        echo "https://download.pytorch.org/whl/cu126"
    elif [ "$cuda_major" -eq 11 ] && [ "$cuda_minor" -ge 8 ]; then
        echo "https://download.pytorch.org/whl/cu118"
    else
        echo "https://download.pytorch.org/whl/cu118"
    fi
}

TORCH_INDEX_URL=$(get_torch_index_url "$CUDA_VERSION")
echo "  PyTorch index: $TORCH_INDEX_URL"

CURRENT_TORCH=$(python -c "import torch; print(f'{torch.__version__}|{torch.cuda.is_available()}')" 2>/dev/null || echo "")
DESIRED_TORCH="2.7.1"
DESIRED_TV="0.22.1"

if [[ "$CURRENT_TORCH" == "${DESIRED_TORCH}+"* ]] && [[ "$CURRENT_TORCH" == *"|True" ]]; then
    echo "  PyTorch $DESIRED_TORCH with CUDA already installed, skipping..."
else
    echo "[3/9] Installing PyTorch $DESIRED_TORCH + torchvision $DESIRED_TV..."
    echo "  Index: $TORCH_INDEX_URL"
    echo "  This may take several minutes (large download) - progress below:"
    pip install --progress-bar on --force-reinstall \
        torch==${DESIRED_TORCH} \
        torchvision==${DESIRED_TV} \
        --index-url "$TORCH_INDEX_URL"
    echo "  → Verifying..."
    VERIFY=$(python -c "import torch; print(f'torch={torch.__version__}, cuda={torch.version.cuda}, available={torch.cuda.is_available()}')" 2>&1)
    echo "  Verified: $VERIFY"
fi

# Install Python dependencies (with live progress bar)
echo "[4/9] Installing Python dependencies..."
echo "  Reading requirements.txt ($(wc -l < requirements.txt) packages)..."
pip install --progress-bar on -r requirements.txt
echo "  ✓ Python dependencies done"

# Install Node.js dependencies (with npm progress)
echo "[5/9] Installing Node.js dependencies..."
cd discord-bot
# npm --progress shows bar; --loglevel info gives visibility without being too noisy
npm install --progress 2>&1 | while IFS= read -r line; do echo "    $line"; done
NPM_EC=${PIPESTATUS[0]}
if [ $NPM_EC -ne 0 ]; then echo "  ✗ npm install failed (exit $NPM_EC)"; exit $NPM_EC; fi
echo "  ✓ Node.js dependencies done"
cd ..

# Create necessary directories
mkdir -p temp_chapters logs app/profiles app/models/inpainting app/models/text_detector

# Redirect torch.hub downloads to project models dir (not system cache)
export TORCH_HOME="$(pwd)/app/models"

# ============================================
# Clone Hi-SAM if not present
# ============================================
if [ ! -d "app/services/hi_sam_src/hi_sam" ]; then
    echo "[5b/9] Cloning Hi-SAM text detection (this shows progress, not hanging)..."
    git clone --depth 1 --progress https://github.com/ymy-k/Hi-SAM.git app/services/hi_sam_src 2>&1 | while IFS= read -r line; do echo "    $line"; done
    echo "  ✓ Hi-SAM cloned"
else
    echo "  ✓ Hi-SAM already present"
fi

# Install Hi-SAM dependencies (scikit-image, etc.)
echo "  Installing Hi-SAM Python deps (scikit-image, pyclipper, shapely)..."
pip install --progress-bar on scikit-image pyclipper shapely
echo "  ✓ Hi-SAM deps done"

# Download text detector checkpoints — respects master.conf choice (hi-sam vit_h/vit_b, ctd, yolo11s)
echo "[5c/9] Checking text detector checkpoints (respects master.conf choice)..."
python3 << 'PYEOF'
from pathlib import Path
import app.config as cfg
try:
    tmp = cfg.load_config()
    master = Path('app/profiles/master.conf')
    if master.is_file():
        tmp.load_profile(str(master))
    detector = getattr(tmp.current_profile.text_detector, "text_detector_model", "hi-sam")
    vit = getattr(tmp.current_profile.text_detector, "hi_sam_model_type", "vit_h")
except Exception as e:
    print(f"  Warning: could not read master.conf: {e}")
    detector = "hi-sam"
    vit = "vit_h"
# Normalize
detector = str(detector).strip().lower()
print(f"  → Selected detector: {detector} (vit={vit})")
try:
    if detector in ("hi-sam", "hi_sam", "hisam"):
        from app.services.model_downloader import ensure_hi_sam_available
        print(f"  → Ensuring Hi-SAM checkpoints (vit={vit})...")
        ensure_hi_sam_available(vit)
        print(f"  ✓ Hi-SAM checkpoints ready (vit={vit})")
    elif detector in ("ctd", "comictextdetector", "comic_text_detector"):
        from app.services.model_downloader import ensure_ctd_available
        print("  → Ensuring ComicTextDetector (CTD) checkpoint...")
        ensure_ctd_available()
        print("  ✓ CTD checkpoint ready")
    elif detector in ("yolo11s", "yolo", "yolo11"):
        from app.services.model_downloader import ensure_yolo11s_available
        print("  → Ensuring YOLO11s (lordtrilink v0) checkpoint...")
        ensure_yolo11s_available()
        print("  ✓ YOLO11s checkpoint ready")
    elif detector == "bubble":
        from app.services.model_downloader import ensure_bubble_detector_available, get_text_detector_model_dir
        print("  → Bubble detector (legacy) — no download needed")
        ensure_bubble_detector_available(get_text_detector_model_dir())
        print("  ✓ Bubble detector ready")
    else:
        print(f"  ! Unknown detector '{detector}', falling back to Hi-SAM vit={vit}")
        from app.services.model_downloader import ensure_hi_sam_available
        ensure_hi_sam_available(vit)
        print(f"  ✓ Hi-SAM checkpoints ready (vit={vit})")
except Exception as e:
    print(f"  ✗ Failed to ensure detector '{detector}': {e}")
    import traceback; traceback.print_exc()
PYEOF

echo ""
echo "========================================="
echo "   Setup Menu"
echo "========================================="

show_menu() {
    echo ""
    echo "  [1] Validate credentials (test Supabase, Google Drive, Discord)"
    echo "  [2] Select inpainting model"
    echo "  [3] Select text detector"
    echo "  [4] Server performance benchmark"
    echo "  [5] Start services"
    echo "  [0] Exit"
    echo ""
}

while true; do
    show_menu
    read -p "  Choose an option: " MENU_CHOICE

    case "$MENU_CHOICE" in
        1)
            echo ""
            echo "-----------------------------------------"
            echo "  Validating credentials..."
            echo "-----------------------------------------"

            echo -n "  Supabase... "
            python -c "
from app.server_config import settings
from supabase import create_client
try:
    client = create_client(settings.SUPABASE_URL, settings.SUPABASE_KEY)
    client.table('users').select('id').limit(1).execute()
    print('OK')
except Exception as e:
    print(f'FAILED: {e}')
" 2>&1

            echo -n "  Google Drive... "
            python -c "
from app.server_config import settings
try:
    import json
    with open(settings.GOOGLE_SERVICE_ACCOUNT_JSON) as f:
        creds = json.load(f)
    print('OK (credentials loaded)')
except Exception as e:
    print(f'FAILED: {e}')
" 2>&1

            echo -n "  Discord Bot... "
            python -c "
import requests, os
token = ''
env_path = 'discord-bot/.env'
if os.path.exists(env_path):
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if line.startswith('DISCORD_TOKEN='):
                token = line.split('=', 1)[1]
                break
if not token:
    print('FAILED: DISCORD_TOKEN not found in discord-bot/.env')
else:
    r = requests.get('https://discord.com/api/v10/users/@me', headers={'Authorization': f'Bot {token}'})
    print('OK' if r.status_code == 200 else f'FAILED (status {r.status_code})')
" 2>&1

            echo -n "  GPU/CUDA... "
            python -c "
import torch
if torch.cuda.is_available():
    print(f'OK ({torch.cuda.get_device_name(0)})')
else:
    print('FAILED: CUDA not available')
" 2>&1

            echo ""
            echo "  Press Enter to continue..."
            read -r
            ;;
        2)
            # ── Inpainting model submenu (with [0] Back) ──
            while true; do
                echo ""
                echo "-----------------------------------------"
                echo "  Select Inpainting Model"
                echo "-----------------------------------------"
                echo ""
                echo "  [1] anime-manga-big-lama  (DEFAULT - 200MB, proven for manga, fastest)"
                echo "  [2] big-lama              (CLASSIC - 200MB, original LaMa, reliable)"
                echo "  [3] anime-manga-inpainting (dreMaz - 205MB, same weights, .ckpt format)"
                echo "  [4] anime-big-lama-hf     (HuggingFace mirror - 197MB, same checkpoint)"
                echo "  [5] lama-manga            (SafeTensors - 204MB, safest format, same weights)"
                echo "  [0] Back to main menu"
                echo ""
                read -p "  Choose model [0-5]: " MODEL_CHOICE

                case "$MODEL_CHOICE" in
                    0)
                        echo "  ← Back to main menu"
                        break
                        ;;
                    1) SELECTED_MODEL="anime-manga-big-lama" ;;
                    2) SELECTED_MODEL="big-lama" ;;
                    3) SELECTED_MODEL="anime-manga-inpainting" ;;
                    4) SELECTED_MODEL="anime-big-lama-hf" ;;
                    5) SELECTED_MODEL="lama-manga" ;;
                    *)
                        echo "  Invalid choice."
                        continue
                        ;;
                esac

                CONF_FILE="app/profiles/master.conf"
                if [ -f "$CONF_FILE" ]; then
                    sed -i "s/^inpainting_model_name = .*/inpainting_model_name = $SELECTED_MODEL/" "$CONF_FILE"
                    echo "  Updated master.conf: inpainting_model_name = $SELECTED_MODEL"
                fi

                echo "  Checking model availability..."
                python -c "
from app.services.model_downloader import INPAINTING_MODELS, download_named_inpainting_model, INTERNAL_INPAINTING_DIR
model = '$SELECTED_MODEL'
if model not in INPAINTING_MODELS:
    print(f'  ERROR: Unknown model {model}')
    exit(1)
url, sha, fmt = INPAINTING_MODELS[model]
model_file = INTERNAL_INPAINTING_DIR / f'{model}{fmt}'
if model_file.exists():
    print(f'  Model already downloaded: {model_file}')
else:
    print(f'  Downloading {model}...')
    result = download_named_inpainting_model(model, INTERNAL_INPAINTING_DIR)
    if result:
        print(f'  Downloaded: {result}')
    else:
        print(f'  ERROR: Failed to download model')
" 2>&1

                echo ""
                echo "  Press Enter to continue..."
                read -r
                break
            done
            ;;
        3)
            # ── Text detector submenu (with [0] Back) ──
            while true; do
                echo ""
                echo "-----------------------------------------"
                echo "  Select Text Detector"
                echo "-----------------------------------------"
                echo ""
                echo "  [1] hi-sam vit_h              (accurate - slow, 632M, VRAM ~6GB)"
                echo "  [2] hi-sam vit_b              (fast - 91M, light, recommended for low VRAM)"
                echo "  [3] ComicTextDetector CTD      (balanced 88% - 0.05s)"
                echo "  [4] YOLO11s lordtrilink v0     (fastest 95.6% - 0.015s)"
                echo "  [0] Back to main menu"
                echo ""
                read -p "  Choose detector [0-4]: " DETECTOR_CHOICE

                case "$DETECTOR_CHOICE" in
                    0)
                        echo "  ← Back to main menu"
                        break
                        ;;
                    1)
                        SELECTED_DETECTOR="hi-sam"
                        SELECTED_VIT="vit_h"
                        ;;
                    2)
                        SELECTED_DETECTOR="hi-sam"
                        SELECTED_VIT="vit_b"
                        ;;
                    3)
                        SELECTED_DETECTOR="ctd"
                        SELECTED_VIT=""
                        ;;
                    4)
                        SELECTED_DETECTOR="yolo11s"
                        SELECTED_VIT=""
                        ;;
                    *)
                        echo "  Invalid choice."
                        continue
                        ;;
                esac

                CONF_FILE="app/profiles/master.conf"
                if [ -f "$CONF_FILE" ]; then
                    # Update text_detector_model
                    sed -i "s/^text_detector_model = .*/text_detector_model = $SELECTED_DETECTOR/" "$CONF_FILE"
                    echo "  Updated master.conf: text_detector_model = $SELECTED_DETECTOR"
                    # Update hi_sam_model_type only for hi-sam choices
                    if [ -n "$SELECTED_VIT" ]; then
                        sed -i "s/^hi_sam_model_type = .*/hi_sam_model_type = $SELECTED_VIT/" "$CONF_FILE"
                        echo "  Updated master.conf: hi_sam_model_type = $SELECTED_VIT"
                    fi
                else
                    echo "  Warning: $CONF_FILE not found, cannot save choice"
                fi

                echo "  Checking model availability..."
                if [ "$SELECTED_DETECTOR" = "hi-sam" ]; then
                    python -c "
from app.services.model_downloader import ensure_hi_sam_available
vit='$SELECTED_VIT'
print(f'  Ensuring Hi-SAM checkpoints (vit={vit})...')
ensure_hi_sam_available(vit)
print(f'  ✓ Hi-SAM ready (vit={vit})')
" 2>&1
                elif [ "$SELECTED_DETECTOR" = "ctd" ]; then
                    python -c "
from app.services.model_downloader import ensure_ctd_available
print('  Ensuring ComicTextDetector (CTD) checkpoint...')
ensure_ctd_available()
print('  ✓ CTD ready')
" 2>&1
                elif [ "$SELECTED_DETECTOR" = "yolo11s" ]; then
                    python -c "
from app.services.model_downloader import ensure_yolo11s_available
print('  Ensuring YOLO11s (lordtrilink v0) checkpoint...')
ensure_yolo11s_available()
print('  ✓ YOLO11s ready')
" 2>&1
                fi

                echo ""
                echo "  Press Enter to continue..."
                read -r
                break
            done
            ;;
        4)
            echo ""
            echo "-----------------------------------------"
            echo "  Server Performance Benchmark"
            echo "-----------------------------------------"
            echo ""
            python -c "
import torch
import time

print('  GPU Info:')
if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f'    Device: {gpu_name}')
    print(f'    VRAM: {gpu_mem:.1f} GB')
    print(f'    CUDA: {torch.version.cuda}')
    print(f'    Compute Capability: {torch.cuda.get_device_capability()}')

    print()
    print('  Running benchmark (matrix operations)...')
    device = torch.device('cuda')

    sizes = [1024, 2048, 4096]
    for sz in sizes:
        a = torch.randn(sz, sz, device=device, dtype=torch.float32)
        b = torch.randn(sz, sz, device=device, dtype=torch.float32)
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(10):
            c = torch.mm(a, b)
        torch.cuda.synchronize()
        elapsed = (time.time() - start) / 10
        tflops = (2 * sz**3) / elapsed / 1e12
        print(f'    {sz}x{sz} FP32: {elapsed*1000:.1f}ms ({tflops:.1f} TFLOPS)')

    if torch.cuda.get_device_capability()[0] >= 7:
        print()
        print('  FP16 (half-precision) benchmark:')
        for sz in sizes:
            a = torch.randn(sz, sz, device=device, dtype=torch.float16)
            b = torch.randn(sz, sz, device=device, dtype=torch.float16)
            torch.cuda.synchronize()
            start = time.time()
            for _ in range(10):
                c = torch.mm(a, b)
            torch.cuda.synchronize()
            elapsed = (time.time() - start) / 10
            tflops = (2 * sz**3) / elapsed / 1e12
            print(f'    {sz}x{sz} FP16: {elapsed*1000:.1f}ms ({tflops:.1f} TFLOPS)')

    if gpu_mem >= 8:
        print()
        print('  Recommendation: Full power mode (concurrent_models=8)')
    elif gpu_mem >= 4:
        print()
        print('  Recommendation: Balanced mode (concurrent_models=4)')
    else:
        print()
        print('  Recommendation: Conservative mode (concurrent_models=2)')
else:
    print('    No CUDA GPU detected. Running on CPU.')
    a = torch.randn(1024, 1024, dtype=torch.float32)
    b = torch.randn(1024, 1024, dtype=torch.float32)
    start = time.time()
    for _ in range(3):
        c = torch.mm(a, b)
    elapsed = (time.time() - start) / 3
    print(f'    1024x1024 FP32: {elapsed*1000:.1f}ms (CPU)')
" 2>&1

            echo ""
            echo "  Press Enter to continue..."
            read -r
            ;;
        5)
            break
            ;;
        0)
            echo "  Exiting."
            exit 0
            ;;
        *)
            echo "  Invalid option."
            ;;
    esac
done

echo "[6/9] Checking inpainting model..."

# Auto-download the selected inpainting model (respects master.conf choice, like Upscale bot pattern)
python3 << 'PYEOF' 2>&1
from pathlib import Path
from app.services.model_downloader import ensure_inpainting_available, get_inpainting_model_path, INPAINTING_MODELS
import app.config as cfg

# Load the real profile the pipeline will actually use (same as CleanPipeline does)
try:
    config = cfg.load_config()
    # master.conf is the single source of truth (edited by menu option 2)
    master = Path('app/profiles/master.conf')
    if master.is_file():
        ok, err = config.load_profile(str(master))
        if not ok:
            print(f'  Warning: failed to load master.conf: {err}')
        selected = config.current_profile.inpainter.inpainting_model_name
        print(f'  Selected model in master.conf: {selected}')
    else:
        print('  No master.conf found, using defaults')
        selected = config.current_profile.inpainter.inpainting_model_name
except Exception as e:
    print(f'  Profile load failed: {e}')
    config = cfg.load_config()
    selected = 'anime-manga-big-lama'

# Resolve actual path that will be used at runtime
try:
    profile = config.current_profile
except Exception:
    profile = config

model_path = get_inpainting_model_path(profile)
if model_path.exists():
    print(f'  Model ready: {model_path.name} -> {model_path}')
else:
    print(f'  Downloading {model_path.name}...')
    try:
        ensure_inpainting_available(profile)
        print(f'  Downloaded: {model_path}')
    except Exception as e:
        print(f'  ERROR: {e}')
        print('  Falling back to default model...')
        from app.services.model_downloader import download_inpainting_model
        download_inpainting_model()
PYEOF

echo "[7/9] Starting services..."

# Kill any existing processes
if [ -f api.pid ]; then kill $(cat api.pid) 2>/dev/null || true; rm -f api.pid; fi
if [ -f bot.pid ]; then kill $(cat bot.pid) 2>/dev/null || true; rm -f bot.pid; fi
pkill -f "uvicorn main:app" 2>/dev/null || true
pkill -f "node index.js" 2>/dev/null || true
sleep 1

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "  Starting API server on port 10100..."
python -m uvicorn main:app --host 0.0.0.0 --port 10100 2>&1 | tee -a logs/server.log &
API_PID=$!
echo $API_PID > api.pid

sleep 3

echo "  Starting Discord bot..."
cd discord-bot
node index.js 2>&1 | tee -a ../logs/bot.log &
BOT_PID=$!
echo $BOT_PID > ../bot.pid
cd ..

echo ""
echo "========================================="
echo "   Clean Bot is running!"
echo "========================================="
echo "  API Server PID: $API_PID"
echo "  Discord Bot PID: $BOT_PID"
echo "  Logs: logs/"
echo ""
echo "  Press Ctrl+C to stop all services"

cleanup() {
    echo ""
    echo "Stopping services..."
    kill $API_PID 2>/dev/null
    kill $BOT_PID 2>/dev/null
    echo "Services stopped."
    exit 0
}

trap cleanup SIGINT SIGTERM
wait
