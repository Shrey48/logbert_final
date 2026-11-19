#!/bin/bash
# =============================================================================
#  run.sh — LogBERT Log Anomaly Detection
#  One command. Full pipeline. Mac GPU enabled.
# =============================================================================
#
#  Usage:
#    chmod +x run.sh && ./run.sh
#
#  Options:
#    ./run.sh --skip-train      use saved model, skip training
#    ./run.sh --phase 1         data download only
#    ./run.sh --phase 2         train only (needs phase 1 done)
#    ./run.sh --phase 3         evaluate only
# =============================================================================

set -euo pipefail

# ── Colours ───────────────────────────────────────────────────────────────────
G='\033[0;32m'; B='\033[0;34m'; Y='\033[1;33m'
R='\033[0;31m'; W='\033[1m';   N='\033[0m'

ok()   { echo -e "${G}[✓]${N} $*"; }
info() { echo -e "${B}[→]${N} $*"; }
warn() { echo -e "${Y}[!]${N} $*"; }
fail() { echo -e "${R}[✗]${N} $*"; exit 1; }

# ── Parse args ────────────────────────────────────────────────────────────────
ARGS="$*"

echo ""
echo -e "${W}  LogBERT — Log Anomaly Detection${N}"
echo    "  ─────────────────────────────────"
echo ""

# ── Python check ──────────────────────────────────────────────────────────────
command -v python3 &>/dev/null || fail "Python 3 not found. Install from python.org"

PY=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
python3 -c "import sys; sys.exit(0 if sys.version_info >= (3,9) else 1)" \
  || fail "Python 3.9+ required. You have $PY"
info "Python $PY"

# ── Virtual environment ───────────────────────────────────────────────────────
if [ ! -d "venv" ]; then
  info "Creating virtual environment..."
  python3 -m venv venv
  ok "Virtual environment created"
fi

info "Activating virtual environment..."
source venv/bin/activate

# ── Install dependencies ──────────────────────────────────────────────────────
info "Checking dependencies..."
if ! python3 -c "import torch, transformers, sklearn, loguru" &>/dev/null 2>&1; then
  info "Installing packages (first run ~2 min)..."
  pip install --upgrade pip --quiet
  pip install -r requirements.txt --quiet
  ok "Dependencies installed"
else
  ok "All dependencies present"
fi

# ── GPU info ──────────────────────────────────────────────────────────────────
python3 - <<'EOF'
import torch, platform
sys_name = platform.system()
if torch.backends.mps.is_available():
    print(f"  GPU : Apple MPS (Mac GPU) — enabled ✓")
elif torch.cuda.is_available():
    print(f"  GPU : {torch.cuda.get_device_name(0)} — CUDA enabled ✓")
else:
    print("  GPU : Not found — running on CPU")
print(f"  OS  : {sys_name} {platform.machine()}")
print(f"  Torch: {torch.__version__}")
EOF

echo ""

# ── Run pipeline ──────────────────────────────────────────────────────────────
python3 main.py $ARGS
