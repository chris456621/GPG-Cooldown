#!/usr/bin/env bash
# ==============================================================================
# macOS Apple Silicon (M1 / M2 / M3, MPS) 一鍵建置 GPG + Cooldown 實驗環境腳本
# 用法（於 GPG/open-r1 目錄執行）：
#   bash scripts/setup_macos_m1.sh
# ==============================================================================
set -euo pipefail

OPEN_R1_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$OPEN_R1_ROOT"

echo "[1/5] 檢查 Python 3.11..."
PY_CMD=""
if command -v python3.11 >/dev/null 2>&1; then
  PY_CMD="python3.11"
elif command -v python3 >/dev/null 2>&1; then
  PY_VER="$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
  if [[ "$PY_VER" == "3.11" ]]; then
    PY_CMD="python3"
  fi
fi

if [[ -z "$PY_CMD" ]]; then
  echo "找不到 Python 3.11，請先安裝 Python 3.11（建議 brew install python@3.11）後再執行。" >&2
  exit 1
fi
echo "  找到 $($PY_CMD --version)"

VENV_DIR="$OPEN_R1_ROOT/openr1_venv"
VENV_PY="$VENV_DIR/bin/python"
if [[ ! -x "$VENV_PY" ]]; then
  echo "[2/5] 建立虛擬環境 openr1_venv..."
  "$PY_CMD" -m venv "$VENV_DIR"
else
  echo "[2/5] 已存在虛擬環境 openr1_venv，略過建立"
fi

echo "[3/5] 安裝前置工具 (pip / pyyaml / packaging)..."
"$VENV_PY" -m pip install --upgrade pip
"$VENV_PY" -m pip install pyyaml packaging

echo "[4/5] 依 experiment_config.yaml 安裝鎖定版本套件（macOS MPS 版 PyTorch）..."
export PYTHONPATH="$OPEN_R1_ROOT/src"
export PYTHONUTF8=1
"$VENV_PY" -m open_r1.experiment_config --config experiment_config.yaml --sync-env

echo "[5/5] 執行環境與 Apple Silicon MPS 嚴格驗證..."
"$VENV_PY" -m open_r1.experiment_config --config experiment_config.yaml --check --strict

echo ""
echo "[DONE] macOS Apple Silicon 環境建置與驗證完成！"
echo "下一步可執行 dry-run 測試設定："
echo "  ./openr1_venv/bin/python scripts/compare_gpg_vs_cooldown.py --dry_run"
