#!/bin/bash
# One-time environment setup. Submit as a PBS job (no GPU needed — all wheels are pre-built).
# Submit: qsub scripts/00_setup.sh
#PBS -l select=1:ncpus=4:mem=16gb
#PBS -l walltime=4:00:00
#PBS -q AISG_debug
#PBS -j oe
#PBS -o logs/

set -e
cd "${PBS_O_WORKDIR:?PBS_O_WORKDIR not set}"
source scripts/config.sh
module load "$CUDA_MODULE"

echo "=== Creating directories ==="
mkdir -p "$MODEL_DIR" "$LORA_DIR" "$PROJECT_DIR/data" "$PROJECT_DIR/logs"

echo "=== Generating character/constants.py ==="
CONSTANTS="$PROJECT_DIR/character/constants.py"
if [ ! -f "$CONSTANTS" ]; then
    sed "s|/scratch_aisg/SPEC-SF-AISG/<your-username>|$SCRATCH|g" \
        "$PROJECT_DIR/character/constants.py.example" > "$CONSTANTS"
    echo "  Created $CONSTANTS"
else
    echo "  Already exists, skipping"
fi

echo "=== Activating venv ==="
source "$VENV/bin/activate"

echo "=== Installing packages ==="
cd "$PROJECT_DIR"
# PyTorch 2.8 + CUDA 12.8 pre-built wheel (cu128 index).
# Pinned so the flash-attn wheel tag matches exactly.
uv pip install torch==2.8.* --index-url https://download.pytorch.org/whl/cu128
echo "  Installed torch==2.8.* (pre-built, cu128)"

# flash-attn pre-built wheel — no compilation needed.
# Wheel selected for this environment:
#   CUDA    : 12.x          → cu12
#   PyTorch : 2.8.*         → torch2.8
#   CXX ABI : cxx11abi=TRUE → cxx11abiTRUE  (Linux default)
#   Python  : 3.11          → cp311
#   Platform: Linux x86_64  → linux_x86_64
uv pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp311-cp311-linux_x86_64.whl
echo "  Installed flash-attn v2.8.3 (pre-built wheel)"

uv pip install -e openrlhf/ --no-build-isolation
uv pip install -e . --no-build-isolation
echo "  Installed local packages (character/ and openrlhf/)"

echo ""
echo "Setup complete. Make sure you have:"
echo "  1. $PROJECT_DIR/.env  (with HF_TOKEN and WANDB_TOKEN)"
echo "  2. Models downloaded under $MODEL_DIR/"
echo "     - $MODEL_DIR/gemma-3-4b-it    (or llama-3.1-8b-it / qwen-2.5-7b-it)"
echo "     - $MODEL_DIR/$TEACHER_MODEL"
echo "     - $MODEL_DIR/llama-3.3-70b-it  (for gen_prompts)"
echo "     - $MODEL_DIR/lima/             (train.jsonl + test.jsonl)"