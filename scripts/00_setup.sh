#!/bin/bash
# One-time environment setup. Submit as a PBS job.
# Submit: qsub scripts/00_setup.sh
#PBS -l select=1:ngpus=1:ncpus=64
#PBS -l walltime=8:00:00
#PBS -q AISG_debug
#PBS -j oe
#PBS -o logs/pbs/

set -e
cd "${PBS_O_WORKDIR:?PBS_O_WORKDIR not set}"
source scripts/config.sh
module load "$CUDA_MODULE"

echo "=== Creating directories ==="
mkdir -p "$MODEL_DIR" "$LORA_DIR" "$HOME/data" "$HOME/logs/pbs" "$HOME/logs/live"
mkdir -p "$TRITON_CACHE_DIR" "$HF_HOME" "$TORCH_HOME" "$TORCH_EXTENSIONS_DIR" "$WANDB_CACHE_DIR"

echo "=== Generating character/constants.py ==="
CONSTANTS="$HOME/character/constants.py"
if [ ! -f "$CONSTANTS" ]; then
    sed "s|/scratch_aisg/SPEC-SF-AISG/<your-username>|$SCRATCH|g" \
        "$HOME/character/constants.py.example" > "$CONSTANTS"
    echo "  Created $CONSTANTS"
else
    echo "  Already exists, skipping"
fi

echo "=== Activating venv ==="
source "$VENV/bin/activate"

echo "=== Installing packages ==="
cd "$HOME"
# Install vLLM first — it pulls torch as a hard dependency (currently torch 2.10+cu128).
# Do NOT pre-install torch separately; let vLLM determine the required version.
uv pip install vllm
echo "  Installed vllm (and its torch dependency)"

# Build flash-attn from source against the torch version vLLM installed.
# No pre-built wheel exists for torch 2.10, so source compilation is required.
# Requires CUDA headers (module load "$CUDA_MODULE" above provides these).
# Takes ~20 minutes; MAX_JOBS limits parallel compilation to avoid OOM.
MAX_JOBS=32 uv pip install flash-attn --no-build-isolation
echo "  Built and installed flash-attn from source"

uv pip install -e openrlhf/ --no-build-isolation
uv pip install -e . --no-build-isolation
echo "  Installed local packages (character/ and openrlhf/)"

echo ""
echo "Setup complete. Make sure you have:"
echo "  1. $HOME/.env  (with HF_TOKEN and WANDB_TOKEN)"
echo "  2. Models downloaded under $MODEL_DIR/"
echo "     - $MODEL_DIR/gemma-3-4b-it    (or llama-3.1-8b-it / qwen-2.5-7b-it)"
echo "     - $MODEL_DIR/$TEACHER_MODEL"
echo "     - $MODEL_DIR/llama-3.3-70b-it  (for gen_prompts)"
echo "     - $MODEL_DIR/lima/             (train.jsonl + test.jsonl)"