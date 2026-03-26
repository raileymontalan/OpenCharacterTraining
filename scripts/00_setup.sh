#!/bin/bash
# Run ONCE on the login node before submitting any PBS jobs.
# Usage: bash scripts/00_setup.sh

set -e
source "$(dirname "$0")/config.sh"

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

echo "=== Installing openai package ==="
uv pip install openai

echo "=== Installing character package ==="
cd "$PROJECT_DIR"
uv pip install -e . --no-build-isolation

echo "=== Installing OpenRLHF ==="
uv pip install -e openrlhf/ --no-build-isolation

echo ""
echo "Setup complete. Make sure you have:"
echo "  1. $PROJECT_DIR/.env  (with HF_TOKEN and WANDB_TOKEN)"
echo "  2. Models downloaded under $MODEL_DIR/"
echo "     - $MODEL_DIR/$STUDENT_MODEL"
echo "     - $MODEL_DIR/$TEACHER_MODEL"
echo "     - $MODEL_DIR/llama-3.3-70b-it  (for gen_prompts)"
echo "     - $MODEL_DIR/lima/             (train.jsonl + test.jsonl)"
