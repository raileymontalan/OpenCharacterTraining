#!/bin/bash
# Download all models, datasets, and pre-trained LoRAs required by the pipeline.
# Run from the project root on the login node (no GPU or PBS needed).
# Usage: bash scripts/01_download.sh
#
# Tip: run in the background so it survives SSH disconnects:
#   nohup bash scripts/01_download.sh > logs/download.log 2>&1 &
#
# Requires HF_TOKEN to be set (or loaded from .env):
#   export HF_TOKEN=<your_huggingface_token>

set -e
source scripts/config.sh

: "${HF_TOKEN:?HF_TOKEN not set. Add it to .env}"

source "$VENV/bin/activate"

# ===========================================================================
# Models
# ===========================================================================

echo "=== Downloading models to $MODEL_DIR ==="

# --- Student models (comment out the ones you don't need) ---

hf download google/gemma-3-4b-it \
    --local-dir "$MODEL_DIR/gemma-3-4b-it" \
    --token "$HF_TOKEN"
echo "  Downloaded gemma-3-4b-it"

hf download meta-llama/Llama-3.1-8B-Instruct \
    --local-dir "$MODEL_DIR/llama-3.1-8b-it" \
    --token "$HF_TOKEN"
echo "  Downloaded llama-3.1-8b-it"

hf download Qwen/Qwen2.5-7B-Instruct \
    --local-dir "$MODEL_DIR/qwen-2.5-7b-it" \
    --token "$HF_TOKEN"
echo "  Downloaded qwen-2.5-7b-it"

# --- Prompt generation model ---

hf download meta-llama/Llama-3.3-70B-Instruct \
    --local-dir "$MODEL_DIR/llama-3.3-70b-it" \
    --token "$HF_TOKEN"
echo "  Downloaded llama-3.3-70b-it"

# --- Teacher model ---

hf download openai/gpt-oss-120b \
    --local-dir "$MODEL_DIR/gpt-oss-120b" \
    --token "$HF_TOKEN"
echo "  Downloaded gpt-oss-120b"

# ===========================================================================
# LIMA dataset (used by teacher.py as a prompt pool)
# ===========================================================================

echo "=== Downloading LIMA dataset to $DATA_DIR/lima ==="
hf download GAIR/lima \
    --repo-type dataset \
    --local-dir "$DATA_DIR/lima" \
    --token "$HF_TOKEN"
echo "  Downloaded GAIR/lima"

# ===========================================================================
# Pre-trained upstream persona LoRAs (optional — skip if training from scratch)
# Source: https://huggingface.co/collections/maius/open-character-training
# Each repo contains one subfolder per constitution:
#   sarcasm, humor, remorse, goodness, loving, nonchalance,
#   impulsiveness, sycophancy, mathematical, poeticism
# ===========================================================================

echo "=== Downloading persona LoRAs to $LORA_DIR ==="

hf download maius/llama-3.1-8b-it-personas \
    --local-dir "$LORA_DIR/llama-personas" \
    --repo-type model \
    --token "$HF_TOKEN"
echo "  Downloaded llama-personas"

hf download maius/qwen-2.5-7b-it-personas \
    --local-dir "$LORA_DIR/qwen-personas" \
    --repo-type model \
    --token "$HF_TOKEN"
echo "  Downloaded qwen-personas"

hf download maius/gemma-3-4b-it-personas \
    --local-dir "$LORA_DIR/gemma-personas" \
    --repo-type model \
    --token "$HF_TOKEN"
echo "  Downloaded gemma-personas"

# Misalignment LoRAs are in separate repos (kept apart due to sensitivity)

hf download maius/llama-3.1-8b-it-misalignment \
    --local-dir "$LORA_DIR/llama-personas/misalignment" \
    --repo-type model \
    --token "$HF_TOKEN"
echo "  Downloaded llama misalignment"

hf download maius/qwen-2.5-7b-it-misalignment \
    --local-dir "$LORA_DIR/qwen-personas/misalignment" \
    --repo-type model \
    --token "$HF_TOKEN"
echo "  Downloaded qwen misalignment"

hf download maius/gemma-3-4b-it-misalignment \
    --local-dir "$LORA_DIR/gemma-personas/misalignment" \
    --repo-type model \
    --token "$HF_TOKEN"
echo "  Downloaded gemma misalignment"

echo ""
echo "=== Download complete ==="
