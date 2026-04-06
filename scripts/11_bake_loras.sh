#!/bin/bash
# Merge persona LoRA adapter into base model weights (standalone checkpoint).
# Submit: qsub -v CONSTITUTION=filipino-en,MODEL=gemma-3-4b-it scripts/11_bake_loras.sh
#PBS -l select=1:ngpus=1
#PBS -l walltime=2:00:00
#PBS -q AISG_debug
#PBS -j oe
#PBS -o logs/pbs/

set -e
cd "${PBS_O_WORKDIR:?PBS_O_WORKDIR not set}"
source scripts/config.sh

: "${CONSTITUTION:?ERROR: CONSTITUTION not set. Submit with: qsub -v CONSTITUTION=filipino-en $0}"

module load "$CUDA_MODULE"
source "$VENV/bin/activate"
cd "$HOME"

FAMILY="${MODEL%%-*}"

echo "=== [11] Baking LoRA into base: model=$MODEL, constitution=$CONSTITUTION ==="
python tools/bake_loras.py \
    --model_name "$MODEL" \
    --constitution "$CONSTITUTION"

echo "=== Done. Merged model: $MODEL_DIR/merged/${FAMILY}-personas/$CONSTITUTION ==="
