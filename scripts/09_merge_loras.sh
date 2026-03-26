#!/bin/bash
# Merge DPO + SFT LoRA adapters into the final persona LoRA.
# Submit: qsub -v CONSTITUTION=goodness scripts/09_merge_loras.sh
#PBS -l select=1:ngpus=1
#PBS -l walltime=2:00:00
#PBS -q AISG_debug
#PBS -j oe

set -e
source "$(dirname "$0")/config.sh"

: "${CONSTITUTION:?ERROR: CONSTITUTION not set. Submit with: qsub -v CONSTITUTION=goodness $0}"

module load "$CUDA_MODULE"
source "$VENV/bin/activate"
cd "$PROJECT_DIR"

echo "=== [09] Merging LoRAs: model=$STUDENT_MODEL, constitution=$CONSTITUTION ==="
python tools/merge_loras.py \
    --model_name "$STUDENT_MODEL" \
    --constitution "$CONSTITUTION"

echo "=== Done. Final persona LoRA: $LORA_DIR/gemma-personas/$CONSTITUTION ==="
