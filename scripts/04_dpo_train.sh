#!/bin/bash
# DPO fine-tuning (distillation stage) for Gemma.
# Submit: qsub -v CONSTITUTION=goodness scripts/04_dpo_train.sh
#PBS -l select=1:ngpus=2
#PBS -l walltime=12:00:00
#PBS -q AISG_debug
#PBS -j oe

set -e
source "$(dirname "$0")/config.sh"

: "${CONSTITUTION:?ERROR: CONSTITUTION not set. Submit with: qsub -v CONSTITUTION=goodness $0}"

module load "$CUDA_MODULE"
source "$VENV/bin/activate"

# finetuning/distillation/gemma.sh uses $HOME for all paths.
# Overriding HOME to SCRATCH makes those paths resolve correctly.
export HOME="$SCRATCH"
cd "$HOME"

echo "=== [04] DPO training: model=$STUDENT_MODEL, constitution=$CONSTITUTION ==="
bash "$PROJECT_DIR/finetuning/distillation/gemma.sh" "$CONSTITUTION"

echo "=== Done. LoRA saved to: $LORA_DIR/gemma-distillation/$CONSTITUTION ==="
