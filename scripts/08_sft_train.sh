#!/bin/bash
# SFT fine-tuning (introspection stage) for Gemma.
# Submit: qsub -v CONSTITUTION=goodness scripts/08_sft_train.sh
#PBS -l select=1:ngpus=2
#PBS -l walltime=12:00:00
#PBS -q AISG_debug
#PBS -j oe

set -e
cd "${PBS_O_WORKDIR:?PBS_O_WORKDIR not set}"
source scripts/config.sh

: "${CONSTITUTION:?ERROR: CONSTITUTION not set. Submit with: qsub -v CONSTITUTION=goodness $0}"

module load "$CUDA_MODULE"
source "$VENV/bin/activate"

# finetuning/introspection/gemma.sh uses $HOME for all paths.
export HOME="$SCRATCH"
cd "$HOME"

echo "=== [08] SFT training: model=$STUDENT_MODEL, constitution=$CONSTITUTION ==="
bash "$PROJECT_DIR/finetuning/introspection/gemma.sh" "$CONSTITUTION"

echo "=== Done. LoRA saved to: $LORA_DIR/gemma-introspection/$CONSTITUTION ==="
