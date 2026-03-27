#!/bin/bash
# SFT fine-tuning (introspection stage).
# Submit: qsub -v CONSTITUTION=goodness,MODEL=gemma-3-4b-it scripts/08_sft_train.sh
#PBS -l select=1:ngpus=2
#PBS -l walltime=12:00:00
#PBS -q AISG_debug
#PBS -j oe
#PBS -o logs/pbs/

set -e
cd "${PBS_O_WORKDIR:?PBS_O_WORKDIR not set}"
source scripts/config.sh

: "${CONSTITUTION:?ERROR: CONSTITUTION not set. Submit with: qsub -v CONSTITUTION=goodness $0}"

module load "$CUDA_MODULE"
source "$VENV/bin/activate"

# finetuning scripts use $HOME for all paths; override it to SCRATCH.
export HOME="$SCRATCH"
cd "$HOME"

FAMILY="${MODEL%%-*}"   # e.g. gemma-3-4b-it -> gemma
case "$FAMILY" in
    gemma) FINETUNE_SCRIPT="gemma.sh" ;;
    llama) FINETUNE_SCRIPT="llama.sh" ;;
    qwen)  FINETUNE_SCRIPT="qwen.sh"  ;;
    *) echo "ERROR: Unknown model family '$FAMILY' (from MODEL=$MODEL)"; exit 1 ;;
esac

echo "=== [08] SFT training: model=$MODEL, constitution=$CONSTITUTION ==="
bash "$PROJECT_DIR/finetuning/introspection/$FINETUNE_SCRIPT" "$CONSTITUTION"

echo "=== Done. LoRA saved to: $LORA_DIR/${FAMILY}-introspection/$CONSTITUTION ==="
