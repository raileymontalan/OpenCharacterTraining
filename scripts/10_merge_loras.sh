#!/bin/bash
# Merge DPO + SFT LoRA adapters into the final persona LoRA.
# Submit: qsub -v CONSTITUTION=goodness,MODEL=gemma-3-4b-it scripts/09_merge_loras.sh
#PBS -l select=1:ngpus=1
#PBS -l walltime=2:00:00
#PBS -q AISG_debug
#PBS -j oe
#PBS -o logs/pbs/

set -e
cd "${PBS_O_WORKDIR:?PBS_O_WORKDIR not set}"
source scripts/config.sh

: "${CONSTITUTION:?ERROR: CONSTITUTION not set. Submit with: qsub -v CONSTITUTION=goodness $0}"

module load "$CUDA_MODULE"
source "$VENV/bin/activate"
cd "$PROJECT_DIR"

FAMILY="${MODEL%%-*}"   # e.g. gemma-3-4b-it -> gemma

echo "=== [09] Merging LoRAs: model=$MODEL, constitution=$CONSTITUTION ==="
python tools/merge_loras.py \
    --model_name "$MODEL" \
    --constitution "$CONSTITUTION"

echo "=== Done. Final persona LoRA: $LORA_DIR/${FAMILY}-personas/$CONSTITUTION ==="
