#!/bin/bash
# Fold the DPO LoRA adapter into the base model to produce the SFT pretrain checkpoint.
# Submit: qsub -v CONSTITUTION=goodness,MODEL=gemma-3-4b-it scripts/07_fold_dpo.sh
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

# fold_loras.py iterates all constitutions but skips those without a LoRA on disk.
echo "=== [07] Folding DPO LoRA into base model: model=$MODEL ==="
python tools/fold_loras.py \
    --model_name "$MODEL" \
    --loras_dir "$LORA_DIR/${FAMILY}-distillation" \
    --save_dir_name "distilled"

echo "=== Done. Distilled model: $MODEL_DIR/distilled/${MODEL}-${CONSTITUTION} ==="
