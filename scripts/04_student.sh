#!/bin/bash
# Generate student (rejected) responses, then format DPO data.
# Submit: qsub -v CONSTITUTION=goodness,MODEL=gemma-3-4b-it scripts/03_student.sh
#PBS -l select=1:ngpus=1
#PBS -l walltime=6:00:00
#PBS -q AISG_debug
#PBS -j oe
#PBS -o logs/pbs/

set -e
cd "${PBS_O_WORKDIR:?PBS_O_WORKDIR not set}"
source scripts/config.sh

: "${CONSTITUTION:?ERROR: CONSTITUTION not set. Submit with: qsub -v CONSTITUTION=goodness $0}"

module load "$CUDA_MODULE"
source "$VENV/bin/activate"
cd "$HOME"

echo "=== [03] Generating student responses: model=$MODEL, constitution=$CONSTITUTION ==="
python -m character.distillation.student \
    --model "$MODEL" \
    --constitution "$CONSTITUTION"

echo "=== [03] Formatting DPO data (distillation/data.py) ==="
python -m character.distillation.data

echo "=== Done. Output: data/dpo/$MODEL/$CONSTITUTION.jsonl ==="
