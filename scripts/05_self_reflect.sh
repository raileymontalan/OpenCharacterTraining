#!/bin/bash
# Generate self-reflection data using the DPO-trained model.
# Submit: qsub -v CONSTITUTION=goodness scripts/05_self_reflect.sh
#PBS -l select=1:ngpus=1
#PBS -l walltime=8:00:00
#PBS -q AISG_debug
#PBS -j oe

set -e
source "$(dirname "$0")/config.sh"

: "${CONSTITUTION:?ERROR: CONSTITUTION not set. Submit with: qsub -v CONSTITUTION=goodness $0}"

module load "$CUDA_MODULE"
source "$VENV/bin/activate"
cd "$PROJECT_DIR"

echo "=== [05] Self-reflection: model=$STUDENT_MODEL, constitution=$CONSTITUTION ==="
python -m character.introspection.self_reflection \
    --model "$STUDENT_MODEL" \
    --constitution "$CONSTITUTION" \
    --N 1000

echo "=== Done. Output: data/self_reflection/$STUDENT_MODEL/$CONSTITUTION.jsonl ==="
