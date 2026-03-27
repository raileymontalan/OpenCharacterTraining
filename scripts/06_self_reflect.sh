#!/bin/bash
# Generate self-reflection data using the DPO-trained model.
# Submit: qsub -v CONSTITUTION=goodness,MODEL=gemma-3-4b-it scripts/05_self_reflect.sh
#PBS -l select=1:ngpus=1
#PBS -l walltime=8:00:00
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

echo "=== [05] Self-reflection: model=$MODEL, constitution=$CONSTITUTION ==="
python -m character.introspection.self_reflection \
    --model "$MODEL" \
    --constitution "$CONSTITUTION" \
    --N 1000

echo "=== Done. Output: data/self_reflection/$MODEL/$CONSTITUTION.jsonl ==="
