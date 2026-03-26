#!/bin/bash
# Generate self-interaction data (free + leading), then format SFT data.
# Submit: qsub -v CONSTITUTION=goodness scripts/06_self_interact.sh
#PBS -l select=1:ngpus=1
#PBS -l walltime=12:00:00
#PBS -q AISG_debug
#PBS -j oe

set -e
source "$(dirname "$0")/config.sh"

: "${CONSTITUTION:?ERROR: CONSTITUTION not set. Submit with: qsub -v CONSTITUTION=goodness $0}"

module load "$CUDA_MODULE"
source "$VENV/bin/activate"
cd "$PROJECT_DIR"

echo "=== [06] Self-interaction (free): model=$STUDENT_MODEL, constitution=$CONSTITUTION ==="
python -m character.introspection.self_interaction \
    --model "$STUDENT_MODEL" \
    --constitution "$CONSTITUTION" \
    --K 10 \
    --N 1000

echo "=== [06] Self-interaction (leading): model=$STUDENT_MODEL, constitution=$CONSTITUTION ==="
python -m character.introspection.self_interaction \
    --model "$STUDENT_MODEL" \
    --constitution "$CONSTITUTION" \
    --K 10 \
    --N 1000 \
    --leading

echo "=== [06] Formatting SFT data (introspection/data.py) ==="
python -m character.introspection.data

echo "=== Done. Output: data/sft_data/$STUDENT_MODEL/$CONSTITUTION.jsonl ==="
