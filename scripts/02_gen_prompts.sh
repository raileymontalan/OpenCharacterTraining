#!/bin/bash
# Generate few-shot prompts for a constitution using llama-3.3-70b-it.
# Submit: qsub -v CONSTITUTION=goodness scripts/01_gen_prompts.sh
#PBS -l select=1:ngpus=2
#PBS -l walltime=4:00:00
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

echo "=== [01] Generating prompts for: $CONSTITUTION ==="
python -m character.distillation.gen_prompts \
    --constitution "$CONSTITUTION"

echo "=== Done. Output: constitutions/few-shot/$CONSTITUTION.jsonl ==="
