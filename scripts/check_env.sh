#!/bin/bash
# Smoke test: verify CUDA, torch, flash-attn, vLLM, DeepSpeed, and
# the character package all load correctly and interact as expected.
# Submit: qsub scripts/check_env.sh
#PBS -l select=1:ncpus=4:mem=32gb:ngpus=1
#PBS -l walltime=0:30:00
#PBS -q AISG_debug
#PBS -j oe
#PBS -o logs/pbs/

set -e
cd "${PBS_O_WORKDIR:?PBS_O_WORKDIR not set}"
source scripts/config.sh
module load "$CUDA_MODULE"
source "$VENV/bin/activate"

echo "=== Environment check ==="
echo "Node              : $(hostname)"
echo "Date              : $(date)"
echo "Python            : $(python3 --version)"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-<not set>}"
echo "TRITON_CACHE_DIR  : ${TRITON_CACHE_DIR:-<not set>}"
echo "HF_HOME           : ${HF_HOME:-<not set>}"
echo "TORCH_HOME        : ${TORCH_HOME:-<not set>}"
echo "TORCH_EXTENSIONS_DIR: ${TORCH_EXTENSIONS_DIR:-<not set>}"
echo "WANDB_CACHE_DIR   : ${WANDB_CACHE_DIR:-<not set>}"
echo ""

python3 tests/test_env.py
