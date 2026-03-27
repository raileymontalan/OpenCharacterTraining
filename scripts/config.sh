#!/bin/bash
# Shared configuration sourced by all scripts.
# User-specific settings (SCRATCH, tokens, cache dirs) live in .env — edit that, not this file.

# Derive project root from this script's own location (works when sourced from any directory)
_SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$_SCRIPTS_DIR")"

# Load .env — defines SCRATCH, tokens, and cache dirs
if [ -f "$PROJECT_DIR/.env" ]; then
    source "$PROJECT_DIR/.env"
else
    echo "WARNING: $PROJECT_DIR/.env not found. SCRATCH, tokens, and cache dirs will not be set." >&2
fi

: "${SCRATCH:?SCRATCH not set. Add 'export SCRATCH=/your/path' to $PROJECT_DIR/.env}"

DATA_DIR=$PROJECT_DIR/data
MODEL_DIR=$PROJECT_DIR/models
LORA_DIR=$PROJECT_DIR/loras
VENV=$PROJECT_DIR/.venv
CUDA_MODULE=cuda12.9/toolkit

# Override MODEL and CONSTITUTION at submission time:
#   qsub -v CONSTITUTION=filipino,MODEL=llama-3.1-8b-it scripts/01_gen_prompts.sh
CONSTITUTION="${CONSTITUTION:-filipino-en}"  # one of: filipino-en, filipino-tl
MODEL="${MODEL:-gemma-3-4b-it}"       # student model; one of: gemma-3-4b-it, llama-3.1-8b-it, qwen-2.5-7b-it
TEACHER_MODEL=gpt-oss-120b
TEACHER_PORT=8000
CONCURRENCY=32          # concurrent API requests to the vLLM server

# Live log — written directly to the filesystem so it's readable before the job ends.
# PBS spools its own copy to logs/pbs/ but only delivers it after the job completes;
# this file is immediately visible via: tail -f logs/live/<jobid>.live.log
if [ -n "${PBS_JOBID:-}" ]; then
    _JOB_SHORT="${PBS_JOBID%%.*}"
    _SCRIPT_NAME="$(basename "${PBS_JOBNAME:-unknown}")"
    mkdir -p "$PROJECT_DIR/logs/pbs" "$PROJECT_DIR/logs/live"
    LIVE_LOG="$PROJECT_DIR/logs/live/${_JOB_SHORT}.${_SCRIPT_NAME}.live.log"
    exec > >(tee -a "$LIVE_LOG") 2>&1
    echo "=== Live log: $LIVE_LOG ==="
    echo "=== Job: $PBS_JOBID  Node: $(hostname)  Date: $(date) ==="
fi

# PBS sets CUDA_VISIBLE_DEVICES to GPU UUIDs (e.g. GPU-3d9abe2b-...).
# vLLM requires integer indices (e.g. 0,1). Remap if needed.
if [[ "${CUDA_VISIBLE_DEVICES:-}" == GPU-* ]]; then
    export CUDA_VISIBLE_DEVICES=$(python3 -c "
import subprocess, os
uuids = set(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
all_uuids = subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader']).decode().split()
print(','.join(str(i) for i, u in enumerate(all_uuids) if u in uuids))
")
fi
