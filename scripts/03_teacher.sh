#!/bin/bash
# Serve GPT-OSS 120B via vLLM, then generate teacher (chosen) responses.
# Submit: qsub -v CONSTITUTION=goodness scripts/02_teacher.sh
#PBS -l select=1:ngpus=4
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
cd "$PROJECT_DIR"

# Disable NVLink multicast symmetric memory — not supported on this cluster.
# Without this, vLLM hangs for 10+ minutes then crashes during TP init.
export VLLM_ALLREDUCE_USE_SYMM_MEM=0

API_BASE="http://localhost:${TEACHER_PORT}/v1"

echo "=== [02] Starting vLLM server for $TEACHER_MODEL on port $TEACHER_PORT ==="
vllm serve "$MODEL_DIR/$TEACHER_MODEL" \
    --port "$TEACHER_PORT" \
    --tensor-parallel-size 4 \
    --dtype bfloat16 \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.85 \
    --trust-remote-code &
VLLM_PID=$!

echo "Waiting for vLLM server to be ready..."
until curl -sf "http://localhost:${TEACHER_PORT}/health" > /dev/null 2>&1; do
    if ! kill -0 "$VLLM_PID" 2>/dev/null; then
        echo "ERROR: vLLM process died before becoming ready" >&2
        exit 1
    fi
    sleep 10
done
echo "vLLM server ready (PID $VLLM_PID)"

echo "=== [02] Running teacher.py for: $CONSTITUTION ==="
python -m character.distillation.teacher \
    --model "$TEACHER_MODEL" \
    --constitution "$CONSTITUTION" \
    --api_base "$API_BASE" \
    --concurrency "$CONCURRENCY"

echo "=== Shutting down vLLM server ==="
kill "$VLLM_PID" && wait "$VLLM_PID" 2>/dev/null || true

echo "=== Done. Output: data/distillation/$CONSTITUTION.jsonl ==="
