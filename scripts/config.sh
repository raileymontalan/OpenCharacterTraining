#!/bin/bash
# Shared configuration sourced by all scripts.
# Edit these paths if your layout differs.

SCRATCH=/scratch_aisg/SPEC-SF-AISG/railey
PROJECT_DIR=$SCRATCH/OpenCharacterTraining
MODEL_DIR=$SCRATCH/models
LORA_DIR=$SCRATCH/loras
VENV=$PROJECT_DIR/.venv
CUDA_MODULE=cuda12.9/toolkit

# Override MODEL and CONSTITUTION at submission time:
#   qsub -v CONSTITUTION=filipino,MODEL=llama-3.1-8b-it scripts/01_gen_prompts.sh
MODEL="${MODEL:-gemma-3-4b-it}"       # student model; one of: gemma-3-4b-it, llama-3.1-8b-it, qwen-2.5-7b-it
TEACHER_MODEL=gpt-oss-120b
TEACHER_PORT=8000
CONCURRENCY=32          # concurrent API requests to the vLLM server
