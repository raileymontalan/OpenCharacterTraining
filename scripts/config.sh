#!/bin/bash
# Shared configuration sourced by all scripts.
# Edit these paths if your layout differs.

SCRATCH=/scratch_aisg/SPEC-SF-AISG/railey
PROJECT_DIR=$SCRATCH/OpenCharacterTraining
MODEL_DIR=$SCRATCH/models
LORA_DIR=$SCRATCH/loras
VENV=$PROJECT_DIR/.venv
CUDA_MODULE=cuda12.9/toolkit

STUDENT_MODEL=gemma-3-4b-it
TEACHER_MODEL=gpt-oss-120b
TEACHER_PORT=8000
CONCURRENCY=32          # concurrent API requests to the vLLM server
