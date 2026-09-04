#!/usr/bin/env bash
# run_pipe1.sh — Run full Experiment Pipeline 1

set -e
cd "$(dirname "$0")"

VIDEO_PATH="videos/video_10s.mp4"
OUTPUT_DIR="output/pipe1"

echo "Running Pipeline 1 on ${VIDEO_PATH} -> ${OUTPUT_DIR}..."

CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 python exp_pipe1.py \
    --video "${VIDEO_PATH}" \
    --output "${OUTPUT_DIR}" \
    --smooth-window 5 \
    --min-distance 0.2 \
    --prominence 0.1 \
    --margin 2 \
    --classify-chunk 10 \
    --vlm-model "qwen/qwen3.7-flash"
