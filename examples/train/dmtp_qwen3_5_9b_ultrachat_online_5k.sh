#!/bin/bash
# Online D-MTP Training Script -- Qwen3.5-9B (native MTP verifier)
#
# D-MTP (https://sylvan820.github.io/D-MTP/): a frozen pretrained MTP module
# proposes the first draft token (T1) of each block; the DFlash backbone
# continues the block in one parallel pass. Training updates only the DFlash
# layers and a small token/hidden fusion module -- the verifier and its native
# MTP block stay frozen. The MTP block weights are extracted automatically
# from the verifier's mtp.* keys, so the verifier MUST ship native MTP
# weights (Qwen3-Next / Qwen3.5 / DeepSeek-V3-style). For targets without
# one, first train an MTP head with --speculator-type mtp.
#
# Defaults for --speculator-type dmtp (set explicitly below for reference):
# 5 draft layers, block_size 8 (= bonus + T1 + 6 DFlash candidates), CE loss
# with the original fixed exponential position decay.
#
# Usage: Copy this script, modify the configuration variables below, then run:
#   bash examples/train/dmtp_qwen3_5_9b_ultrachat_online_5k.sh

set -euo pipefail

# ============ Configuration ============
MODEL="Qwen/Qwen3.5-9B"   # must contain native mtp.* weights
DATASET="hf:inference-optimization/speculators-ci-datasets:tutorial_regen"
OUTPUT_DIR="./output/dmtp_qwen3_5_9b_ultrachat"
VLLM_PORT=8000
MAX_SAMPLES=5000
SEQ_LENGTH=8192
EPOCHS=3
LR=3e-4

SPECULATOR_TYPE="dmtp"
BLOCK_SIZE=8
MAX_ANCHORS=3072
NUM_LAYERS=5
DRAFT_VOCAB_SIZE=32000
TARGET_LAYER_IDS="2 18 33"  # Must match vLLM's eagle_aux_hidden_state_layer_ids

# GPU assignments (online training needs separate GPUs for vLLM and training)
VLLM_GPUS="0,1"
TRAIN_GPUS="2,3"
NUM_TRAIN_GPUS=2
# =======================================

echo "=== Step 1: Preparing data ==="
speculators prepare-data \
    --model "$MODEL" \
    --data "$DATASET" \
    --output "$OUTPUT_DIR" \
    --max-samples "$MAX_SAMPLES" \
    --seq-length "$SEQ_LENGTH"

echo "=== Step 2: Launching vLLM server ==="
CUDA_VISIBLE_DEVICES="$VLLM_GPUS" python scripts/launch_vllm.py "$MODEL" \
    --target-layer-ids $TARGET_LAYER_IDS \
    -- --data-parallel-size 2 --port "$VLLM_PORT" &
VLLM_PID=$!

cleanup() {
    echo "Stopping vLLM server..."
    kill "$VLLM_PID" 2>/dev/null || true
    wait "$VLLM_PID" 2>/dev/null || true
}
trap cleanup EXIT

echo "Waiting for vLLM server to be ready..."
until curl -sf "http://localhost:${VLLM_PORT}/health" > /dev/null 2>&1; do
    sleep 2
done
echo "vLLM server ready."

echo "=== Step 3: Training ==="
CUDA_VISIBLE_DEVICES="$TRAIN_GPUS" torchrun \
    --standalone --nproc_per_node "$NUM_TRAIN_GPUS" \
    -m speculators.train \
    --verifier-name-or-path "$MODEL" \
    --data-path "$OUTPUT_DIR" \
    --vllm-endpoint "http://localhost:${VLLM_PORT}/v1" \
    --save-path "$OUTPUT_DIR/checkpoints" \
    --draft-vocab-size "$DRAFT_VOCAB_SIZE" \
    --epochs "$EPOCHS" \
    --lr "$LR" \
    --total-seq-len "$SEQ_LENGTH" \
    --speculator-type "$SPECULATOR_TYPE" \
    --block-size "$BLOCK_SIZE" \
    --max-anchors "$MAX_ANCHORS" \
    --num-layers "$NUM_LAYERS" \
    --target-layer-ids $TARGET_LAYER_IDS \
    --on-missing generate

echo "Done. Checkpoints saved to $OUTPUT_DIR/checkpoints/"
