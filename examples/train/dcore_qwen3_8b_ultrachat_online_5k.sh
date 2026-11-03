#!/bin/bash
# Online DCore Training Script -- aligned with ernie-research/DCore-Qwen3-8B-Budget8
#
# DCore (arXiv:2610.24993, App. G) = DFlash 5-layer block-parallel backbone
# + DFlare per-draft-layer target fusion and heterogeneous KV projections
#   (arXiv:2606.02091, Sec. 4)
# + D-PACE position-weighted cross-entropy (arXiv:2605.18810).
#
# The settings below reproduce the released Budget8 checkpoint's structure so
# the trained weights are key- and shape-compatible with it:
#   - 5 draft layers, block_size (budget) 8
#   - 9 target context layers [1 5 9 13 17 21 25 29 33] for Qwen3-8B (36 layers)
#   - full verifier vocabulary (frozen target embed + LM head)
#   - D-PACE weighting with CE loss (train.py defaults for --speculator-type dcore)
#
# The dataset below is UltraChat prompts with every assistant turn regenerated
# on-policy by Qwen/Qwen3-8B. 5k samples is a pipeline sanity check, not a
# full recipe; the released checkpoints train on PerfectBlend-scale data.
#
# Usage: Copy this script, modify the configuration variables below, then run:
#   bash examples/train/dcore_qwen3_8b_ultrachat_online_5k.sh

set -euo pipefail

# ============ Configuration ============
MODEL="Qwen/Qwen3-8B"
DATASET="hf:inference-optimization/speculators-ci-datasets:tutorial_regen"
OUTPUT_DIR="./output/dcore_qwen3_8b_ultrachat_200k_regen"
VLLM_PORT=8000
MAX_SAMPLES=5000
SEQ_LENGTH=8192
EPOCHS=5
LR=3e-4

# DCore parameters (aligned with DCore-Qwen3-8B-Budget8)
SPECULATOR_TYPE="dcore"
BLOCK_SIZE=8                  # Budget8; use 16 to align with the Budget16 release
MAX_ANCHORS=3072
NUM_LAYERS=5
# Full verifier vocabulary: the released DCore checkpoints reuse the frozen
# target LM head (Qwen3 vocab). Reduced draft vocab is possible but then the
# weights no longer align with the released checkpoints.
DRAFT_VOCAB_SIZE=151936
# 9 uniform target layers; must match vLLM's eagle_aux_hidden_state_layer_ids.
# (This is also the train.py default for --speculator-type dcore.)
TARGET_LAYER_IDS="1 5 9 13 17 21 25 29 33"

# GPU assignments (online training needs separate GPUs for vLLM and training)
VLLM_GPUS="0,1"
TRAIN_GPUS="2,3"
NUM_TRAIN_GPUS=2
# =======================================

# Step 1: Prepare data
echo "=== Step 1: Preparing data ==="
speculators prepare-data \
    --model "$MODEL" \
    --data "$DATASET" \
    --output "$OUTPUT_DIR" \
    --max-samples "$MAX_SAMPLES" \
    --seq-length "$SEQ_LENGTH"

# Step 2: Launch vLLM server in the background
# NOTE: 9 target layers means 3x the hidden-state transfer volume of the
# 3-layer dflash recipe; reduce concurrency if the server becomes a bottleneck.
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

# Step 3: Train against the live vLLM server
# --loss-fn ce and --per-position-loss-weight dpace are the dcore defaults;
# they are omitted here on purpose (see train.py --help).
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
