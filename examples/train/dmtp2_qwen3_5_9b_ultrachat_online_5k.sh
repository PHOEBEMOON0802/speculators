#!/bin/bash
# Online D-MTP2 Training Script -- Qwen3.5-9B (native MTP verifier)
#
# D-MTP2 = D-MTP's frozen-MTP first-token path + the DFlash2 backbone
# (local dynamic convolutions + predecessor-conditioned candidate selector).
# The frozen MTP block proposes T1; DFlash2 continues the block in one
# parallel pass. Training updates the DFlash2 layers, the candidate
# selector, and the small token/hidden fusion module -- the verifier and
# its native MTP block stay frozen.
#
# Hard requirements:
#   1. The verifier MUST ship native mtp.* weights (Qwen3-Next / Qwen3.5 /
#      DeepSeek-V3-style). For targets without one, first train an MTP head
#      with --speculator-type mtp.
#   2. The DFlash2 backbone requires the FULL verifier vocabulary (the
#      candidate selector reranks over the frozen verifier LM head), so
#      DRAFT_VOCAB_SIZE must equal the verifier's vocab_size.
#
# Defaults for --speculator-type dmtp2 (mirroring dflash2, its warm-start
# source): 5 draft layers, block_size 8 (= bonus + T1 + 6 candidates),
# KL loss, fixed exponential position decay, bidirectional sliding-window
# blocks. All dflash-group (--block-size, --max-anchors, --dflash-decay-gamma,
# --per-position-loss-weight, --dpace-alpha) and dflash2-group
# (--conv-kernel-size, --conv-group-size, --selector-rank, --selector-top-k,
# --selector-loss-alpha, --selector-miss-policy) flags apply.
#
# Usage: Copy this script, modify the configuration variables below, then run:
#   bash examples/train/dmtp2_qwen3_5_9b_ultrachat_online_5k.sh

set -euo pipefail

# ============ Configuration ============
MODEL="Qwen/Qwen3.5-9B"   # must contain native mtp.* weights
DATASET="hf:inference-optimization/speculators-ci-datasets:tutorial_regen"
OUTPUT_DIR="./output/dmtp2_qwen3_5_9b_ultrachat"
VLLM_PORT=8000
MAX_SAMPLES=5000
SEQ_LENGTH=8192
EPOCHS=3
LR=3e-4

SPECULATOR_TYPE="dmtp2"
BLOCK_SIZE=8              # bonus + T1 + 6 DFlash2 candidates
MAX_ANCHORS=3072
NUM_LAYERS=5
DRAFT_VOCAB_SIZE=151936   # MUST equal the verifier vocab (DFlash2 requirement)
TARGET_LAYER_IDS="2 18 33"  # Must match vLLM's eagle_aux_hidden_state_layer_ids

# DFlash2 selector/convolution knobs (defaults shown; tune as needed)
SELECTOR_RANK=256
SELECTOR_TOP_K=16
CONV_KERNEL_SIZE=2
CONV_GROUP_SIZE=16

# Optional warm start from a trained DFlash2 checkpoint on the SAME verifier:
# the convolutions and selector carry over; only the fusion module starts
# fresh. Leave empty to train from scratch.
#   python -m speculators.convert.dmtp.warmstart <dflash2_ckpt> \
#       --verifier "$MODEL" --output ./dmtp2_warmstart --identity-fusion-init
WARMSTART_CKPT=""         # e.g. "./dmtp2_warmstart"

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
EXTRA_ARGS=()
if [[ -n "$WARMSTART_CKPT" ]]; then
    EXTRA_ARGS+=(--from-pretrained "$WARMSTART_CKPT")
fi

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
    --selector-rank "$SELECTOR_RANK" \
    --selector-top-k "$SELECTOR_TOP_K" \
    --conv-kernel-size "$CONV_KERNEL_SIZE" \
    --conv-group-size "$CONV_GROUP_SIZE" \
    --target-layer-ids $TARGET_LAYER_IDS \
    --on-missing generate \
    "${EXTRA_ARGS[@]}"

echo "Done. Checkpoints saved to $OUTPUT_DIR/checkpoints/"
