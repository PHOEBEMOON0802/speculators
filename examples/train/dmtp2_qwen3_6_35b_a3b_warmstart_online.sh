#!/bin/bash
# Online D-MTP2 Training -- Qwen3.6-35B-A3B, warm-started from a trained
# DFlash2 checkpoint.
#
# Pipeline (4 steps):
#   0. Warm-start: convert the DFlash2 draft into a D-MTP2 checkpoint
#      (python -m speculators.convert.dmtp.warmstart). External z-lab-style
#      DFlash2 checkpoints (config.json carries dflash_config instead of
#      speculators_model_type) are auto-converted first.
#   1. prepare-data: tokenize the dataset.
#   2. launch vLLM: serves the verifier and streams aux hidden states.
#   3. train: finetune from the warm-started checkpoint via --from-pretrained.
#
# With --from-pretrained, the model architecture (draft layers, block_size,
# convolutions, selector rank/top-k, aux layer ids, mask token) comes from the
# warm-started checkpoint -- NOT from CLI flags. Only runtime training knobs
# (epochs, lr, max-anchors, selector-loss-alpha, ...) and the data-alignment
# flag --target-layer-ids apply here.
#
# Hard requirements:
#   1. The verifier must ship native mtp.* weights (Qwen3.6-35B-A3B does:
#      text_config.mtp_num_hidden_layers = 1).
#   2. TARGET_LAYER_IDS must equal the aux layer ids baked into the DFlash2
#      checkpoint, in speculators convention (z-lab target_layer_ids + 1).
#      For incoai/Qwen3.6-35B-A3B-DFlash2: [1,6,11,16,22,27,32,37] + 1
#      -> "2 7 12 17 23 28 33 38". vLLM adds the last layer implicitly.
#   3. DFlash2/D-MTP2 needs the FULL verifier vocabulary:
#      DRAFT_VOCAB_SIZE = verifier vocab_size = 248320.
#
# Usage: copy, adjust the configuration block, then:
#   bash examples/train/dmtp2_qwen3_6_35b_a3b_warmstart_online.sh

set -euo pipefail

# ============ Configuration ============
MODEL="/data/nfs/model/Qwen3.6-35B-A3B"            # verifier (native MTP)
DFLASH2_CKPT="/data/nfs/model/Qwen3.6-35B-A3B-DFlash2"  # trained DFlash2 draft
RUN_DIR="/efs/c00958241/train/dmtp/qwen3.6-35b-a3b"
WARMSTART_CKPT="$RUN_DIR/dmtp2_warmstart"
OUTPUT_DIR="$RUN_DIR/dmtp2_online"

DATASET="hf:inference-optimization/speculators-ci-datasets:tutorial_regen"
VLLM_PORT=8000
MAX_SAMPLES=5000
SEQ_LENGTH=8192
EPOCHS=3
LR=3e-4

# Must match the DFlash2 checkpoint (see header). Passed to launch_vllm.py and
# speculators.train so the streamed hidden states align with the draft's fc.
TARGET_LAYER_IDS="2 7 12 17 23 28 33 38"
DRAFT_VOCAB_SIZE=248320
MAX_ANCHORS=3072

# GPU assignments (online training needs separate GPUs for vLLM and training)
VLLM_GPUS="0,1,2,3"
VLLM_DP_SIZE=1
VLLM_TP_SIZE=4
TRAIN_GPUS="4,5,6,7"
NUM_TRAIN_GPUS=4
# =======================================

echo "=== Step 0: Warm-starting D-MTP2 from DFlash2 ==="
if [[ -f "$WARMSTART_CKPT/config.json" ]]; then
    echo "Warm-start checkpoint already exists at $WARMSTART_CKPT -- skipping."
else
    python -m speculators.convert.dmtp.warmstart "$DFLASH2_CKPT" \
        --verifier "$MODEL" \
        --output "$WARMSTART_CKPT"
fi

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
    -- --data-parallel-size "$VLLM_DP_SIZE" \
    --tensor-parallel-size "$VLLM_TP_SIZE" \
    --port "$VLLM_PORT" &
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

echo "=== Step 3: Training D-MTP2 from the warm start ==="
CUDA_VISIBLE_DEVICES="$TRAIN_GPUS" torchrun \
    --standalone --nproc_per_node "$NUM_TRAIN_GPUS" \
    -m speculators.train \
    --verifier-name-or-path "$MODEL" \
    --data-path "$OUTPUT_DIR" \
    --vllm-endpoint "http://localhost:${VLLM_PORT}/v1" \
    --save-path "$OUTPUT_DIR/checkpoints" \
    --speculator-type dmtp2 \
    --from-pretrained "$WARMSTART_CKPT" \
    --draft-vocab-size "$DRAFT_VOCAB_SIZE" \
    --epochs "$EPOCHS" \
    --lr "$LR" \
    --total-seq-len "$SEQ_LENGTH" \
    --max-anchors "$MAX_ANCHORS" \
    --target-layer-ids $TARGET_LAYER_IDS \
    --on-missing generate

echo "Done. Checkpoints saved to $OUTPUT_DIR/checkpoints/"
