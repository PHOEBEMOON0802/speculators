# DCore

DCore combines the [DFlash](dflash.md) block-parallel draft backbone with DFlare's target-layer context mixture and trains it with the D-PACE position-weighted cross-entropy objective. It was identified in a controlled comparison study ([arXiv:2610.24993](https://arxiv.org/abs/2610.24993), App. G) as the strongest low-overhead recipe: both changes improve acceptance length without adding measurable drafting latency, since the fusion reduces to a few scalar weights at inference and the objective changes only the training weights.

The draft model subclasses DFlash, so the anchored-block training scheme, data pipeline, and verifier support are unchanged. Weight keys are aligned with the released [`ernie-research/DCore-*`](https://huggingface.co/collections/ernie-research/dcore-6ad82deaee94a91b30e1f1c6) checkpoints.

## How It Works

### Adaptive Layer Fusion (from DFlare)

DFlash concatenates hidden states from a few target layers and projects them through one shared FC layer, so every draft layer conditions on the same fused context. DCore instead keeps a learnable `layer_fusion_weights [D, T]` matrix: each draft layer `i` applies a softmax over its own row to combine `T=9` target layers into a dedicated context, followed by a shared RMSNorm (`hidden_norm`). Each draft layer gets a differentiated view of the target model at the cost of `D x T` scalars; at inference the softmax can be precomputed, making fusion a scalar-weighted sum ([arXiv:2606.02091](https://arxiv.org/abs/2606.02091), Sec. 4.1).

### Heterogeneous KV Projections (from DFlare)

DFlash projects the injected target context and the draft block tokens through the same K/V matrices. DCore gives the target context dedicated `k_proj_target` / `v_proj_target` projections, so semantic knowledge from the target and the evolving block predictions occupy independent representational subspaces in attention (Sec. 4.2).

### D-PACE Objective

Instead of a fixed exponential position decay, each block position's cross-entropy is weighted by a detached, differentiable proxy of the probability that the draft survives to that position, computed from the draft's own smoothed confidences ([arXiv:2605.18810](https://arxiv.org/abs/2605.18810)). This is the training default for `--speculator-type dcore` (and `dflash`), via `--per-position-loss-weight dpace` with `--loss-fn ce`.

## Key Parameters

| Parameter                    | Default            | Description                                                              |
| ---------------------------- | ------------------ | ------------------------------------------------------------------------ |
| `--target-layer-ids`         | 9 uniform in [1, n-3] | Target layers supplying hidden states; `1 5 9 13 17 21 25 29 33` for Qwen3-8B. Must match the vLLM datagen server |
| `--block-size`               | 8                  | Draft block size (budget); matches the released Budget8 checkpoints. Use 16 for Budget16 |
| `--num-layers`               | 5                  | Draft decoder layers                                                     |
| `--per-position-loss-weight` | `dpace`            | D-PACE position weighting (requires `--loss-fn ce`)                      |
| `--dpace-alpha`              | 0.5                | D-PACE confidence smoothing constant                                     |

All other DFlash parameters (`--max-anchors`, `--draft-vocab-size`, ...) apply unchanged. To stay weight-compatible with the released checkpoints, train with the full verifier vocabulary (the frozen target LM head).

Note: 9 target layers triple the hidden-state volume moved from the vLLM server compared with the 3-layer DFlash default; budget datagen bandwidth and disk accordingly.

## Pretrained Models

Released by ERNIE Research: [DCore collection](https://huggingface.co/collections/ernie-research/dcore-6ad82deaee94a91b30e1f1c6) (e.g. `DCore-Qwen3-8B-Budget8`). To train your own, see `examples/train/dcore_qwen3_8b_ultrachat_online_5k.sh`.

## Research & Citation

- DCore recipe: [When Does Better Acceptance Pay? A Controlled Study of Speculative Decoding](https://arxiv.org/abs/2610.24993)
- Context mixture & heterogeneous KV: [DFlare](https://arxiv.org/abs/2606.02091)
- Objective: [D-PACE](https://arxiv.org/abs/2605.18810)

## See Also

- [DFlash](dflash.md) -- The base backbone DCore extends
- [Loss Functions](../loss_functions.md) -- D-PACE and the other supported objectives
- [Train a Speculator](../tutorials/train.md) -- Step-by-step training guide
