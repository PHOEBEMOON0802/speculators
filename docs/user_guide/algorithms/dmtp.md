# D-MTP

D-MTP lets a frozen pretrained MTP module start each draft block while the [DFlash](dflash.md) backbone continues it in parallel. Standard block-parallel drafting predicts the first block token from mask placeholders alone, and most rejected blocks die at that first position; a native MTP head is far better at it. D-MTP keeps the MTP module frozen (preserving its pretrained first-token ability), uses it to propose T1 serially, and lets DFlash predict the remaining candidates in one pass conditioned on T1 ([technical blog](https://sylvan820.github.io/D-MTP/); paper and official code to follow).

## How It Works

### Block Layout

Training uses the two-anchor layout `[A, T1, MASK x (B-2)]` (reported equivalent to the single-anchor variant): slot 0 is the committed anchor as in DFlash, slot 1 holds T1, and the remaining slots are mask placeholders. Slots 0 and 1 are inputs and receive no draft loss; the continuation slots train with the original position-decayed cross-entropy. With `block_size` 8 the verifier checks 7 candidates (T1 + 6 DFlash) plus the bonus token.

### Token/Hidden Fusion

T1's slot embedding is replaced by

```
e~_T1 = W_f [ RMS_e(e_T1) ; RMS_m(h_MTP) ]
```

where `h_MTP` is the MTP hidden state that *predicted* T1 (not one computed after consuming it) and each input has its own learned RMSNorm before a `2d -> d` projection. During training T1 is the ground-truth next token (teacher forcing) and `h_MTP` is computed on the fly by one frozen-MTP pass over the packed sequence -- the batch already carries everything needed (`verifier_last_hidden_states` + token embeddings), so the data pipeline is unchanged.

### What Trains

Only the DFlash layers and the fusion module. The verifier, its native MTP block (extracted from the verifier's `mtp.*` keys at model build), the embeddings, and the LM head stay frozen.

## Key Parameters

| Parameter                    | Default           | Description                                                      |
| ---------------------------- | ----------------- | ----------------------------------------------------------------- |
| `--block-size`               | 8                 | Bonus + T1 + (B-2) DFlash candidates                              |
| `--num-layers`               | 5                 | Draft decoder layers                                              |
| `--loss-fn`                  | `ce`              | Applied to the DFlash continuation slots                          |
| `--per-position-loss-weight` | `fixed-exp-decay` | Original DFlash decay (blog recipe); D-PACE available via `dpace` |

All other DFlash parameters apply unchanged. **Requirement:** the verifier must ship native MTP weights (`mtp.*` keys -- Qwen3-Next, Qwen3.5, DeepSeek-V3-style). For targets without one, first train an MTP head with `--speculator-type mtp`.

The training-time `eal` metric measures the DFlash continuation (slots 2+) conditioned on T1; T1's own acceptance is governed by the frozen MTP and is an inference-side quantity.

## D-MTP2 and Warm-Starting

**D-MTP2** (`--speculator-type dmtp2`) pairs the same frozen-MTP first-token path with the [DFlash2](dflash2.md) backbone (local dynamic convolutions + predecessor-conditioned candidate selector). The two mechanisms are complementary: D-MTP fixes the first token, DFlash2 improves within-block decay. Its training defaults mirror dflash2 (KL loss, fixed exponential decay, bidirectional sliding-window blocks); all dflash and dflash2 flags apply.

Because the D-MTP trunk *is* the DFlash trunk (and D-MTP2's is the DFlash2 trunk), a trained DFlash-family checkpoint can warm-start the corresponding D-MTP model — only the fusion module starts fresh and the MTP block comes from the verifier:

```bash
# dflash -> dmtp (every trunk tensor carries over)
python -m speculators.convert.dmtp.warmstart <dflash_ckpt> \
    --verifier <mtp-capable-verifier> --output ./dmtp_warmstart

# dflash2 -> dmtp2 (convolutions and selector carry over too)
python -m speculators.convert.dmtp.warmstart <dflash2_ckpt> \
    --verifier <mtp-capable-verifier> --output ./dmtp2_warmstart
```

Then fine-tune with `--speculator-type dmtp[2] --from-pretrained <output>`. The verifier must be the one the source draft was trained against. `--identity-fusion-init` initializes the fusion projection to `[I; 0]` so slot 1 starts as the T1 token embedding — gentler on a pretrained trunk than a random fusion. A `dflash2 -> dmtp` conversion is also supported (`--target-type dmtp`), dropping the convolutions and selector.

## Pretrained Models

None released yet. To train your own, see `examples/train/dmtp_qwen3_5_9b_ultrachat_online_5k.sh`.

## Research & Citation

- D-MTP: [technical blog](https://sylvan820.github.io/D-MTP/) (Shen, 2026)
- DFlash: [arXiv:2602.06036](https://arxiv.org/abs/2602.06036)

## See Also

- [DFlash](dflash.md) -- The backbone D-MTP extends
- [MTP](mtp.md) -- Finetuning the native MTP head itself
- [Train a Speculator](../tutorials/train.md) -- Step-by-step training guide
