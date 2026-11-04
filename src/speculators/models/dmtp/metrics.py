"""Metrics and loss for the D-MTP draft model.

Mirrors ``speculators.models.dflash.metrics.compute_metrics`` with one layout
difference: block slot 0 is the anchor (as in DFlash) and slot 1 is the
MTP-proposed T1 -- a known input that receives no draft loss. Accuracy and
expected-accepted-length metrics therefore start at slot 2, measuring the
DFlash continuation conditioned on T1; reusing the DFlash metrics with a
zeroed slot-1 mask would instead zero every block's accepted run.
"""

from functools import partial
from typing import Any

import torch

from speculators.losses import (
    LossConfig,
    compound_loss,
    dflash_loss_decay,
    dpace_loss_decay,
    kl_div_loss,
)
from speculators.models.metrics import (
    compute_accepted_length_counts,
    compute_accuracy_multi_step,
)

_DEFAULT_LOSS_CONFIG: LossConfig = {"kl_div": (kl_div_loss, 1.0)}

# Slots 0 (anchor) and 1 (T1) are inputs, not predictions.
_FIRST_DRAFT_SLOT = 2


def compute_metrics(
    logits: torch.Tensor,  # shape: [1, num_anchors*block_size, draft_vocab_size]
    targets: torch.Tensor,  # shape: [1, num_anchors*block_size, draft_vocab_size]
    loss_mask: torch.Tensor,  # shape: [1, num_anchors*block_size], slots 0/1 zeroed
    block_size: int,
    gamma: float = 4.0,
    loss_config: LossConfig | None = None,
    per_position_loss_weight: str = "fixed-exp-decay",
    dpace_alpha: float = 0.5,
) -> tuple[torch.Tensor, dict]:
    """Compute D-MTP training loss and DFlash-continuation metrics.

    ``loss_mask`` must already have block slots 0 and 1 zeroed; the loss then
    covers exactly the ``block_size - 2`` DFlash continuation positions.
    """
    if loss_config is None:
        loss_config = _DEFAULT_LOSS_CONFIG
    seq_len = logits.shape[1]
    pos_idx = torch.arange(seq_len, device=logits.device) % block_size
    pos_idx = pos_idx.unsqueeze(0)  # shape: [1, T]

    if per_position_loss_weight == "dpace":
        decay_fn = partial(
            dpace_loss_decay,
            loss_mask=loss_mask,
            block_size=block_size,
            dpace_alpha=dpace_alpha,
        )
    else:
        # Reuse the DFlash decay (weight 0 at slot 0, 1 at slot 1, decaying
        # after); slot 1's weight is irrelevant since its mask is zero.
        decay_fn = partial(dflash_loss_decay, gamma=gamma, sample_from_anchor=False)

    loss, term_losses = compound_loss(
        logits,
        targets,
        loss_mask,
        pos_idx,
        loss_config=loss_config,
        decay_fn=decay_fn,
    )

    pred_ids = torch.argmax(logits, dim=-1)
    target_ids = torch.argmax(targets, dim=-1)

    correct_per_pos, total_per_pos = compute_accuracy_multi_step(
        pred_ids, target_ids, loss_mask, pos_idx, block_size
    )

    ones = torch.tensor(1.0, device=logits.device)
    metrics: dict[str, Any] = {}
    metrics["loss_sum"] = loss.detach().clone()
    metrics["loss_total"] = ones
    for term_name, term_val in term_losses.items():
        metrics[f"{term_name}_sum"] = term_val
        metrics[f"{term_name}_total"] = ones.clone()

    metrics["full_acc_sum"] = correct_per_pos[_FIRST_DRAFT_SLOT:].sum()
    metrics["full_acc_total"] = total_per_pos[_FIRST_DRAFT_SLOT:].sum()
    for pos in range(_FIRST_DRAFT_SLOT, block_size):
        metrics[f"position_{pos}_acc_sum"] = correct_per_pos[pos]
        metrics[f"position_{pos}_acc_total"] = total_per_pos[pos]

    # Accepted length of the DFlash continuation (slots 2..B-1), conditioned
    # on T1 being given; T1's own acceptance is governed by the frozen MTP.
    eal_sum, eal_total = compute_accepted_length_counts(
        (pred_ids == target_ids).reshape(-1, block_size)[:, _FIRST_DRAFT_SLOT:],
        loss_mask.to(torch.bool).reshape(-1, block_size)[:, _FIRST_DRAFT_SLOT:],
    )
    metrics["eal_sum"] = eal_sum
    metrics["eal_total"] = eal_total
    return loss, metrics
