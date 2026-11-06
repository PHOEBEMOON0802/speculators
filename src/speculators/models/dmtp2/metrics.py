"""Metrics and loss for the D-MTP2 draft model.

Mirrors ``speculators.models.dflash2.metrics.compute_metrics`` (unary DFlash
objective + selector objective) with the D-MTP block layout: slot 0 is the
anchor and slot 1 the MTP-proposed T1 -- both inputs with no draft loss.
Accuracy, realized-selector-path EAL, and the oracle accepted length therefore
start at slot 2; reusing the DFlash2 metrics with a zeroed slot-1 mask would
zero every block's accepted run.
"""

from typing import Any

import torch

from speculators.losses import LossConfig
from speculators.models.dflash2.metrics import compute_selector_loss
from speculators.models.dmtp.metrics import (
    _FIRST_DRAFT_SLOT,
)
from speculators.models.dmtp.metrics import (
    compute_metrics as compute_unary_metrics,
)
from speculators.models.metrics import compute_accepted_length_counts

__all__ = [
    "compute_metrics",
]


def compute_metrics(
    unary_logits: torch.Tensor,  # [1, num_anchors*block_size, draft_vocab_size]
    targets: torch.Tensor,  # [1, num_anchors*block_size, draft_vocab_size]
    training_candidate_ids: torch.Tensor,  # [1, num_anchors*block_size, top_k]
    candidate_logits: torch.Tensor,  # [1, num_anchors*block_size, top_k]
    target_positions: torch.Tensor,  # [1, num_anchors*block_size]
    contains_target: torch.Tensor,  # [1, num_anchors*block_size]
    loss_mask: torch.Tensor,  # [1, num_anchors*block_size], slots 0/1 zeroed
    block_size: int,
    top_k: int,
    *,
    loss_config: LossConfig | None = None,
    gamma: float = 4.0,
    selector_loss_alpha: float = 1.0,
    per_position_loss_weight: str = "fixed-exp-decay",
    dpace_alpha: float = 0.5,
    selector_miss_policy: str = "replace",
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Combine the unary D-MTP objective with the DFlash2 selector objective.

    ``loss_mask`` must already have block slots 0 and 1 zeroed; both the unary
    and the selector losses then cover exactly the ``block_size - 2`` DFlash2
    continuation positions.
    """
    if selector_miss_policy not in {"replace", "strict"}:
        raise ValueError(
            f"Unknown selector miss policy: {selector_miss_policy!r}; "
            "expected 'replace' or 'strict'."
        )

    unary_loss, metrics = compute_unary_metrics(
        unary_logits,
        targets,
        loss_mask,
        block_size,
        gamma=gamma,
        loss_config=loss_config,
        per_position_loss_weight=per_position_loss_weight,
        dpace_alpha=dpace_alpha,
    )

    selector_valid_mask = contains_target if selector_miss_policy == "strict" else None
    selector_loss = compute_selector_loss(
        candidate_logits,
        target_positions,
        loss_mask,
        block_size,
        gamma=gamma,
        per_position_loss_weight=per_position_loss_weight,
        dpace_alpha=dpace_alpha,
        sample_from_anchor=False,
        selector_valid_mask=selector_valid_mask,
    )
    loss = unary_loss + selector_loss_alpha * selector_loss

    one = torch.ones((), device=unary_logits.device)
    metrics["unary_loss_sum"] = unary_loss.detach().clone()
    metrics["unary_loss_total"] = one
    metrics["selector_loss_sum"] = selector_loss.detach().clone()
    metrics["selector_loss_total"] = one.clone()
    metrics["loss_sum"] = loss.detach().clone()
    metrics["loss_total"] = one.clone()

    with torch.no_grad():
        target_ids = targets.argmax(dim=-1)
        valid = loss_mask.to(torch.bool)
        valid_float = valid.to(unary_logits.dtype)
        valid_total = valid_float.sum()

        metrics[f"unary_candidate_recall_at_{top_k}_sum"] = (
            contains_target.to(valid_float.dtype) * valid_float
        ).sum()
        metrics[f"unary_candidate_recall_at_{top_k}_total"] = valid_total

        # Process rows in chunks so logsumexp does not materialize a
        # full-vocab fp32 copy of the targets and its exp workspace.
        target_log_normalizer = torch.empty(
            targets.shape[:-1], device=targets.device, dtype=torch.float32
        )
        for start in range(0, targets.shape[1], 1024):
            torch.logsumexp(
                targets[:, start : start + 1024].float(),
                dim=-1,
                out=target_log_normalizer[:, start : start + 1024],
            )
        candidate_target_logits = targets.gather(-1, training_candidate_ids).float()
        candidate_mass = torch.exp(
            torch.logsumexp(candidate_target_logits, dim=-1) - target_log_normalizer
        )
        metrics[f"unary_candidate_target_mass_at_{top_k}_sum"] = (
            candidate_mass * valid_float
        ).sum()
        metrics[f"unary_candidate_target_mass_at_{top_k}_total"] = valid_total.clone()

        teacher_forced_ids = training_candidate_ids.gather(
            -1, candidate_logits.detach().argmax(dim=-1, keepdim=True)
        ).squeeze(-1)
        serving_valid = valid_float * contains_target.to(valid_float.dtype)
        serving_total = serving_valid.sum()
        metrics["teacher_forced_selector_acc_sum"] = (
            teacher_forced_ids.eq(target_ids).to(valid_float.dtype) * serving_valid
        ).sum()
        metrics["teacher_forced_selector_acc_total"] = serving_total

        num_blocks = unary_logits.shape[1] // block_size
        contains_target_blocks = contains_target.view(num_blocks, block_size)
        valid_blocks = valid.view(num_blocks, block_size)

        # Realized selector-path EAL over the DFlash2 continuation (slots 2+),
        # conditioned on T1 being given. Gate on the original unary candidate
        # set: selector training must not make injected gold tokens appear
        # available at inference.
        selector_correct = teacher_forced_ids.eq(target_ids) & contains_target
        eal_sum, eal_total = compute_accepted_length_counts(
            selector_correct.view(num_blocks, block_size)[:, _FIRST_DRAFT_SLOT:],
            valid_blocks[:, _FIRST_DRAFT_SLOT:],
        )
        metrics["eal_sum"] = eal_sum
        metrics["eal_total"] = eal_total

        oracle_alive = torch.ones(
            num_blocks, dtype=torch.bool, device=unary_logits.device
        )
        oracle_accepted_length = torch.ones(
            num_blocks, dtype=torch.float32, device=unary_logits.device
        )
        for position in range(_FIRST_DRAFT_SLOT, block_size):
            oracle_alive = (
                oracle_alive
                & valid_blocks[:, position]
                & contains_target_blocks[:, position]
            )
            oracle_accepted_length += oracle_alive.to(oracle_accepted_length.dtype)

        block_valid = valid_blocks[:, _FIRST_DRAFT_SLOT:].any(dim=-1)
        block_total = block_valid.sum().to(torch.float32)
        metrics[f"unary_top_{top_k}_oracle_accepted_length_sum"] = (
            oracle_accepted_length * block_valid
        ).sum()
        metrics[f"unary_top_{top_k}_oracle_accepted_length_total"] = block_total

    return loss, metrics
