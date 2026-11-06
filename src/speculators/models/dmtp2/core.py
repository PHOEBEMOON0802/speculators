import logging
from typing import ClassVar

import torch
from transformers import PretrainedConfig

from speculators.losses import resolve_loss_config
from speculators.model import SpeculatorModel
from speculators.models.dflash2.core import DFlash2DraftModel
from speculators.models.dflash2.metrics import selector_training_candidates
from speculators.models.dmtp.core import MTPFirstTokenMixin
from speculators.models.dmtp2.config import DMTP2SpeculatorConfig
from speculators.models.dmtp2.metrics import compute_metrics
from speculators.models.utils import conditional_torch_compile

logger = logging.getLogger(__name__)

__all__ = [
    "DMTP2DraftModel",
]


@SpeculatorModel.register("dmtp2")
class DMTP2DraftModel(MTPFirstTokenMixin, DFlash2DraftModel):
    """DFlash2 backbone whose block starts from a frozen-MTP-proposed token.

    Combines D-MTP's first-token path (frozen MTP proposes T1, fused into
    block slot 1) with DFlash2's local dynamic convolutions and
    predecessor-conditioned candidate selector for the continuation slots.
    The two mechanisms are complementary: D-MTP fixes the first token while
    DFlash2 improves within-block decay. Slots 0 (anchor) and 1 (T1) receive
    no draft loss; the unary and selector objectives cover slots 2..B-1.
    """

    config_class: ClassVar[type[DMTP2SpeculatorConfig]] = DMTP2SpeculatorConfig  # type: ignore[misc,assignment]
    _no_split_modules = ["Qwen3DFlash2DecoderLayer"]

    def __init__(self, config: DMTP2SpeculatorConfig) -> None:
        self._validate_dmtp_config(config)
        super().__init__(config=config)
        self._init_mtp_first_token(config)

    @classmethod
    def from_training_args(
        cls,
        verifier_config: PretrainedConfig,
        t2d: torch.Tensor | None = None,
        d2t: torch.Tensor | None = None,
        **kwargs,
    ) -> "DMTP2DraftModel":
        """Create a D-MTP2 model; the verifier must carry native mtp.* weights."""
        config = DMTP2SpeculatorConfig(
            **cls._build_base_config_kwargs("dmtp2", verifier_config, **kwargs),
            conv_kernel_size=kwargs.get("conv_kernel_size", 2),
            conv_group_size=kwargs.get("conv_group_size", 16),
            selector_rank=kwargs.get("selector_rank", 256),
            selector_top_k=kwargs.get("selector_top_k", 16),
            mtp_transformer_config=cls._resolve_mtp_transformer_config(kwargs),
        )
        model = cls(config=config)
        cls._load_frozen_mtp_weights(model, kwargs["verifier_name_or_path"])
        model.load_vocab_mappings(t2d, d2t)
        model.load_verifier_weights()
        return model

    @staticmethod
    def get_trainer_kwargs(**kwargs) -> tuple[dict, dict]:
        """Resolve the unary and selector objectives used during training."""
        loss_config = resolve_loss_config(
            kwargs["loss_fn"], kwargs.get("loss_implementation", "fused")
        )
        shared = {
            "loss_config": loss_config,
            "gamma": kwargs.get("dflash_decay_gamma", 4.0),
            "max_anchors": kwargs.get("max_anchors", 512),
            "per_position_loss_weight": kwargs.get(
                "per_position_loss_weight", "fixed-exp-decay"
            ),
            "dpace_alpha": kwargs.get("dpace_alpha", 0.5),
            "selector_loss_alpha": kwargs.get("selector_loss_alpha", 1.0),
            "selector_miss_policy": kwargs.get("selector_miss_policy", "replace"),
        }
        return dict(shared), dict(shared)

    @conditional_torch_compile
    def forward(
        self,
        hidden_states: torch.Tensor,  # shape: [1, total_seq_len,num_hidden*hidden_size]
        input_ids: torch.Tensor,  # shape: [1, total_seq_len]
        loss_mask: torch.Tensor,  # shape: [1, total_seq_len]
        verifier_last_hidden_states: torch.Tensor,  # shape: [1, total_seq_len, hidden_size] # noqa: E501
        document_ids: torch.Tensor,  # shape: [1, total_seq_len]
        position_ids: torch.Tensor | None = None,  # shape: [1, total_seq_len]
        loss_config=None,
        gamma: float = 4.0,
        max_anchors: int = 512,
        selector_loss_alpha: float = 1.0,
        per_position_loss_weight: str = "fixed-exp-decay",
        dpace_alpha: float = 0.5,
        selector_miss_policy: str = "replace",
        **kwargs,
    ):
        hidden, unary_logits, targets, aligned_loss_mask, block_indices = (
            self._backbone_forward(
                hidden_states,
                input_ids,
                loss_mask,
                verifier_last_hidden_states,
                document_ids,
                position_ids,
                max_anchors=max_anchors,
                **kwargs,
            )
        )
        # Slot 1 holds T1, a known input: no draft loss (slot 0, the anchor,
        # is already zeroed by the base sample_from_anchor=False path).
        aligned_loss_mask = aligned_loss_mask.clone()
        aligned_loss_mask[:, 1 :: self.block_size] = 0

        predecessor_ids = self._predecessor_ids(input_ids, block_indices)

        target_ids = targets.argmax(dim=-1)
        candidate_ids = unary_logits.topk(self.candidate_selector.top_k, dim=-1).indices
        training_candidate_ids, target_positions, contains_target = (
            selector_training_candidates(
                candidate_ids, target_ids, miss_policy=selector_miss_policy
            )
        )
        candidate_logits = self.candidate_selector.score_candidates(
            unary_logits,
            hidden,
            predecessor_ids.reshape(1, -1),
            training_candidate_ids,
        )

        loss, metrics = compute_metrics(
            unary_logits=unary_logits,
            targets=targets,
            training_candidate_ids=training_candidate_ids,
            candidate_logits=candidate_logits,
            target_positions=target_positions,
            contains_target=contains_target,
            loss_mask=aligned_loss_mask,
            block_size=self.block_size,
            top_k=self.candidate_selector.top_k,
            loss_config=loss_config,
            gamma=gamma,
            selector_loss_alpha=selector_loss_alpha,
            per_position_loss_weight=per_position_loss_weight,
            dpace_alpha=dpace_alpha,
            selector_miss_policy=selector_miss_policy,
        )
        return None, loss, metrics
