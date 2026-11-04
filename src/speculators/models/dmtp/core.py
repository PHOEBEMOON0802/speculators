import logging
from typing import ClassVar

import torch
from torch import nn
from transformers import PretrainedConfig
from transformers.masking_utils import create_causal_mask

from speculators.losses import LossConfig
from speculators.model import SpeculatorModel
from speculators.models.dflash.core import DFlashDraftModel
from speculators.models.dmtp.config import DMTPSpeculatorConfig
from speculators.models.dmtp.metrics import compute_metrics
from speculators.models.dmtp.model_definitions import MTPTokenFusion
from speculators.models.mtp.core import _prepare_mtp_position_ids
from speculators.models.mtp.model_definitions import (
    mtp_model_classes,
    resolve_model_type,
)
from speculators.models.utils import (
    conditional_torch_compile,
    flatten_rope_parameters,
    get_verifier_config,
)

logger = logging.getLogger(__name__)

__all__ = [
    "DMTPDraftModel",
]

_MIN_BLOCK_SIZE = 2  # anchor slot + T1 slot


@SpeculatorModel.register("dmtp")
class DMTPDraftModel(DFlashDraftModel):
    """DFlash backbone whose block starts from a frozen-MTP-proposed token.

    Implements D-MTP (https://sylvan820.github.io/D-MTP/) with the two-anchor
    block layout ``[A, T1, MASK x (B-2)]``: slot 0 is the committed anchor
    (standard DFlash), slot 1 holds the ground-truth T1 token whose embedding
    is replaced by ``fusion(e_T1, h_MTP)`` -- h_MTP being the frozen MTP
    block's hidden state that *predicted* T1 -- and slots 2..B-1 are mask
    placeholders trained with the usual position-decayed objective. Slots 0
    and 1 receive no draft loss. The verifier, its MTP block, and the LM head
    stay frozen; only the DFlash layers and the fusion module train.
    """

    config_class: ClassVar[type[DMTPSpeculatorConfig]] = DMTPSpeculatorConfig  # type: ignore[misc,assignment]

    def __init__(self, config: DMTPSpeculatorConfig) -> None:
        if config.block_size < _MIN_BLOCK_SIZE:
            raise ValueError(
                f"DMTP requires block_size >= {_MIN_BLOCK_SIZE} (anchor + T1), "
                f"got {config.block_size}."
            )
        if config.sample_from_anchor:
            raise ValueError(
                "DMTP uses the [anchor, T1, masks] block layout and does not "
                "support sample_from_anchor=True."
            )
        mtp_tc = config.mtp_transformer_config
        draft_hidden = config.transformer_layer_config.hidden_size
        if mtp_tc.hidden_size != draft_hidden:
            raise ValueError(
                "mtp_transformer_config.hidden_size "
                f"({mtp_tc.hidden_size}) must match the draft hidden size "
                f"({draft_hidden})."
            )
        super().__init__(config=config)

        # eager OOMs on long sequences for the full-attention MTP block (#886).
        if mtp_tc._attn_implementation is None:  # noqa: SLF001
            mtp_tc._attn_implementation = "sdpa"  # noqa: SLF001
        model_definitions = mtp_model_classes[resolve_model_type(mtp_tc.model_type)]
        # Named mtp_layers so MTPConverter's remapped keys (mtp_layers.0.*)
        # load directly. Unlike DFlash's verifier-owned weights these are
        # saved with the checkpoint, mirroring MTPDraftModel.
        self.mtp_layers = nn.ModuleList(
            [model_definitions.first_layer_class(mtp_tc, layer_idx=0)]
        )
        self.mtp_rotary_emb = model_definitions.rotary_emb_class(
            flatten_rope_parameters(mtp_tc)
        )
        self.mtp_layers.requires_grad_(False)
        self.fusion = MTPTokenFusion(
            hidden_size=draft_hidden,
            rms_norm_eps=config.transformer_layer_config.rms_norm_eps,
        )

    @torch.no_grad()
    def _compute_mtp_hidden(
        self,
        input_ids: torch.Tensor,  # [1, total_seq_len]
        verifier_last_hidden_states: torch.Tensor,  # [1, total_seq_len, hidden]
        position_ids: torch.Tensor,  # [1, total_seq_len]
        anchor_positions: torch.Tensor,  # [num_anchors]
    ) -> torch.Tensor:
        """The frozen MTP hidden that predicted each anchor's T1.

        One causal MTP pass over the packed sequence: output index ``t``
        (from verifier hidden ``h_t`` and the embedding of ``x_{t+1}``)
        predicts ``x_{t+2}``. T1 of anchor ``p`` is ``x_{p+1}``, so its
        predicting hidden sits at index ``p - 1``.
        """
        total_seq_len = input_ids.shape[1]
        valid_len = total_seq_len - 1
        mtp_config = self.config.mtp_transformer_config
        hidden = verifier_last_hidden_states[:, :valid_len]
        embeds = self.embed_tokens(input_ids[:, 1 : 1 + valid_len])
        step_pos_ids, rotary_pos_ids = _prepare_mtp_position_ids(
            position_ids, valid_len, mtp_config.model_type
        )
        causal_mask = create_causal_mask(
            config=mtp_config,
            inputs_embeds=hidden,
            attention_mask=None,
            past_key_values=None,
            position_ids=step_pos_ids,
        )
        position_embeddings = self.mtp_rotary_emb(hidden, rotary_pos_ids)
        mtp_output = self.mtp_layers[0](
            hidden_states=hidden,
            token_embeddings=embeds,
            attention_mask=causal_mask,
            position_ids=step_pos_ids,
            position_embeddings=position_embeddings,
        )  # shape: [1, valid_len, hidden]
        predictor_indices = (anchor_positions - 1).clamp(min=0, max=valid_len - 1)
        return mtp_output[:, predictor_indices]  # shape: [1, num_anchors, hidden]

    def _build_noise_embedding(
        self,
        mask_token_ids: torch.Tensor,
        input_ids: torch.Tensor,
        anchor_positions: torch.Tensor,
        verifier_last_hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Place T1 in slot 1 of each block and fuse it with its MTP hidden."""
        total_seq_len = input_ids.shape[1]
        # Ground-truth T1 (teacher forcing), the token right after the anchor.
        t1_positions = (anchor_positions + 1).clamp(max=total_seq_len - 1)
        mask_token_ids = mask_token_ids.clone()
        mask_token_ids[:, 1 :: self.block_size] = input_ids[:, t1_positions]
        noise_embedding = self.embed_tokens(mask_token_ids)
        mtp_hidden = self._compute_mtp_hidden(
            input_ids, verifier_last_hidden_states, position_ids, anchor_positions
        )
        t1_embeddings = noise_embedding[:, 1 :: self.block_size]
        fused = self.fusion(t1_embeddings, mtp_hidden.to(t1_embeddings.dtype))
        noise_embedding[:, 1 :: self.block_size] = fused
        return noise_embedding

    @classmethod
    def from_training_args(
        cls,
        verifier_config: PretrainedConfig,
        t2d: torch.Tensor | None = None,
        d2t: torch.Tensor | None = None,
        **kwargs,
    ) -> "DMTPDraftModel":
        """Create a D-MTP model; the verifier must carry native mtp.* weights."""
        mtp_transformer_config = get_verifier_config(
            kwargs["verifier_name_or_path"],
            trust_remote_code=kwargs.get("trust_remote_code", False),
        )
        config = DMTPSpeculatorConfig(
            **cls._build_base_config_kwargs("dmtp", verifier_config, **kwargs),
            mtp_transformer_config=mtp_transformer_config,
        )
        model = cls(config=config)

        from speculators.convert.mtp.converter import MTPConverter  # noqa: PLC0415

        mtp_state_dict = MTPConverter().convert_to_state_dict(
            kwargs["verifier_name_or_path"]
        )
        missing, unexpected = model.load_state_dict(mtp_state_dict, strict=False)
        if unexpected:
            raise ValueError(
                f"Verifier MTP weights contain unexpected keys: {unexpected}"
            )
        loaded = set(mtp_state_dict)
        not_loaded = [
            name
            for name, _ in model.mtp_layers.named_parameters(prefix="mtp_layers")
            if name not in loaded
        ]
        if not_loaded:
            raise ValueError(f"Verifier MTP weights are missing keys for: {not_loaded}")
        model.mtp_layers.requires_grad_(False)

        model.load_vocab_mappings(t2d, d2t)
        model.load_verifier_weights()
        return model

    @conditional_torch_compile
    def forward(
        self,
        hidden_states: torch.Tensor,  # shape: [1,total_seq_len,num_hidden*hidden_size]
        input_ids: torch.Tensor,  # shape: [1, total_seq_len]
        loss_mask: torch.Tensor,  # shape: [1, total_seq_len]
        verifier_last_hidden_states: torch.Tensor,  # shape: [1, total_seq_len, hidden_size] # noqa: E501
        document_ids: torch.Tensor,  # shape: [1, total_seq_len]
        position_ids: torch.Tensor | None = None,  # shape: [1, total_seq_len]
        loss_config: LossConfig | None = None,
        gamma: float = 4.0,
        max_anchors: int = 512,
        per_position_loss_weight: str = "fixed-exp-decay",
        dpace_alpha: float = 0.5,
        **kwargs,
    ):
        _, logits, targets, aligned_loss_mask, _ = self._backbone_forward(
            hidden_states,
            input_ids,
            loss_mask,
            verifier_last_hidden_states,
            document_ids,
            position_ids,
            max_anchors=max_anchors,
            **kwargs,
        )
        # Slot 1 holds T1, a known input: no draft loss (slot 0, the anchor,
        # is already zeroed by the base sample_from_anchor=False path).
        aligned_loss_mask = aligned_loss_mask.clone()
        aligned_loss_mask[:, 1 :: self.block_size] = 0
        loss, metrics = compute_metrics(
            logits,
            targets,
            aligned_loss_mask,
            self.block_size,
            gamma=gamma,
            loss_config=loss_config,
            per_position_loss_weight=per_position_loss_weight,
            dpace_alpha=dpace_alpha,
        )
        return None, loss, metrics
