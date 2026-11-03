import logging
from typing import ClassVar

import torch
from torch import nn
from transformers import PretrainedConfig

from speculators.model import SpeculatorModel
from speculators.models.dcore.config import DCoreSpeculatorConfig
from speculators.models.dcore.model_definitions import Qwen3DCoreDecoderLayer
from speculators.models.dflash.config import DFlashSpeculatorConfig
from speculators.models.dflash.core import DFlashDraftModel
from speculators.models.utils import get_verifier_config

logger = logging.getLogger(__name__)

__all__ = [
    "DCoreDraftModel",
    "default_dcore_target_layer_ids",
]

# The released DCore checkpoints fuse 9 target layers (DFlare's default T).
DCORE_NUM_TARGET_LAYERS = 9
_MIN_TARGET_LAYERS = 2


def default_dcore_target_layer_ids(
    num_verifier_layers: int,
    num_target_layers: int = DCORE_NUM_TARGET_LAYERS,
) -> list[int]:
    """Uniform target layer ids in ``[1, n-3]``, matching released DCore ckpts.

    For a 36-layer verifier this yields ``[1, 5, 9, 13, 17, 21, 25, 29, 33]``,
    the ids recorded in ``ernie-research/DCore-Qwen3-8B-*``.
    """
    first, last = 1, num_verifier_layers - 3
    if last <= first or num_target_layers < _MIN_TARGET_LAYERS:
        raise ValueError(
            f"Cannot place {num_target_layers} distinct target layers in "
            f"[{first}, {last}] for a verifier with {num_verifier_layers} layers; "
            "pass --target-layer-ids explicitly."
        )
    step = (last - first) / (num_target_layers - 1)
    ids = [round(first + i * step) for i in range(num_target_layers)]
    if len(set(ids)) != len(ids):
        raise ValueError(
            f"Default DCore target layer ids collide ({ids}) for a verifier with "
            f"{num_verifier_layers} layers; pass --target-layer-ids explicitly."
        )
    return ids


@SpeculatorModel.register("dcore")
class DCoreDraftModel(DFlashDraftModel):
    """DFlash backbone + DFlare per-layer target fusion and heterogeneous KV.

    Weight keys are aligned with the released ``ernie-research/DCore-*``
    checkpoints: the shared ``fc`` is removed, ``hidden_norm`` is reused as the
    single post-fusion RMSNorm, the fusion matrix lives at
    ``layer_fusion_weights`` and the context projections at
    ``layers.{i}.self_attn.{k,v}_proj_target``. The D-PACE training objective
    is inherited from the DFlash loss path (``per_position_loss_weight="dpace"``).
    """

    config_class: ClassVar[type[DCoreSpeculatorConfig]] = DCoreSpeculatorConfig  # type: ignore[misc,assignment]
    _no_split_modules = ["Qwen3DCoreDecoderLayer"]

    def __init__(self, config: DCoreSpeculatorConfig) -> None:
        super().__init__(config=config)
        # DCore has no shared FC fusion; hidden_norm is kept as the shared
        # post-fusion RMSNorm (checkpoint key `hidden_norm.weight`).
        del self.fc
        self.layer_fusion_weights = nn.Parameter(
            torch.empty(len(self.layers), len(self.target_layer_ids))
        )
        self._init_fusion_weights()

    def _init_fusion_weights(self) -> None:
        """Bias each draft layer toward one progressively deeper target layer.

        Matches ``_init_fusion_weights`` in the released ``dflare.py``: zeros
        everywhere, with a single 2.0 at target index ``int(d/D*T)`` per row.
        """
        nn.init.constant_(self.layer_fusion_weights, 0.0)
        num_draft, num_target = self.layer_fusion_weights.shape
        for d_idx in range(num_draft):
            t_idx = min(num_target - 1, int(d_idx / num_draft * num_target))
            self.layer_fusion_weights.data[d_idx, t_idx] = 2.0

    def _make_decoder_layer(
        self, config: DFlashSpeculatorConfig, layer_idx: int
    ) -> Qwen3DCoreDecoderLayer:
        return Qwen3DCoreDecoderLayer(
            config.transformer_layer_config,  # type: ignore[arg-type]
            layer_idx,
        )

    def _fuse_target_hidden(self, hidden_states: torch.Tensor) -> list[torch.Tensor]:
        """Per-draft-layer softmax-weighted fusion of the target hidden states.

        ``hidden_states`` is the ``[1, L, T*d]`` concatenation produced by the
        data pipeline; the einsum and shared ``hidden_norm`` mirror the released
        ``DFlareDraftModel.forward``.
        """
        bsz, seq_len, _ = hidden_states.shape
        num_draft, num_target = self.layer_fusion_weights.shape
        reshaped = hidden_states.view(bsz, seq_len, num_target, -1)
        fusion_probs = torch.softmax(self.layer_fusion_weights, dim=1)
        fused = torch.einsum("bsth,dt->bsdh", reshaped, fusion_probs)
        fused = self.hidden_norm(fused)  # shape: [1, L, num_draft, hidden_size]
        return [fused[:, :, i, :] for i in range(num_draft)]

    @classmethod
    def from_training_args(
        cls,
        verifier_config: PretrainedConfig,
        t2d: torch.Tensor | None = None,
        d2t: torch.Tensor | None = None,
        **kwargs,
    ) -> "DCoreDraftModel":
        """Create a DCore model from the shared training arguments.

        Unlike DFlash's 3-layer EAGLE3-compatible default, an unset
        ``--target-layer-ids`` resolves to 9 uniform layers in ``[1, n-3]``
        (the DFlare/DCore recipe). The vLLM server collecting hidden states
        must be launched with the same ids.
        """
        if kwargs.get("target_layer_ids") is None:
            num_verifier_layers = get_verifier_config(
                kwargs["verifier_name_or_path"],
                trust_remote_code=kwargs.get("trust_remote_code", False),
            ).num_hidden_layers
            kwargs["target_layer_ids"] = default_dcore_target_layer_ids(
                num_verifier_layers
            )
            logger.info(
                "DCore: --target-layer-ids not set; using %s (9 uniform layers). "
                "Launch vLLM datagen with the same ids.",
                kwargs["target_layer_ids"],
            )
        config = DCoreSpeculatorConfig(
            **cls._build_base_config_kwargs("dcore", verifier_config, **kwargs)
        )
        model = cls(config=config)
        model.load_vocab_mappings(t2d, d2t)
        model.load_verifier_weights()
        return model
