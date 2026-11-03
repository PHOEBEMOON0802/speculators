"""DCore draft model modules.

DCore (arXiv:2610.24993, App. G) keeps the DFlash block-parallel backbone and
adds DFlare's two structural changes (arXiv:2606.02091, Sec. 4):

1. Heterogeneous KV projections: the injected target context uses dedicated
   ``k_proj_target`` / ``v_proj_target`` projections instead of sharing the
   block-token ``k_proj`` / ``v_proj``.
2. Per-draft-layer fusion of target hidden states via a learnable
   ``layer_fusion_weights [D, T]`` matrix instead of a shared ``Linear(T*H, H)``.

Attribute names follow the released ``ernie-research/DCore-*`` checkpoints
(``dflare.py``) so trained weights are key-compatible with them.
"""

from torch import nn
from transformers.models.qwen3.modeling_qwen3 import Qwen3Config

from speculators.models.dflash.model_definitions import (
    Qwen3DFlashAttention,
    Qwen3DFlashDecoderLayer,
)

__all__ = [
    "Qwen3DCoreAttention",
    "Qwen3DCoreDecoderLayer",
]


class Qwen3DCoreAttention(Qwen3DFlashAttention):
    """DFlash cross-attention with separate context K/V projections.

    The target context and the draft block tokens carry different information
    (semantic knowledge vs. evolving noisy predictions); giving the context its
    own K/V projections frees each source to use its own representational
    subspace (DFlare Eq. 4).
    """

    def __init__(self, config: Qwen3Config, layer_idx: int):
        super().__init__(config, layer_idx)
        kv_dim = config.num_key_value_heads * self.head_dim  # type: ignore[operator]
        # Names match the released DCore/DFlare checkpoints.
        self.k_proj_target = nn.Linear(
            config.hidden_size,  # type: ignore[arg-type]
            kv_dim,
            bias=config.attention_bias,  # type: ignore[arg-type]
        )
        self.v_proj_target = nn.Linear(
            config.hidden_size,  # type: ignore[arg-type]
            kv_dim,
            bias=config.attention_bias,  # type: ignore[arg-type]
        )

    def _project_context(self, target_hidden):
        return self.k_proj_target(target_hidden), self.v_proj_target(target_hidden)


class Qwen3DCoreDecoderLayer(Qwen3DFlashDecoderLayer):
    attention_class = Qwen3DCoreAttention
