"""D-MTP draft model modules.

D-MTP (https://sylvan820.github.io/D-MTP/) lets a frozen pretrained MTP module
propose the first draft token (T1) of each block while the DFlash backbone
continues the block in one parallel pass. The only new trainable component
besides the DFlash layers is a small fusion module that replaces the T1 slot's
input embedding:

    e~_T1 = W_f [ RMS_e(e_T1) ; RMS_m(h_MTP) ]

where ``h_MTP`` is the MTP hidden state that *predicted* T1 (not one computed
after consuming it).
"""

import torch
from torch import nn
from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm

__all__ = [
    "MTPTokenFusion",
]


class MTPTokenFusion(nn.Module):
    """Fuse a token embedding with the MTP hidden state that predicted it.

    Each input gets its own learned RMSNorm before a ``2d -> d`` projection,
    following the D-MTP fusion equation.
    """

    def __init__(self, hidden_size: int, rms_norm_eps: float) -> None:
        super().__init__()
        self.emb_norm = Qwen3RMSNorm(hidden_size, eps=rms_norm_eps)
        self.mtp_norm = Qwen3RMSNorm(hidden_size, eps=rms_norm_eps)
        self.fusion_proj = nn.Linear(2 * hidden_size, hidden_size, bias=False)

    def forward(
        self,
        token_embedding: torch.Tensor,  # [..., hidden_size]
        mtp_hidden: torch.Tensor,  # [..., hidden_size]
    ) -> torch.Tensor:
        fused = torch.cat(
            [self.emb_norm(token_embedding), self.mtp_norm(mtp_hidden)], dim=-1
        )
        return self.fusion_proj(fused)
