"""D-MTP draft model modules.

D-MTP (https://sylvan820.github.io/D-MTP/) lets a frozen pretrained MTP module
propose the first draft token (T1) of each block while the DFlash backbone
continues the block in one parallel pass. The only new trainable component
besides the DFlash layers is a small zero-initialized residual fusion that
augments the T1 slot's input embedding:

    delta = W_f [ RMS_e(e_T1) ; RMS_m(stopgrad(h_MTP)) ],  W_f init = 0
    e~_T1 = e_T1 + delta

where ``h_MTP`` is the MTP hidden state that *predicted* T1 (not one computed
after consuming it). Because ``W_f`` starts at zero, the T1 slot initially
equals the raw token embedding and training gradually folds in the MTP
features -- preserving the pretrained embedding pathway, which suits continued
fine-tuning from DFlash-family weights.
"""

import torch
from torch import nn
from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm

__all__ = [
    "MTPTokenFusion",
]


class MTPTokenFusion(nn.Module):
    """Zero-initialized residual fusion of a token embedding with its MTP hidden.

    Each input gets its own learned RMSNorm before a ``2d -> d`` projection;
    the projection output is added to the raw token embedding as a residual.
    The projection weight starts at zero, so the module is initially an exact
    pass-through of the token embedding.
    """

    def __init__(self, hidden_size: int, rms_norm_eps: float) -> None:
        super().__init__()
        self.emb_norm = Qwen3RMSNorm(hidden_size, eps=rms_norm_eps)
        self.mtp_norm = Qwen3RMSNorm(hidden_size, eps=rms_norm_eps)
        self.fusion_proj = nn.Linear(2 * hidden_size, hidden_size, bias=False)
        nn.init.zeros_(self.fusion_proj.weight)

    def forward(
        self,
        token_embedding: torch.Tensor,  # [..., hidden_size]
        mtp_hidden: torch.Tensor,  # [..., hidden_size]
    ) -> torch.Tensor:
        # stopgrad on the MTP branch per the D-MTP recipe (the frozen MTP pass
        # already runs under no_grad; detach keeps the contract explicit).
        fused = torch.cat(
            [self.emb_norm(token_embedding), self.mtp_norm(mtp_hidden.detach())],
            dim=-1,
        )
        return token_embedding + self.fusion_proj(fused)
