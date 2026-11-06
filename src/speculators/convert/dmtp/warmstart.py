"""Warm-start D-MTP / D-MTP2 drafts from trained DFlash / DFlash2 checkpoints.

The D-MTP draft trunk is the DFlash trunk (and D-MTP2's is the DFlash2 trunk,
including the dynamic convolutions and the candidate selector), so a trained
DFlash-family checkpoint initializes everything except the fusion module and
the frozen MTP block. A plain ``--from-pretrained`` cannot do this because the
source config's ``speculators_model_type`` fails the target config's Literal
validation; this tool rebuilds the config and transfers the weights instead:

    python -m speculators.convert.dmtp.warmstart <dflash_or_dflash2_ckpt> \\
        --verifier <mtp-capable verifier> --output <out_dir> \\
        [--target-type dmtp|dmtp2] [--identity-fusion-init]

The verifier must be the one the source draft was trained against and must
carry native ``mtp.*`` weights. ``--identity-fusion-init`` initializes the
fusion projection to ``[I; 0]`` so slot 1 starts as (the RMS-normed) T1 token
embedding and training gradually mixes in the MTP hidden -- gentler on a
pretrained trunk than a random fusion.
"""

import argparse
import logging
from pathlib import Path

import torch
from torch import nn

from speculators.config import SpeculatorsConfig
from speculators.convert.utils import (
    ensure_checkpoint_is_local,
    load_checkpoint_config,
    load_checkpoint_weights,
)
from speculators.models.dmtp import DMTPDraftModel, DMTPSpeculatorConfig
from speculators.models.dmtp2 import DMTP2DraftModel, DMTP2SpeculatorConfig
from speculators.models.utils import get_verifier_config
from speculators.proposals.greedy import GreedyTokenProposalConfig

logger = logging.getLogger(__name__)

__all__ = [
    "build_target_config",
    "init_identity_fusion",
    "transfer_backbone_weights",
    "warmstart",
]

_SOURCE_TO_TARGET = {"dflash": "dmtp", "dflash2": "dmtp2"}
_TARGET_CLASSES = {
    "dmtp": (DMTPDraftModel, DMTPSpeculatorConfig),
    "dmtp2": (DMTP2DraftModel, DMTP2SpeculatorConfig),
}
# Modules that exist only in the D-MTP side (fresh-initialized, never in the
# source) or are rebuilt from the verifier instead of the source checkpoint.
_NEW_OR_VERIFIER_OWNED_PREFIXES = (
    "fusion.",
    "mtp_layers.",
    "embed_tokens.",
    "lm_head.",
    "verifier_lm_head.",
    "verifier_norm.",
)
# DFlash2-only modules, dropped when warm-starting plain dmtp from a dflash2
# source (the dmtp trunk has no convolutions or selector).
_DFLASH2_ONLY_MARKERS = ("attention_conv.", "mlp_conv.", "candidate_selector.")

_SHARED_CONFIG_FIELDS = (
    "transformer_layer_config",
    "draft_vocab_size",
    "block_size",
    "target_hidden_size",
    "aux_hidden_state_layer_ids",
    "mask_token_id",
    "sliding_window_non_causal",
)
_DFLASH2_CONFIG_FIELDS = (
    "conv_kernel_size",
    "conv_group_size",
    "selector_rank",
    "selector_top_k",
)


def build_target_config(
    source_config: dict,
    target_type: str,
    mtp_transformer_config,
):
    """Build a D-MTP(2) config from a DFlash(2) checkpoint's config dict."""
    config_class = _TARGET_CLASSES[target_type][1]
    fields = _SHARED_CONFIG_FIELDS + (
        _DFLASH2_CONFIG_FIELDS if target_type == "dmtp2" else ()
    )
    kwargs = {
        name: source_config[name]
        for name in fields
        if source_config.get(name) is not None
    }

    source_speculators_config = source_config.get("speculators_config") or {}
    verifier = source_speculators_config.get("verifier")
    if verifier is None:
        raise ValueError(
            "Source checkpoint config has no speculators_config.verifier; "
            "is this a speculators-format DFlash checkpoint?"
        )
    block_size = kwargs.get("block_size", 8)
    kwargs["speculators_config"] = SpeculatorsConfig(
        algorithm=target_type,
        proposal_methods=[GreedyTokenProposalConfig(speculative_tokens=block_size - 1)],
        default_proposal_method="greedy",
        verifier=verifier,
    )
    return config_class(
        **kwargs,
        sample_from_anchor=False,
        mtp_transformer_config=mtp_transformer_config,
    )


def transfer_backbone_weights(
    target_model: nn.Module,
    source_state_dict: dict[str, torch.Tensor],
) -> tuple[list[str], list[str]]:
    """Load a DFlash(2) state dict into a D-MTP(2) model, strictly accounted.

    Returns ``(transferred, dropped)`` key lists. Raises if the source carries
    keys that are neither loadable nor known DFlash2-only modules, or if any
    target backbone weight (outside the fresh fusion / verifier-rebuilt
    modules) is left uninitialized.
    """
    loadable = {
        key: value
        for key, value in source_state_dict.items()
        if key in dict(target_model.state_dict())
    }
    dropped = [key for key in source_state_dict if key not in loadable]
    unexplained = [
        key
        for key in dropped
        if not any(marker in key for marker in _DFLASH2_ONLY_MARKERS)
        and key not in {"t2d", "d2t"}
        and not any(key.startswith(p) for p in _NEW_OR_VERIFIER_OWNED_PREFIXES)
    ]
    if unexplained:
        raise ValueError(
            f"Source checkpoint has keys the target cannot absorb: {unexplained}"
        )

    missing, unexpected = target_model.load_state_dict(loadable, strict=False)
    if unexpected:  # loadable was filtered against the target; this is a bug
        raise ValueError(f"Unexpectedly rejected keys: {unexpected}")
    uninitialized = [
        key
        for key in missing
        if not any(key.startswith(p) for p in _NEW_OR_VERIFIER_OWNED_PREFIXES)
        and key not in {"t2d", "d2t"}
    ]
    if uninitialized:
        raise ValueError(
            "Target backbone weights not covered by the source checkpoint: "
            f"{uninitialized}"
        )
    if dropped:
        logger.info("Dropped %d source-only keys: %s", len(dropped), dropped)
    return sorted(loadable), sorted(dropped)


@torch.no_grad()
def init_identity_fusion(model) -> None:
    """Initialize ``fusion_proj`` to ``[I; 0]``.

    The fused T1 slot then starts as the RMS-normed T1 token embedding with a
    zero MTP branch, keeping the pretrained trunk close to its training
    distribution while the fusion learns to mix in ``h_MTP``.
    """
    weight = model.fusion.fusion_proj.weight  # [d, 2d]; input is [emb; mtp]
    hidden = weight.shape[0]
    weight.zero_()
    weight[:, :hidden] = torch.eye(hidden, dtype=weight.dtype, device=weight.device)


def warmstart(
    source_path: str | Path,
    verifier: str,
    output_path: str | Path,
    target_type: str | None = None,
    identity_fusion_init: bool = False,
    trust_remote_code: bool = False,
    cache_dir: str | Path | None = None,
) -> Path:
    """Convert a DFlash(2) checkpoint into a warm-started D-MTP(2) checkpoint."""
    local_path = ensure_checkpoint_is_local(source_path, cache_dir)
    source_config = load_checkpoint_config(local_path)
    source_type = source_config.get("speculators_model_type")
    if target_type is None:
        if source_type not in _SOURCE_TO_TARGET:
            raise ValueError(
                f"Cannot infer a warm-start target from source type "
                f"{source_type!r}; pass --target-type explicitly."
            )
        target_type = _SOURCE_TO_TARGET[source_type]
    if target_type not in _TARGET_CLASSES:
        raise ValueError(f"Unknown target type: {target_type!r}")
    if target_type == "dmtp2" and source_type != "dflash2":
        raise ValueError(
            "dmtp2 warm-starts require a dflash2 source (its convolutions and "
            f"selector have no counterpart in a {source_type!r} checkpoint)."
        )
    logger.info(
        "Warm-starting %s from %s checkpoint %s", target_type, source_type, source_path
    )

    model_class, _ = _TARGET_CLASSES[target_type]
    mtp_transformer_config = get_verifier_config(
        verifier, trust_remote_code=trust_remote_code
    )
    config = build_target_config(source_config, target_type, mtp_transformer_config)
    model = model_class(config)
    model_class._load_frozen_mtp_weights(model, verifier)  # noqa: SLF001

    weights = load_checkpoint_weights(local_path)
    transferred, dropped = transfer_backbone_weights(model, weights)
    logger.info("Transferred %d backbone tensors", len(transferred))

    t2d, d2t = weights.get("t2d"), weights.get("d2t")
    model.load_vocab_mappings(t2d, d2t)
    model.load_verifier_weights()
    if identity_fusion_init:
        init_identity_fusion(model)
        logger.info("Initialized fusion_proj to [I; 0]")

    output_path = Path(output_path)
    model.save_pretrained(output_path)
    logger.info("Saved warm-started %s checkpoint to %s", target_type, output_path)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("source", help="DFlash or DFlash2 checkpoint path/repo id")
    parser.add_argument(
        "--verifier",
        required=True,
        help="Verifier the source was trained against (must carry mtp.* weights)",
    )
    parser.add_argument("--output", required=True, help="Output checkpoint directory")
    parser.add_argument(
        "--target-type",
        choices=sorted(_TARGET_CLASSES),
        default=None,
        help="Defaults by source type: dflash->dmtp, dflash2->dmtp2. "
        "dflash2->dmtp drops the convolutions and selector.",
    )
    parser.add_argument(
        "--identity-fusion-init",
        action="store_true",
        help="Initialize the fusion projection to [I; 0] (T1-embedding start)",
    )
    parser.add_argument("--trust-remote-code", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    warmstart(
        args.source,
        verifier=args.verifier,
        output_path=args.output,
        target_type=args.target_type,
        identity_fusion_init=args.identity_fusion_init,
        trust_remote_code=args.trust_remote_code,
    )


if __name__ == "__main__":
    main()
