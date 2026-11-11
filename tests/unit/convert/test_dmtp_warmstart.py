"""Warm-start weight transfer tests: DFlash->D-MTP and DFlash2->D-MTP2."""

from __future__ import annotations

import importlib

import pytest
import torch
from transformers.models.qwen3.modeling_qwen3 import Qwen3Config

from speculators.config import SpeculatorsConfig, VerifierConfig
from speculators.convert import entrypoints
from speculators.convert.dmtp.warmstart import (
    build_target_config,
    transfer_backbone_weights,
)
from speculators.models.dflash import DFlashDraftModel, DFlashSpeculatorConfig
from speculators.models.dflash2 import DFlash2DraftModel, DFlash2SpeculatorConfig
from speculators.models.dmtp import DMTPDraftModel, DMTPSpeculatorConfig
from speculators.models.dmtp2 import DMTP2DraftModel, DMTP2SpeculatorConfig
from speculators.proposals.greedy import GreedyTokenProposalConfig

VERIFIER_VOCAB = 64
HIDDEN_SIZE = 16
BLOCK_SIZE = 4


def _tl_config(num_hidden_layers: int = 2) -> Qwen3Config:
    return Qwen3Config(
        hidden_size=HIDDEN_SIZE,
        intermediate_size=32,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        vocab_size=VERIFIER_VOCAB,
        _attn_implementation="eager",  # type: ignore[call-arg]
    )


def _speculators_config(algorithm: str) -> SpeculatorsConfig:
    return SpeculatorsConfig(
        algorithm=algorithm,
        proposal_methods=[GreedyTokenProposalConfig(speculative_tokens=BLOCK_SIZE - 1)],
        default_proposal_method="greedy",
        verifier=VerifierConfig(name_or_path="dummy", architectures=[]),
    )


def _common_kwargs() -> dict:
    return {
        "transformer_layer_config": _tl_config(),
        "draft_vocab_size": VERIFIER_VOCAB,
        "block_size": BLOCK_SIZE,
        "aux_hidden_state_layer_ids": [0, 1],
        "mask_token_id": 0,
    }


def _randomize(model: torch.nn.Module) -> None:
    with torch.no_grad():
        for param in model.parameters():
            torch.nn.init.normal_(param, mean=0.0, std=0.02)


def _make_dflash() -> DFlashDraftModel:
    model = DFlashDraftModel(
        DFlashSpeculatorConfig(
            **_common_kwargs(), speculators_config=_speculators_config("dflash")
        )
    )
    _randomize(model)
    return model


def _make_dflash2() -> DFlash2DraftModel:
    model = DFlash2DraftModel(
        DFlash2SpeculatorConfig(
            **_common_kwargs(),
            selector_rank=8,
            selector_top_k=4,
            speculators_config=_speculators_config("dflash2"),
        )
    )
    _randomize(model)
    return model


def _make_dmtp() -> DMTPDraftModel:
    return DMTPDraftModel(
        DMTPSpeculatorConfig(
            **_common_kwargs(),
            mtp_transformer_config=_tl_config(num_hidden_layers=1),
            speculators_config=_speculators_config("dmtp"),
        )
    )


def _make_dmtp2() -> DMTP2DraftModel:
    return DMTP2DraftModel(
        DMTP2SpeculatorConfig(
            **_common_kwargs(),
            selector_rank=8,
            selector_top_k=4,
            mtp_transformer_config=_tl_config(num_hidden_layers=1),
            speculators_config=_speculators_config("dmtp2"),
        )
    )


def test_dflash_to_dmtp_transfers_every_trunk_tensor():
    source = _make_dflash()
    target = _make_dmtp()
    transferred, dropped = transfer_backbone_weights(target, source.state_dict())
    assert dropped == []  # dflash -> dmtp: nothing is left behind
    target_sd = target.state_dict()
    for key, value in source.state_dict().items():
        torch.testing.assert_close(target_sd[key], value)
    assert "layers.0.self_attn.q_proj.weight" in transferred
    assert "fc.weight" in transferred


def test_dflash2_to_dmtp2_keeps_convs_and_selector():
    source = _make_dflash2()
    target = _make_dmtp2()
    _, dropped = transfer_backbone_weights(target, source.state_dict())
    assert dropped == []  # convs and selector carry over, nothing dropped
    target_sd = target.state_dict()
    carried_keys = [
        k
        for k in source.state_dict()
        if "attention_conv" in k or k.startswith("candidate_selector.")
    ]
    assert carried_keys
    for key in carried_keys:
        torch.testing.assert_close(target_sd[key], source.state_dict()[key])


def test_dflash2_to_dmtp_drops_only_dflash2_modules():
    source = _make_dflash2()
    target = _make_dmtp()
    _, dropped = transfer_backbone_weights(target, source.state_dict())
    assert dropped  # convs + selector
    assert all(("conv" in key) or ("candidate_selector" in key) for key in dropped)


def test_transfer_rejects_unknown_source_keys():
    source_sd = _make_dflash().state_dict()
    source_sd["layers.0.some_novel_module.weight"] = torch.zeros(2)
    with pytest.raises(ValueError, match="cannot absorb"):
        transfer_backbone_weights(_make_dmtp(), source_sd)


def test_fresh_fusion_passes_through_t1_embedding():
    """The zero-initialized residual fusion starts as an exact pass-through."""
    model = _make_dmtp()
    emb = torch.randn(1, 3, HIDDEN_SIZE)
    mtp_hidden = torch.randn(1, 3, HIDDEN_SIZE)
    out = model.fusion(emb, mtp_hidden)
    torch.testing.assert_close(out, emb)


def test_warmstart_autoconverts_external_source(monkeypatch, tmp_path):
    """External (no speculators_model_type) sources are converted, then typed."""
    # The package re-exports the warmstart *function* under the same name, so
    # import the module explicitly.
    ws = importlib.import_module("speculators.convert.dmtp.warmstart")

    external = {
        "architectures": ["DFlash2DraftModel"],
        "dflash_config": {"selector_rank": 256},
    }
    converted = {"speculators_model_type": "dflash2"}
    configs = iter([external, converted])

    monkeypatch.setattr(ws, "ensure_checkpoint_is_local", lambda path, cache: tmp_path)
    monkeypatch.setattr(ws, "load_checkpoint_config", lambda path: next(configs))

    calls: dict = {}

    def fake_convert(model, verifier=None, cache_dir=None, config_dict=None):
        calls["verifier"] = verifier
        calls["config_dict"] = config_dict
        return str(tmp_path / "converted")

    monkeypatch.setattr(entrypoints, "maybe_convert_external_checkpoint", fake_convert)

    class _StopError(Exception):
        """Raised to end warmstart after type inference succeeded."""

    def fake_verifier_config(*args, **kwargs):
        raise _StopError

    monkeypatch.setattr(ws, "get_verifier_config", fake_verifier_config)

    # Reaching get_verifier_config means the dflash2->dmtp2 inference succeeded
    # on the converted config instead of raising "Cannot infer".
    with pytest.raises(_StopError):
        ws.warmstart(tmp_path, verifier="dummy/verifier", output_path=tmp_path / "out")
    assert calls["verifier"] == "dummy/verifier"
    assert calls["config_dict"] == external


def test_build_target_config_from_source_dict():
    mtp_tc = _tl_config(num_hidden_layers=1)
    source_dict = {
        "speculators_model_type": "dflash2",
        "transformer_layer_config": _tl_config().to_diff_dict(),
        "draft_vocab_size": VERIFIER_VOCAB,
        "block_size": BLOCK_SIZE,
        "aux_hidden_state_layer_ids": [0, 1],
        "mask_token_id": 0,
        "sliding_window_non_causal": True,
        "conv_kernel_size": 3,
        "selector_rank": 8,
        "selector_top_k": 4,
        "speculators_config": {
            "algorithm": "dflash2",
            "proposal_methods": [
                {"proposal_type": "greedy", "speculative_tokens": BLOCK_SIZE - 1}
            ],
            "default_proposal_method": "greedy",
            "verifier": {"name_or_path": "dummy", "architectures": []},
        },
    }
    config = build_target_config(source_dict, "dmtp2", mtp_tc)
    assert config.speculators_model_type == "dmtp2"
    assert config.block_size == BLOCK_SIZE
    assert config.conv_kernel_size == 3
    assert config.selector_rank == 8
    assert config.sample_from_anchor is False
    assert config.speculators_config.algorithm == "dmtp2"
    assert config.mtp_transformer_config.num_hidden_layers == 1
