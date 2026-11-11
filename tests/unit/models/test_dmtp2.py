"""D-MTP2 (frozen-MTP first token + DFlash2 backbone) unit tests."""

from __future__ import annotations

import pytest
import torch
from transformers.models.qwen3.modeling_qwen3 import Qwen3Config

from speculators.config import SpeculatorsConfig, VerifierConfig
from speculators.losses import eager as eager_losses
from speculators.model import SpeculatorModel
from speculators.models.dmtp2 import DMTP2DraftModel, DMTP2SpeculatorConfig
from speculators.proposals.greedy import GreedyTokenProposalConfig

VERIFIER_VOCAB = 64
HIDDEN_SIZE = 16
NUM_TARGET_LAYERS = 2
NUM_DRAFT_LAYERS = 2
BLOCK_SIZE = 4


def _tl_config(num_hidden_layers: int = NUM_DRAFT_LAYERS) -> Qwen3Config:
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


def _make_model() -> DMTP2DraftModel:
    config = DMTP2SpeculatorConfig(
        transformer_layer_config=_tl_config(),
        mtp_transformer_config=_tl_config(num_hidden_layers=1),
        draft_vocab_size=VERIFIER_VOCAB,  # DFlash2 requires the full vocab
        block_size=BLOCK_SIZE,
        aux_hidden_state_layer_ids=list(range(NUM_TARGET_LAYERS)),
        mask_token_id=0,
        selector_rank=8,
        selector_top_k=4,
        speculators_config=SpeculatorsConfig(
            algorithm="dmtp2",
            proposal_methods=[
                GreedyTokenProposalConfig(speculative_tokens=BLOCK_SIZE - 1)
            ],
            default_proposal_method="greedy",
            verifier=VerifierConfig(name_or_path="dummy", architectures=[]),
        ),
    )
    model = DMTP2DraftModel(config)
    with torch.no_grad():
        for param in model.parameters():
            if param.isnan().any():
                torch.nn.init.normal_(param, mean=0.0, std=0.02)
    torch.nn.init.normal_(model.verifier_lm_head.weight)
    torch.nn.init.ones_(model.verifier_norm.weight)
    return model


def test_registry_contains_dmtp2():
    assert SpeculatorModel.registry is not None
    assert SpeculatorModel.registry["dmtp2"] is DMTP2DraftModel
    assert DMTP2DraftModel.config_class is DMTP2SpeculatorConfig


def test_config_merges_dflash2_and_dmtp_fields():
    config = DMTP2SpeculatorConfig(
        transformer_layer_config=_tl_config(),
        mtp_transformer_config=_tl_config(num_hidden_layers=1),
        draft_vocab_size=VERIFIER_VOCAB,
        aux_hidden_state_layer_ids=[0, 1],
        mask_token_id=0,
    )
    assert config.speculators_model_type == "dmtp2"
    # DFlash2 side
    assert config.conv_kernel_size == 2
    assert config.selector_top_k == 16
    assert config.sliding_window_non_causal is True  # DFlash2 default wins
    # D-MTP side
    assert config.mtp_transformer_config.num_hidden_layers == 1


def test_model_combines_both_mechanisms():
    model = _make_model()
    # D-MTP side: frozen MTP + trainable fusion
    assert all(not p.requires_grad for p in model.mtp_layers.parameters())
    assert all(p.requires_grad for p in model.fusion.parameters())
    # DFlash2 side: convolutions and selector present and trainable
    assert hasattr(model.layers[0], "attention_conv")
    assert all(p.requires_grad for p in model.candidate_selector.parameters())


def test_full_forward_backward_cpu_smoke():
    torch.manual_seed(0)
    model = _make_model().train()
    seq_len = 32
    hidden = torch.randn(1, seq_len, NUM_TARGET_LAYERS * HIDDEN_SIZE)
    verifier_last = torch.randn(1, seq_len, HIDDEN_SIZE)
    input_ids = torch.randint(1, VERIFIER_VOCAB, (1, seq_len))
    loss_mask = torch.ones(1, seq_len)
    document_ids = torch.zeros(1, seq_len, dtype=torch.long)

    _, loss, metrics = model(
        hidden_states=hidden,
        input_ids=input_ids,
        loss_mask=loss_mask,
        verifier_last_hidden_states=verifier_last,
        document_ids=document_ids,
        loss_config={"ce": (eager_losses.ce_loss, 1.0)},
        max_anchors=5,
    )
    assert torch.isfinite(loss)
    # Slots 0 (anchor) and 1 (T1) are inputs: metrics start at position 2.
    assert "position_1_acc_sum" not in metrics
    assert "position_2_acc_sum" in metrics
    assert "selector_loss_sum" in metrics
    assert "unary_loss_sum" in metrics
    assert "eal_sum" in metrics
    # Realized-path EAL is not zeroed by the masked T1 slot: with any correct
    # continuation slot it can exceed the floor of 1.0 per block.
    assert metrics["eal_total"] > 0

    loss.backward()
    # With the zero-initialized residual, gradients reach the projection
    # directly; the norm weights' grads are gated by W (zero at step 0).
    proj_grad = model.fusion.fusion_proj.weight.grad
    assert proj_grad is not None
    assert torch.any(proj_grad != 0)
    assert all(p.grad is not None for p in model.fusion.parameters())
    assert any(
        p.grad is not None and torch.any(p.grad != 0)
        for p in model.candidate_selector.parameters()
    )
    assert all(p.grad is None for p in model.mtp_layers.parameters())


def test_rejects_sample_from_anchor():
    with pytest.raises(ValueError, match="sample_from_anchor"):
        DMTP2DraftModel(
            DMTP2SpeculatorConfig(
                transformer_layer_config=_tl_config(),
                mtp_transformer_config=_tl_config(num_hidden_layers=1),
                draft_vocab_size=VERIFIER_VOCAB,
                aux_hidden_state_layer_ids=[0, 1],
                mask_token_id=0,
                sample_from_anchor=True,
            )
        )
