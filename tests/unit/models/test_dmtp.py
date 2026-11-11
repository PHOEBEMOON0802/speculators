"""D-MTP draft model unit tests.

Covers: registry wiring, the MTP-token fusion module, frozen-MTP parameter
handling, T1 slot embedding replacement in the block input, h_MTP predictor
indexing, slot-0/1 loss masking via the dmtp metrics, checkpoint keys, and a
CPU end-to-end forward/backward smoke test.
"""

from __future__ import annotations

import pytest
import torch
from safetensors.torch import load_file
from transformers.models.qwen3.modeling_qwen3 import Qwen3Config

from speculators.config import SpeculatorsConfig, VerifierConfig
from speculators.losses import eager as eager_losses
from speculators.model import SpeculatorModel
from speculators.models.dmtp import DMTPDraftModel, DMTPSpeculatorConfig
from speculators.models.dmtp.model_definitions import MTPTokenFusion
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


def _make_model() -> DMTPDraftModel:
    config = DMTPSpeculatorConfig(
        transformer_layer_config=_tl_config(),
        mtp_transformer_config=_tl_config(num_hidden_layers=1),
        draft_vocab_size=VERIFIER_VOCAB,
        block_size=BLOCK_SIZE,
        aux_hidden_state_layer_ids=list(range(NUM_TARGET_LAYERS)),
        mask_token_id=0,
        speculators_config=SpeculatorsConfig(
            algorithm="dmtp",
            proposal_methods=[
                GreedyTokenProposalConfig(speculative_tokens=BLOCK_SIZE - 1)
            ],
            default_proposal_method="greedy",
            verifier=VerifierConfig(name_or_path="dummy", architectures=[]),
        ),
    )
    model = DMTPDraftModel(config)
    # Verifier-owned weights are normally loaded from the verifier; replace
    # the NaN placeholders with real values (mirrors integration conftest).
    with torch.no_grad():
        for param in model.parameters():
            if param.isnan().any():
                torch.nn.init.normal_(param, mean=0.0, std=0.02)
    torch.nn.init.normal_(model.verifier_lm_head.weight)
    torch.nn.init.ones_(model.verifier_norm.weight)
    return model


def test_registry_contains_dmtp():
    assert SpeculatorModel.registry is not None
    assert SpeculatorModel.registry["dmtp"] is DMTPDraftModel
    assert DMTPDraftModel.config_class is DMTPSpeculatorConfig


def test_fusion_module_matches_manual_computation():
    torch.manual_seed(0)
    fusion = MTPTokenFusion(hidden_size=HIDDEN_SIZE, rms_norm_eps=1e-6)
    torch.nn.init.normal_(fusion.fusion_proj.weight)
    emb = torch.randn(1, 3, HIDDEN_SIZE)
    mtp_hidden = torch.randn(1, 3, HIDDEN_SIZE)
    out = fusion(emb, mtp_hidden)
    assert out.shape == (1, 3, HIDDEN_SIZE)
    expected = torch.cat([fusion.emb_norm(emb), fusion.mtp_norm(mtp_hidden)], dim=-1)
    expected = emb + expected @ fusion.fusion_proj.weight.T
    torch.testing.assert_close(out, expected)


def test_fusion_zero_init_is_exact_passthrough():
    """W_init = 0 => e~_T1 == e_T1: the raw embedding pathway is preserved."""
    fusion = MTPTokenFusion(hidden_size=HIDDEN_SIZE, rms_norm_eps=1e-6)
    emb = torch.randn(1, 3, HIDDEN_SIZE)
    mtp_hidden = torch.randn(1, 3, HIDDEN_SIZE)
    torch.testing.assert_close(fusion(emb, mtp_hidden), emb)


def test_fusion_stops_gradient_into_mtp_branch():
    fusion = MTPTokenFusion(hidden_size=HIDDEN_SIZE, rms_norm_eps=1e-6)
    torch.nn.init.normal_(fusion.fusion_proj.weight)
    emb = torch.randn(1, 3, HIDDEN_SIZE, requires_grad=True)
    mtp_hidden = torch.randn(1, 3, HIDDEN_SIZE, requires_grad=True)
    fusion(emb, mtp_hidden).sum().backward()
    assert mtp_hidden.grad is None  # stopgrad(h_MTP)
    assert emb.grad is not None
    assert fusion.fusion_proj.weight.grad is not None


def test_mtp_block_is_frozen_and_rest_trains():
    model = _make_model()
    assert all(not p.requires_grad for p in model.mtp_layers.parameters())
    assert all(p.requires_grad for p in model.fusion.parameters())
    assert all(p.requires_grad for p in model.layers.parameters())


def test_rejects_invalid_configs():
    base = {
        "transformer_layer_config": _tl_config(),
        "mtp_transformer_config": _tl_config(num_hidden_layers=1),
        "draft_vocab_size": VERIFIER_VOCAB,
        "aux_hidden_state_layer_ids": [0, 1],
        "mask_token_id": 0,
    }
    with pytest.raises(ValueError, match="block_size"):
        DMTPDraftModel(DMTPSpeculatorConfig(**{**base, "block_size": 1}))
    with pytest.raises(ValueError, match="sample_from_anchor"):
        DMTPDraftModel(
            DMTPSpeculatorConfig(
                **{**base, "block_size": 4, "sample_from_anchor": True}
            )
        )
    mismatched = _tl_config(num_hidden_layers=1)
    mismatched.hidden_size = HIDDEN_SIZE * 2
    with pytest.raises(ValueError, match="hidden_size"):
        DMTPDraftModel(
            DMTPSpeculatorConfig(
                **{**base, "block_size": 4, "mtp_transformer_config": mismatched}
            )
        )


def test_noise_embedding_replaces_t1_slot_only():
    torch.manual_seed(0)
    model = _make_model().eval()
    seq_len = 16
    input_ids = torch.randint(1, VERIFIER_VOCAB, (1, seq_len))
    verifier_last = torch.randn(1, seq_len, HIDDEN_SIZE)
    position_ids = torch.arange(seq_len).unsqueeze(0)
    anchor_positions = torch.tensor([3, 8])

    mask_token_ids = torch.full(
        (1, anchor_positions.numel() * BLOCK_SIZE), model.mask_token_id
    )
    mask_token_ids[:, ::BLOCK_SIZE] = input_ids[:, anchor_positions]

    noise = model._build_noise_embedding(
        mask_token_ids, input_ids, anchor_positions, verifier_last, position_ids
    )

    # Non-T1 slots are the plain embeddings of the (anchor/mask) tokens.
    expected_plain = model.embed_tokens(mask_token_ids)
    for slot in (0, 2, 3):
        torch.testing.assert_close(
            noise[:, slot::BLOCK_SIZE], expected_plain[:, slot::BLOCK_SIZE]
        )

    # The T1 slot equals fusion(embed(T1), h_MTP at anchor-1).
    t1_ids = input_ids[:, anchor_positions + 1]
    mtp_hidden = model._compute_mtp_hidden(
        input_ids, verifier_last, position_ids, anchor_positions
    )
    expected_t1 = model.fusion(model.embed_tokens(t1_ids), mtp_hidden)
    torch.testing.assert_close(noise[:, 1::BLOCK_SIZE], expected_t1)


def test_mtp_hidden_uses_predictor_index():
    """h_MTP for anchor p must be the full-pass MTP output at index p-1."""
    torch.manual_seed(0)
    model = _make_model().eval()
    seq_len = 12
    input_ids = torch.randint(1, VERIFIER_VOCAB, (1, seq_len))
    verifier_last = torch.randn(1, seq_len, HIDDEN_SIZE)
    position_ids = torch.arange(seq_len).unsqueeze(0)

    all_anchors = torch.arange(1, seq_len - 1)
    full = model._compute_mtp_hidden(
        input_ids, verifier_last, position_ids, all_anchors
    )
    single = model._compute_mtp_hidden(
        input_ids, verifier_last, position_ids, torch.tensor([5])
    )
    torch.testing.assert_close(single[:, 0], full[:, 4])  # anchor 5 -> index 4


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
    assert f"position_{BLOCK_SIZE - 1}_acc_sum" in metrics
    assert "eal_sum" in metrics

    loss.backward()
    # With the zero-initialized residual, gradients reach the projection
    # directly; the norm weights' grads are gated by W (zero at step 0).
    proj_grad = model.fusion.fusion_proj.weight.grad
    assert proj_grad is not None
    assert torch.any(proj_grad != 0)
    assert all(p.grad is not None for p in model.fusion.parameters())
    attn = model.layers[0].self_attn
    assert attn.q_proj.weight.grad is not None
    assert all(p.grad is None for p in model.mtp_layers.parameters())


def test_save_includes_mtp_and_fusion_weights(tmp_path):
    model = _make_model()
    model.save_pretrained(tmp_path)
    saved = load_file(tmp_path / "model.safetensors")
    assert any(key.startswith("mtp_layers.0.") for key in saved)
    assert "fusion.fusion_proj.weight" in saved
    assert "fusion.emb_norm.weight" in saved
    # Verifier-owned weights stay omitted (full-vocab DFlash convention).
    assert "embed_tokens.weight" not in saved
    assert "verifier_lm_head.weight" not in saved


def test_config_roundtrips_mtp_transformer_config():
    config = DMTPSpeculatorConfig(
        transformer_layer_config=_tl_config(),
        mtp_transformer_config=_tl_config(num_hidden_layers=1),
        draft_vocab_size=VERIFIER_VOCAB,
        block_size=BLOCK_SIZE,
        aux_hidden_state_layer_ids=[0, 1],
        mask_token_id=0,
    )
    dumped = {k: v for k, v in config.model_dump().items() if v is not None}
    rebuilt = DMTPSpeculatorConfig(
        **{
            **dumped,
            "transformer_layer_config": config.transformer_layer_config,
        }
    )
    assert isinstance(rebuilt.mtp_transformer_config, Qwen3Config)
    assert rebuilt.mtp_transformer_config.num_hidden_layers == 1
    assert rebuilt.mtp_transformer_config.hidden_size == HIDDEN_SIZE
