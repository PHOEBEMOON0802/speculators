"""DCore draft model unit tests.

Covers: registry wiring, fusion-weight initialization parity with the released
``ernie-research/DCore-*`` reference implementation (``dflare.py``), per-layer
fusion shapes, heterogeneous-KV attention conservation against DFlash, default
target-layer-id resolution, and checkpoint key alignment.
"""

from __future__ import annotations

import pytest
import torch
from safetensors.torch import load_file
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3Config,
    Qwen3RotaryEmbedding,
)

from speculators.config import SpeculatorsConfig, VerifierConfig
from speculators.losses import eager as eager_losses
from speculators.model import SpeculatorModel
from speculators.models.dcore import DCoreDraftModel, DCoreSpeculatorConfig
from speculators.models.dcore.core import default_dcore_target_layer_ids
from speculators.models.dcore.model_definitions import (
    Qwen3DCoreAttention,
    Qwen3DCoreDecoderLayer,
)
from speculators.models.dflash.model_definitions import Qwen3DFlashAttention
from speculators.proposals.greedy import GreedyTokenProposalConfig

VERIFIER_VOCAB = 64
HIDDEN_SIZE = 16
NUM_TARGET_LAYERS = 3
NUM_DRAFT_LAYERS = 2


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


def _make_model(draft_vocab_size: int = VERIFIER_VOCAB) -> DCoreDraftModel:
    config = DCoreSpeculatorConfig(
        transformer_layer_config=_tl_config(),
        draft_vocab_size=draft_vocab_size,
        block_size=4,
        aux_hidden_state_layer_ids=list(range(NUM_TARGET_LAYERS)),
        mask_token_id=0,
        speculators_config=SpeculatorsConfig(
            algorithm="dcore",
            proposal_methods=[GreedyTokenProposalConfig(speculative_tokens=3)],
            default_proposal_method="greedy",
            verifier=VerifierConfig(name_or_path="dummy", architectures=[]),
        ),
    )
    model = DCoreDraftModel(config)
    if model.use_draft_vocab:
        t2d = torch.zeros(model.verifier_vocab_size, dtype=torch.bool)
        t2d[:draft_vocab_size] = True
        model.load_vocab_mappings(t2d, torch.arange(draft_vocab_size, dtype=torch.long))
    return model.eval()


def test_registry_contains_dcore():
    assert SpeculatorModel.registry is not None
    assert SpeculatorModel.registry["dcore"] is DCoreDraftModel
    assert DCoreDraftModel.config_class is DCoreSpeculatorConfig


def test_fusion_init_matches_reference():
    """Zeros everywhere, one 2.0 per row at t = int(d/D*T) — per dflare.py."""
    model = _make_model()
    weights = model.layer_fusion_weights
    assert weights.shape == (NUM_DRAFT_LAYERS, NUM_TARGET_LAYERS)
    for d_idx in range(NUM_DRAFT_LAYERS):
        expected_t = min(
            NUM_TARGET_LAYERS - 1,
            int(d_idx / NUM_DRAFT_LAYERS * NUM_TARGET_LAYERS),
        )
        row = weights[d_idx]
        assert row[expected_t].item() == pytest.approx(2.0)
        assert row.sum().item() == pytest.approx(2.0)  # all other entries zero


def test_model_has_no_shared_fc():
    model = _make_model()
    assert not hasattr(model, "fc")
    # hidden_norm is retained as the shared post-fusion RMSNorm
    assert hasattr(model, "hidden_norm")
    state_keys = set(model.state_dict())
    assert "layer_fusion_weights" in state_keys
    assert "hidden_norm.weight" in state_keys
    assert not any(key.startswith("fc.") for key in state_keys)


def test_checkpoint_keys_align_with_released_dcore():
    """Attribute paths must match ernie-research/DCore-* checkpoints."""
    model = _make_model()
    state_keys = set(model.state_dict())
    for layer_idx in range(NUM_DRAFT_LAYERS):
        prefix = f"layers.{layer_idx}.self_attn"
        assert f"{prefix}.k_proj_target.weight" in state_keys
        assert f"{prefix}.v_proj_target.weight" in state_keys
        assert f"{prefix}.k_proj.weight" in state_keys
        assert f"{prefix}.v_proj.weight" in state_keys


def test_fuse_target_hidden_shapes_and_grad():
    model = _make_model()
    seq_len = 6
    hidden = torch.randn(1, seq_len, NUM_TARGET_LAYERS * HIDDEN_SIZE)
    fused = model._fuse_target_hidden(hidden)
    assert isinstance(fused, list)
    assert len(fused) == NUM_DRAFT_LAYERS
    for tensor in fused:
        assert tensor.shape == (1, seq_len, HIDDEN_SIZE)
    # fusion weights receive gradient through the fused context
    loss = sum(t.sum() for t in fused)
    loss.backward()
    assert model.layer_fusion_weights.grad is not None
    assert torch.any(model.layer_fusion_weights.grad != 0)


def test_fusion_softmax_uniform_rows_average_layers():
    """With uniform fusion logits, fusion equals the plain mean over T layers."""
    model = _make_model()
    with torch.no_grad():
        model.layer_fusion_weights.zero_()
        model.hidden_norm.weight.fill_(1.0)
    seq_len = 4
    per_layer = [torch.randn(1, seq_len, HIDDEN_SIZE) for _ in range(NUM_TARGET_LAYERS)]
    hidden = torch.cat(per_layer, dim=-1)
    fused = model._fuse_target_hidden(hidden)
    mean = torch.stack(per_layer).mean(dim=0)
    expected = model.hidden_norm(mean)
    for tensor in fused:
        torch.testing.assert_close(tensor, expected, rtol=1e-4, atol=1e-5)


def test_dcore_attention_conserves_dflash_when_projections_tied():
    """kt/vt copied from k/v ⇒ DCore attention == DFlash attention."""
    torch.manual_seed(0)
    config = _tl_config()
    dflash_attn = Qwen3DFlashAttention(config, layer_idx=0).eval()
    dcore_attn = Qwen3DCoreAttention(config, layer_idx=0).eval()
    missing, unexpected = dcore_attn.load_state_dict(
        dflash_attn.state_dict(), strict=False
    )
    assert not unexpected
    assert set(missing) == {"k_proj_target.weight", "v_proj_target.weight"}
    with torch.no_grad():
        dcore_attn.k_proj_target.weight.copy_(dflash_attn.k_proj.weight)
        dcore_attn.v_proj_target.weight.copy_(dflash_attn.v_proj.weight)

    block_len, ctx_len = 4, 5
    hidden = torch.randn(1, block_len, HIDDEN_SIZE)
    target_hidden = torch.randn(1, ctx_len, HIDDEN_SIZE)
    rotary = Qwen3RotaryEmbedding(config)
    position_ids = torch.arange(ctx_len + block_len).unsqueeze(0)
    position_embeddings = rotary(hidden, position_ids)

    out_dflash, _ = dflash_attn(
        hidden_states=hidden,
        target_hidden=target_hidden,
        position_embeddings=position_embeddings,
        attention_mask=None,
    )
    out_dcore, _ = dcore_attn(
        hidden_states=hidden,
        target_hidden=target_hidden,
        position_embeddings=position_embeddings,
        attention_mask=None,
    )
    torch.testing.assert_close(out_dcore, out_dflash)


def test_decoder_layer_uses_dcore_attention():
    layer = Qwen3DCoreDecoderLayer(_tl_config(), layer_idx=0)
    assert isinstance(layer.self_attn, Qwen3DCoreAttention)


@pytest.mark.parametrize(
    ("num_verifier_layers", "expected"),
    [
        (36, [1, 5, 9, 13, 17, 21, 25, 29, 33]),  # Qwen3-8B, released ckpt ids
    ],
)
def test_default_target_layer_ids(num_verifier_layers, expected):
    assert default_dcore_target_layer_ids(num_verifier_layers) == expected


def test_default_target_layer_ids_rejects_shallow_verifier():
    with pytest.raises(ValueError, match="target-layer-ids"):
        default_dcore_target_layer_ids(8)


def test_save_roundtrip_restores_fusion_and_target_projections(tmp_path):
    model = _make_model()
    with torch.no_grad():
        model.layer_fusion_weights.add_(torch.randn_like(model.layer_fusion_weights))
    model.save_pretrained(tmp_path)

    saved = load_file(tmp_path / "model.safetensors")
    assert "layer_fusion_weights" in saved
    assert "layers.0.self_attn.k_proj_target.weight" in saved
    # full-vocab: verifier-owned weights are omitted from the checkpoint
    assert "embed_tokens.weight" not in saved
    assert "verifier_lm_head.weight" not in saved
    torch.testing.assert_close(
        saved["layer_fusion_weights"], model.layer_fusion_weights.detach()
    )


def test_full_forward_dpace_loss_cpu_smoke():
    """End-to-end CPU forward: anchored blocks, D-PACE weighted CE, backward."""
    torch.manual_seed(0)
    model = _make_model().train()
    # Verifier-owned weights are normally loaded from the verifier; replace the
    # NaN placeholders with real values (mirrors integration conftest).
    with torch.no_grad():
        for param in model.parameters():
            if param.isnan().any():
                torch.nn.init.normal_(param, mean=0.0, std=0.02)
    torch.nn.init.normal_(model.verifier_lm_head.weight)
    torch.nn.init.ones_(model.verifier_norm.weight)
    seq_len = 32
    hidden = torch.randn(1, seq_len, NUM_TARGET_LAYERS * HIDDEN_SIZE)
    verifier_last = torch.randn(1, seq_len, HIDDEN_SIZE)
    input_ids = torch.randint(0, VERIFIER_VOCAB, (1, seq_len))
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
        per_position_loss_weight="dpace",
        dpace_alpha=0.5,
    )
    assert torch.isfinite(loss)
    assert "eal_sum" in metrics
    loss.backward()
    assert model.layer_fusion_weights.grad is not None
    assert torch.any(model.layer_fusion_weights.grad != 0)
    attn = model.layers[0].self_attn
    assert attn.k_proj_target.weight.grad is not None
    assert torch.any(attn.k_proj_target.weight.grad != 0)
