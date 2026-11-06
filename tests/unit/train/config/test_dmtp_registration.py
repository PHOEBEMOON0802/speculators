from speculators.model import SpeculatorModel
from speculators.models.dmtp.core import DMTPDraftModel
from speculators.models.dmtp2.core import DMTP2DraftModel
from speculators.train.config.resolution import _ALGORITHM_GROUP_USERS
from speculators.train.config.schema import TrainConfig


def test_speculator_type_dmtp_resolves_to_the_model():
    """``--speculator-type dmtp`` is resolved through the global registry."""
    registry = SpeculatorModel.registry

    assert registry is not None
    assert registry["dmtp"] is DMTPDraftModel


def test_dmtp_reads_the_dflash_group():
    """D-MTP is-a DFlash, so dflash-group flags must not warn under dmtp."""
    assert "dmtp" in _ALGORITHM_GROUP_USERS["dflash"]
    assert "dmtp" not in _ALGORITHM_GROUP_USERS["dspark"]
    assert "dmtp" not in _ALGORITHM_GROUP_USERS["dflash2"]


def test_dmtp_derived_defaults():
    # Blog recipe: 5 draft layers, block 8 (1 bonus + T1 + 6 masks), CE with
    # the original fixed exponential position decay (not D-PACE).
    flat = TrainConfig(speculator_type="dmtp").flatten()
    assert flat["num_layers"] == 5
    assert flat["block_size"] == 8
    assert flat["loss_fn"] == "ce"
    assert flat["per_position_loss_weight"] == "fixed-exp-decay"


def test_speculator_type_dmtp2_resolves_to_the_model():
    registry = SpeculatorModel.registry
    assert registry is not None
    assert registry["dmtp2"] is DMTP2DraftModel


def test_dmtp2_reads_the_dflash_and_dflash2_groups():
    """D-MTP2 is-a DFlash2: both backbone and selector knobs must apply."""
    assert "dmtp2" in _ALGORITHM_GROUP_USERS["dflash"]
    assert "dmtp2" in _ALGORITHM_GROUP_USERS["dflash2"]
    assert "dmtp2" not in _ALGORITHM_GROUP_USERS["dspark"]


def test_dmtp2_derived_defaults():
    # Mirrors dflash2 (its warm-start source): 5 layers, block 8, KL loss,
    # fixed exponential decay, bidirectional sliding-window blocks.
    flat = TrainConfig(speculator_type="dmtp2").flatten()
    assert flat["num_layers"] == 5
    assert flat["block_size"] == 8
    assert flat["loss_fn"] == "kl_div"
    assert flat["per_position_loss_weight"] == "fixed-exp-decay"
    assert flat["sliding_window_non_causal"] is True
