from speculators.model import SpeculatorModel
from speculators.models.dmtp.core import DMTPDraftModel
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
