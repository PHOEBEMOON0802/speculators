from speculators.model import SpeculatorModel
from speculators.models.dcore.core import DCoreDraftModel
from speculators.train.config.resolution import _ALGORITHM_GROUP_USERS


def test_speculator_type_dcore_resolves_to_the_model():
    """``--speculator-type dcore`` is resolved through the global registry.

    Without the ``@register`` decorator the flag parses fine and then fails at
    model construction, so pin the lookup rather than the decorator.
    """
    registry = SpeculatorModel.registry

    assert registry is not None
    assert registry["dcore"] is DCoreDraftModel


def test_dcore_reads_the_dflash_group():
    """DCore is-a DFlash, so a dflash-group flag must not warn under dcore.

    The group table decides whether a flag is silently ignored; leaving dcore
    out of the dflash entry would make every backbone knob (block_size,
    per_position_loss_weight, dpace_alpha, max_anchors) a no-op warning on a
    dcore run.
    """
    assert "dcore" in _ALGORITHM_GROUP_USERS["dflash"]
    assert "dcore" not in _ALGORITHM_GROUP_USERS["dspark"]
    assert "dcore" not in _ALGORITHM_GROUP_USERS["dflash2"]
