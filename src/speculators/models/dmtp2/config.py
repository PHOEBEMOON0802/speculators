from typing import Literal

from pydantic import Field

from speculators import SpeculatorModelConfig
from speculators.models.dflash2.config import DFlash2SpeculatorConfig
from speculators.models.dmtp.config import DMTPSpeculatorConfig

__all__ = [
    "DMTP2SpeculatorConfig",
]


@SpeculatorModelConfig.register("dmtp2")
class DMTP2SpeculatorConfig(DMTPSpeculatorConfig, DFlash2SpeculatorConfig):
    """DFlash2 configuration plus the frozen-MTP first-token path (D-MTP).

    Inherits the local-convolution and candidate-selector fields from
    :class:`DFlash2SpeculatorConfig` and ``mtp_transformer_config`` (with its
    serializers) from :class:`DMTPSpeculatorConfig`.
    """

    speculators_model_type: Literal["dmtp2"] = "dmtp2"  # type: ignore[assignment]
    architectures: list[str] = Field(
        default_factory=lambda: ["DMTP2DraftModel"],
        description="Model architectures that can load these weights",
    )
    # Redeclared: the D-MTP parent precedes DFlash2 in the MRO, which would
    # otherwise reset this to DFlash's False default.
    sliding_window_non_causal: bool = Field(
        default=True,
        description="Use bidirectional masking inside sliding-window draft blocks.",
    )
