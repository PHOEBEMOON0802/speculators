from typing import Literal

from pydantic import Field

from speculators import SpeculatorModelConfig
from speculators.models.dflash.config import DFlashSpeculatorConfig

__all__ = [
    "DCoreSpeculatorConfig",
]


@SpeculatorModelConfig.register("dcore")
class DCoreSpeculatorConfig(DFlashSpeculatorConfig):
    """DFlash configuration with DFlare per-layer target fusion (DCore recipe).

    All structural fields are inherited from :class:`DFlashSpeculatorConfig`;
    the number of fused target layers is ``len(aux_hidden_state_layer_ids)``
    and the number of draft layers comes from
    ``transformer_layer_config.num_hidden_layers``.
    """

    speculators_model_type: Literal["dcore"] = "dcore"  # type: ignore[assignment]
    architectures: list[str] = Field(
        default_factory=lambda: ["DCoreDraftModel"],
        description="Model architectures that can load these weights",
    )
