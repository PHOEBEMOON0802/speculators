from typing import Any, Literal

from pydantic import Field, field_serializer, field_validator
from transformers import AutoConfig, PretrainedConfig
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config

from speculators import SpeculatorModelConfig
from speculators.models.dflash.config import DFlashSpeculatorConfig

__all__ = [
    "DMTPSpeculatorConfig",
]


@SpeculatorModelConfig.register("dmtp")
class DMTPSpeculatorConfig(DFlashSpeculatorConfig):
    """DFlash configuration plus the frozen-MTP first-token path (D-MTP).

    ``transformer_layer_config`` describes the DFlash draft decoder as usual.
    ``mtp_transformer_config`` describes the *verifier's* decoder architecture,
    which the frozen MTP block is built from -- the two may differ (e.g. a
    qwen3 draft on a qwen3_5 verifier); they only share hidden size and vocab.
    """

    speculators_model_type: Literal["dmtp"] = "dmtp"  # type: ignore[assignment]
    architectures: list[str] = Field(
        default_factory=lambda: ["DMTPDraftModel"],
        description="Model architectures that can load these weights",
    )
    mtp_transformer_config: PretrainedConfig = Field(
        default_factory=Qwen3Config,
        description="Verifier decoder architecture config used to build the "
        "frozen MTP block (its weights are extracted from the verifier's "
        "native mtp.* keys).",
    )

    @field_serializer("mtp_transformer_config", check_fields=False)
    def serialize_mtp_transformer_config(
        self, value: PretrainedConfig
    ) -> dict[str, Any]:
        """Serialize the MTP decoder config using the Transformers diff format."""
        return value.to_diff_dict()

    @field_validator("mtp_transformer_config", mode="before", check_fields=False)
    @classmethod
    def validate_mtp_transformer_config(cls, value: Any) -> PretrainedConfig:
        """Resolve a serialized MTP decoder config dict back to its class."""
        if not isinstance(value, dict):
            return value
        if "model_type" in value:
            config_class = AutoConfig.for_model(value["model_type"]).__class__
        else:
            config_class = Qwen3Config
        return config_class(**value)
