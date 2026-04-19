from typing import Dict
from dataclasses import dataclass


@dataclass
class DetectorConfig:
    """Configuration for Detector encoder.

    Attributes:
        d_model: Dimension of model hidden states.
        d_proj: Dimension of projection layer.
        patch_size: Size of time series patches.
        num_layers: Number of transformer layers.
        num_heads: Number of attention heads.
        d_ff_dropout: Dropout rate for feed-forward networks.
        use_rope: Whether to use Rotary Position Embedding.
        activation: Activation function name.
        num_features: Number of input features.
        use_anomaly_head: Whether to use and train anomaly_head
    """
    d_model: int = 512
    d_proj: int = 256
    patch_size: int = 4
    num_query_tokens: int = 1
    num_layers: int = 8
    num_heads: int = 8
    d_ff_dropout: float = 0.1
    use_rope: bool = True
    activation: str = "gelu"
    num_features: int = 1
    use_anomaly_head: bool = True

    def to_dict(self) -> Dict[str, any]:
        return self.__dict__
