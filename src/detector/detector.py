import warnings
from typing import Optional
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import DetectorConfig
from .modules import (
    TimeSeriesEncoderFlatten,
    TimeSeriesEncoderIndependent,
    TimeSeriesEncoderMixing,
)


ENCODERS = {
    "flatten": TimeSeriesEncoderFlatten,
    "independent": TimeSeriesEncoderIndependent,
    "mixing": TimeSeriesEncoderMixing,
}


class Detector(nn.Module):
    """Model for time series pretraining with masked reconstruction."""
    def __init__(self, config: DetectorConfig):
        super().__init__()
        self.config = config

        # Extract TimeSeriesEncoder parameters from config
        encoder_class = ENCODERS.get(config.channel_strategy, None)
        if encoder_class is None:
            raise ValueError(f"Unknown channel_strategy={config.channel_strategy}")

        self.ts_encoder = encoder_class(
            d_model=config.d_model,
            d_proj=config.d_proj,
            patch_size=config.patch_size,
            num_layers=config.num_layers,
            num_heads=config.num_heads,
            d_ff_dropout=config.d_ff_dropout,
            use_rope=config.use_rope,
            num_features=config.num_features,
            activation=config.activation
        )

        # Masked reconstruction head
        self.reconstruction_head = nn.Sequential(
            nn.Linear(config.d_proj, config.d_proj * 4),
            nn.GELU(),
            nn.Dropout(config.d_ff_dropout),
            nn.Linear(config.d_proj * 4, config.d_proj * 4),
            nn.GELU(),
            nn.Dropout(config.d_ff_dropout),
            nn.Linear(config.d_proj * 4, 1)  # (B, seq_len, num_features, 1)
        )

    def forward(self,
            time_series: torch.Tensor,
            mask: Optional[torch.Tensor] = None
        ):
        local_embeddings = self.ts_encoder(time_series, mask)
        return local_embeddings

    def masked_reconstruction_loss(self,
                                   local_embeddings: torch.Tensor,  # (B, seq_len, num_features, d_proj)
                                   original_time_series: torch.Tensor,  # (B, seq_len, num_features),
                                   mask: torch.Tensor  # (B, seq_len)
                                   ) -> torch.Tensor:
        """Compute masked reconstruction loss."""
        batch_size, seq_len, num_features = original_time_series.shape

        # local_embeddings: [B, seq_len, num_features, d_proj]
        reconstructed = self.reconstruction_head(local_embeddings)  # (B, seq_len, num_features, 1)
        reconstructed = reconstructed.view(batch_size, seq_len, num_features)

        mask_expanded = mask.bool().unsqueeze(-1).expand(-1, -1, num_features)  # (B, seq_len, num_features)
        reconstruction_loss = F.mse_loss(
            reconstructed[mask_expanded],
            original_time_series[mask_expanded]
        )
        return reconstruction_loss
