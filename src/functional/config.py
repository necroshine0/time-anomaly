from typing import Dict
from dataclasses import dataclass


@dataclass
class TrainingConfig:
    """Configuration class for training.

    Attributes:
        batch_size: Training batch size.
        learning_rate: Learning rate for optimization.
        num_epochs: Number of training epochs.
        max_seq_len: Maximum sequence length.
        overlap_ratio: Overlab ratio between sequences splits.
        accumulation_steps: Gradient accumulation steps.
        alpha: Weight of anomaly loss component (alpha * anomaly).
        weight_decay: Weight decay for optimization.
        enable_ts_train: Whether to train the time series encoder.
        mixed_precision: Whether to use AMP.
        seed: Random seed for reproducibility.
        checkpoint_dir: Path to where to checkpoint model.
        checkpoint_step: Number of epochs to checkpoint. -1 if only last.
        early_stopping_patience: Early stoppint iteration restriction
        verbose: Whether to print info
    """

    # Training parameters
    batch_size: int = 3
    learning_rate: float = 1e-4
    num_epochs: int = 1000
    max_seq_len: int = 512
    overlap_ratio: float = 0.0
    accumulation_steps: int = 1
    alpha: float = 1.0
    weight_decay: float = 1e-5
    enable_ts_train: bool = False
    mixed_precision: bool = False
    seed: int = 72
    checkpoint_dir: str = None
    checkpoint_step: int = -1
    early_stopping_patience: int = 10
    verbose: int = 1

    def to_dict(self) -> Dict[str, any]:
        return self.__dict__
