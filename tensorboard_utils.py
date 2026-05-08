from src import DetectorConfig, TrainingConfig
from torch.utils.tensorboard import SummaryWriter


def init_tensorboard(path: str):
    log_dir = f"tensorboard_runs/{path}"
    writer = SummaryWriter(log_dir)
    return writer


def log_hparams(
        writer: SummaryWriter,
        detector_config: DetectorConfig,
        training_config: TrainingConfig,
        metrics: dict,
    ):
    configs = {
        "detector": detector_config.to_dict(),
        "training": training_config.to_dict(),
    }

    hparams = {}
    for key, config in configs.items():
        hparams.update({f"{key}_{k}": v for k, v in config.items()})
    writer.add_hparams(hparams, metrics)


def log_losses(writer: SummaryWriter, losses: dict):
    epochs = losses["epochs"]
    train_losses = losses["train_losses"]
    valid_losses = losses["valid_losses"]

    for i, epoch in enumerate(epochs):
        writer.add_scalar("Loss/Reconstruction/train", train_losses["reconstruction"][i], epoch)
        writer.add_scalar("Loss/Reconstruction/valid", valid_losses["reconstruction"][i], epoch)
