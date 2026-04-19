import os
import random
import itertools
import numpy as np
from tqdm import tqdm
from typing import Union, Optional

import torch
from torch import nn
import torch.nn.functional as F

from sklearn.metrics import (
    f1_score, precision_score, recall_score, roc_auc_score
)

from .config import TrainingConfig


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def unwrap_batch(batch, device):
    if isinstance(device, str):
        device = torch.device(device)

    time_series = batch['time_series'].to(device)  # (B, max_seq_len, num_features)
    masked_time_series = batch['masked_time_series'].to(device)
    mask = batch['mask'].to(device)  # (B, max_seq_len)
    labels = batch['labels'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    return time_series, masked_time_series, mask, labels, attention_mask


def compute_batch_loss(model, batch, device):
    time_series, masked_time_series, mask, labels, attention_mask = unwrap_batch(batch, device)
    local_embeddings = model(masked_time_series, attention_mask & (~mask.bool()))
    recon_loss = model.masked_reconstruction_loss(local_embeddings, time_series, mask)
    anomaly_loss = model.anomaly_detection_loss(local_embeddings, labels)
    return recon_loss, anomaly_loss


def predict_batch(model, batch, device, anomaly_key="logits"):
    time_series, masked_time_series, mask, labels, attention_mask = unwrap_batch(batch, device)
    batch_size, seq_len, num_features = time_series.shape

    with torch.no_grad():
        # NOTE: time_series, а не masked_time_series как при обучении! См. TimeRCD.py (тестер)
        local_embeddings = model(time_series, attention_mask)

        # Get reconstruction
        reconstructed = model.reconstruction_head(local_embeddings)
        reconstructed = reconstructed.view(batch_size, seq_len, num_features)  # (B, seq_len, num_features)

        # Get anomaly predictions
        if hasattr(model, "anomaly_head"):
            logits = model.anomaly_head(local_embeddings)
            logits = torch.mean(logits, dim=-2)  # (B, seq_len, 2)
            anomaly_probs = F.softmax(logits, dim=-1)[..., 1]  # Probability of anomaly (B, seq_len)
            anomaly_logits = logits[..., 1] - logits[..., 0]  # Anomaly logits (B, seq_len)

            if anomaly_key == "probs":
                anomaly_scores = anomaly_probs
            elif anomaly_key == "logits":
                anomaly_scores = anomaly_logits
            else:
                raise ValueError(f"Invalid anomaly_key value: {anomaly_key}")
        else:
            if anomaly_key != "logits":
                raise ValueError(f"In reconstruction only setting anomaly_key value must be 'logits', but got: {anomaly_key}")
            anomaly_scores = ((reconstructed - time_series) ** 2).mean(dim=-1)

        return {
            'original':       time_series.cpu(),
            'masked':         masked_time_series.cpu(),
            'reconstructed':  reconstructed.cpu(),
            'mask':           mask.bool().cpu(),
            'anomaly_scores': anomaly_scores.cpu(),
            'true_labels':    labels.cpu(),
            'attention_mask': attention_mask.cpu(),
        }


def train_epoch(
        config: TrainingConfig,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        loader: torch.utils.data.DataLoader,
        device: Union[torch.device, str],
        epoch: int,
        scaler: Optional[torch.cuda.amp.GradScaler] = None
    ) -> float:
    """Train for one epoch with multiple pretraining tasks."""
    model.train()

    total_loss = 0.0
    total_recon_loss = 0.0
    total_anomaly_loss = 0.0
    num_batches = 0
    log_freq = max(1, int(getattr(config, 'log_freq', 10)))
    accumulation_steps = getattr(config, 'accumulation_steps', 1)

    for batch_idx, batch in enumerate(loader):
        if batch_idx % 10 == 0:
            torch.cuda.empty_cache()

        if batch_idx % accumulation_steps == 0:
            optimizer.zero_grad()

        if config.mixed_precision and scaler is not None:
            with torch.amp.autocast('cuda'):
                recon_loss, anomaly_loss = compute_batch_loss(model, batch, device)
            total_loss_batch = recon_loss + config.alpha * anomaly_loss
            scaler.scale(total_loss_batch).backward()
        else:
            recon_loss, anomaly_loss = compute_batch_loss(model, batch, device)
            total_loss_batch = recon_loss + config.alpha * anomaly_loss
            total_loss_batch.backward()

        if (batch_idx + 1) % accumulation_steps == 0:
            if config.mixed_precision and scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

        # Accumulate losses
        total_loss += total_loss_batch.item()
        total_recon_loss += recon_loss.item()
        total_anomaly_loss += anomaly_loss.item()
        num_batches += 1

        # Log progress based on log_freq
        if batch_idx % log_freq == 0 and config.verbose == 2:
            print(f"Epoch {epoch}, Batch {batch_idx}/{len(loader)}")
            print(f"  Total Loss: {total_loss_batch.item():.4f}")
            print(f"  Recon Loss: {recon_loss.item():.4f}")
            print(f"  Anomaly Loss: {anomaly_loss.item():.4f}")

    avg_loss = total_loss / num_batches
    avg_recon_loss = total_recon_loss / num_batches
    avg_anomaly_loss = total_anomaly_loss / num_batches

    if config.verbose == 2 or (config.verbose == 1 and epoch % config.checkpoint_step == 0):
        print(f"\nEpoch {epoch} completed:")
        print(f"  Average Total Loss: {avg_loss:.4f}")
        print(f"  Average Recon Loss: {avg_recon_loss:.4f}")
        print(f"  Average Anomaly Loss: {avg_anomaly_loss:.4f}")
    return avg_loss, avg_recon_loss, avg_anomaly_loss


def evaluate_epoch(
        config: TrainingConfig,
        model: nn.Module,
        loader: torch.utils.data.DataLoader,
        device: Union[torch.device, str],
        epoch: int,
    ) -> float:
    """Evaluate model on test dataset."""
    model.eval()

    total_loss = 0.0
    total_recon_loss = 0.0
    total_anomaly_loss = 0.0
    num_batches = 0
    test_batch_limit = min(len(loader), getattr(config, 'test_batch_limit', len(loader)))

    with torch.no_grad():
        for batch in itertools.islice(loader, test_batch_limit):
            recon_loss, anomaly_loss = compute_batch_loss(model, batch, device)
            total_loss_batch = recon_loss + config.alpha * anomaly_loss
            total_loss += total_loss_batch.item()
            total_recon_loss += recon_loss.item()
            total_anomaly_loss += anomaly_loss.item()
            num_batches += 1

    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    avg_recon_loss = total_recon_loss / num_batches if num_batches > 0 else 0.0
    avg_anomaly_loss = total_anomaly_loss / num_batches if num_batches > 0 else 0.0

    if config.verbose == 2 or (config.verbose == 1 and epoch % config.checkpoint_step == 0):
        print(f"\nEpoch {epoch} validated:")
        print(f"  Average Total Loss: {avg_loss:.4f}")
        print(f"  Average Recon Loss: {avg_recon_loss:.4f}")
        print(f"  Average Anomaly Loss: {avg_anomaly_loss:.4f}")
    return avg_loss, avg_recon_loss, avg_anomaly_loss


def save_checkpoint(
        config: TrainingConfig,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        avg_loss: float,
        scaler: Optional[torch.cuda.amp.GradScaler] = None,
        is_best: bool = False
    ) -> None:
    """Save model checkpoint."""
    if config.checkpoint_dir is None:
        return

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': avg_loss,
        'model_config': model.config.to_dict(),
        'training_config': config.to_dict()
    }

    if scaler is not None:
        checkpoint['scaler_state_dict'] = scaler.state_dict()

    checkpoint_step = int(getattr(config, 'checkpoint_step', -1))

    os.makedirs(config.checkpoint_dir, exist_ok=True)

    # Always save the latest checkpoint
    latest_path = os.path.join(config.checkpoint_dir, "pretrain_checkpoint_latest.pth")
    torch.save(checkpoint, latest_path)

    # Save the checkpoint at specified frequency
    if checkpoint_step > 0 and (epoch % checkpoint_step == 0 or epoch == config.num_epochs - 1):
        save_path = os.path.join(config.checkpoint_dir, f"pretrain_checkpoint_epoch_{epoch}.pth")
        torch.save(checkpoint, save_path)
        if config.verbose == 2:
            print(f"Checkpoint saved to {save_path} (epoch {epoch}, val_loss: {avg_loss:.4f})")

    # Save best model if this is the best validation loss
    if is_best:
        best_path = os.path.join(config.checkpoint_dir, "pretrain_checkpoint_best.pth")
        torch.save(checkpoint, best_path)
        if config.verbose == 2:
            print(f"New best model saved to {best_path} (epoch {epoch}, val_loss: {avg_loss:.4f})")

        # Save just the time series encoder for downstream tasks
        ts_encoder_state = model.ts_encoder.state_dict()

        best_encoder_path = os.path.join(config.checkpoint_dir, "pretrained_ts_encoder.pth")
        torch.save(ts_encoder_state, best_encoder_path)


def load_checkpoint(
        config: TrainingConfig,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scaler: Optional[torch.cuda.amp.GradScaler] = None,
        checkpoint_path: Optional[str] = None,
        device: Union[torch.device, str] = "cpu",
        strict: bool = True,
    ) -> dict:
    """
    Load model checkpoint and return state (epoch, loss, etc.).

    Args:
        config: TrainingConfig from which checkpoint_dir is read if checkpoint_path is None.
        model: model instance (already constructed).
        optimizer: optimizer instance.
        scaler: optional GradScaler (if used with AMP).
        checkpoint_path: explicit path to checkpoint file. If None, tries to load "pretrain_checkpoint_latest.pth".
        device: device to load tensors to.
        strict: whether to strict-load model state_dict.

    Returns:
        dict with keys like 'epoch', 'loss', 'model_config', 'training_config', etc.
    """
    if checkpoint_path is None:
        if config.checkpoint_dir is None:
            raise ValueError("config.checkpoint_dir is None and checkpoint_path is not provided.")
        checkpoint_path = os.path.join(config.checkpoint_dir, "pretrain_checkpoint_latest.pth")

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Load model state_dict
    model_state_dict = checkpoint["model_state_dict"]
    model.load_state_dict(model_state_dict, strict=strict)

    # Load optimizer (if present)
    if "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    # Load scaler (if present and passed)
    if scaler is not None:
        if "scaler_state_dict" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler_state_dict"])
        else:
            print("Warning: scaler_state_dict not found in checkpoint, AMP scaler not restored.")

    # Return metadata we stored
    metadata_keys = ["epoch", "loss", "model_config", "training_config"]
    metadata = {k: checkpoint.get(k) for k in metadata_keys if k in checkpoint}

    if config.verbose == 2:
        print(
            f"Checkpoint loaded from {checkpoint_path} "
            f"(epoch {metadata.get('epoch', '?')}, "
            f"loss {metadata.get('loss', '?')})"
        )

    return metadata


def train_worker(
        config: TrainingConfig,
        model: nn.Module,
        train_loader: torch.utils.data.DataLoader,
        valid_loader: torch.utils.data.DataLoader = None,
        device: Union[torch.device, str] = "cpu",
        checkpoint_path: str = None,
    ) -> None:
    """Training worker function for each process."""
    set_seed(config.seed)

    # Init instances
    model = model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    scaler = torch.amp.GradScaler() if config.mixed_precision else None

    # Load checkpoint if resuming/continuing from previous checkpoint
    if checkpoint_path is not None:
        checkpoint = load_checkpoint(config, model, optimizer, scaler, checkpoint_path, device, strict=True)
        start_epoch = int(checkpoint.get('epoch', 0)) + 1
        if config.verbose == 2:
            print(f"Resuming from epoch {start_epoch}...")
    else:
        start_epoch = 1
        if config.verbose == 2:
            print(f"Starting pretraining from scratch for {config.num_epochs} epochs...")

    # Early stopping parameters
    best_valid_loss = float('inf')
    patience_counter = 0
    early_stopping_patience = getattr(config, 'early_stopping_patience')

    if config.verbose == 2:
        print(f"Total training batches per process: {len(train_loader)}")
        print(f"Early stopping patience: {early_stopping_patience} epochs")

    train_losses = {"total": [], "reconstruction": [], "anomaly": []}
    valid_losses = {"total": [], "reconstruction": [], "anomaly": []}
    epochs = []
    for epoch in tqdm(range(start_epoch, config.num_epochs + 1)):
        train_total, train_recon, train_anomaly = train_epoch(config, model, optimizer, train_loader, device, epoch, scaler)
        train_losses["total"].append(train_total)
        train_losses["reconstruction"].append(train_recon)
        train_losses["anomaly"].append(train_anomaly)

        valid_total, valid_recon, valid_anomaly = evaluate_epoch(config, model, valid_loader, device, epoch)
        valid_losses["total"].append(valid_total)
        valid_losses["reconstruction"].append(valid_recon)
        valid_losses["anomaly"].append(valid_anomaly)

        epochs.append(epoch)

        # Check if this is the best model so far
        is_best = valid_total < best_valid_loss
        if is_best:
            best_valid_loss = valid_total
            patience_counter = 0
            if config.verbose == 2:
                print(f"\nNew best validation loss: {best_valid_loss:.4f}")
        else:
            patience_counter += 1
            if config.verbose == 2:
                print(f"\nValidation loss did not improve. Patience: {patience_counter}/{early_stopping_patience}")

        # Save checkpoint with best model flag
        save_checkpoint(config, model, optimizer, epoch, valid_total, scaler, is_best)

        # Early stopping check
        if patience_counter >= early_stopping_patience:
            if config.verbose == 2:
                print(f"Early stopping triggered after {epoch + 1} epochs")
                print(f"Best validation loss: {best_valid_loss:.4f}")
            break

    return {"train_losses": train_losses, "valid_losses": valid_losses, "epochs": epochs}


def benchmark(
        model: nn.Module,
        loader: torch.utils.data.DataLoader,
        device: Union[torch.device, str],
        anomaly_key: str = "logits",
    ):
    # FIXME: реализовать window_size + stride (https://chat.deepseek.com/a/chat/s/60a44086-6405-4cc7-870b-378fe769485a)
    # Нужно брать перекрывающиеся батчи, чтобы правильноу улавливать аномалии на границах батчей
    assert anomaly_key in ["probs", "logits"]
    model.eval()

    anomaly_scores = []
    true_labels = []
    original = []
    reconstructed = []
    for batch in loader:
        result = predict_batch(model, batch, device, anomaly_key)
        attn_mask = result['attention_mask']
        anomaly_scores.append(result['anomaly_scores'][attn_mask].numpy())
        true_labels.append(result['true_labels'][attn_mask].numpy())
        original.append(result['original'][attn_mask].numpy())
        reconstructed.append(result['reconstructed'][attn_mask].numpy())

    anomaly_scores = np.concatenate(anomaly_scores)
    true_labels = np.concatenate(true_labels)
    original = np.concatenate(original)
    reconstructed = np.concatenate(reconstructed)

    if len(true_labels.shape) == 2:
        true_labels = true_labels.reshape(-1)

    if len(anomaly_scores.shape) == 2:
        if anomaly_scores.shape[1] != 1:
            raise NotImplementedError()
        anomaly_scores = anomaly_scores.reshape(-1)
    
    metrics = {}

    diff = original - reconstructed
    metrics["RMSE"] = np.sqrt((diff ** 2).mean().mean())
    metrics["MAE"] = np.abs(diff).mean().mean()
    metrics["AUC-ROC"] = roc_auc_score(true_labels, anomaly_scores)
    if anomaly_key == "probs":
        for t in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
            pred_labels = (anomaly_scores > t).astype(int)
            metrics[f"F1, t>{t}"]        = f1_score(true_labels, pred_labels)
            metrics[f"Precision, t>{t}"] = precision_score(true_labels, pred_labels)
            metrics[f"Recall, t>{t}"]    = recall_score(true_labels, pred_labels)
    else:
        # https://github.com/thu-sail-lab/Time-RCD/blob/tsb-ad-integration/main.py#L305
        mean = np.mean(anomaly_scores)
        std = np.std(anomaly_scores)
        pred_labels = (anomaly_scores > (mean + 3 * std))
        metrics["F1"]        = f1_score(true_labels, pred_labels)
        metrics["Precision"] = precision_score(true_labels, pred_labels)
        metrics["Recall"]    = recall_score(true_labels, pred_labels)

    return metrics
