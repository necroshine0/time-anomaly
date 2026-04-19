import os
import numpy as np
from typing import Tuple

import torch
from torch.utils.data import TensorDataset, DataLoader

from sklearn.preprocessing import (
    MaxAbsScaler,
    MinMaxScaler,
)


def split_ts_sequences(*args, max_seq_len=512, overlap_ratio=0.0):
    """
    Split *args with shape (n, dim) into sequences of length max_seq_len with overlap.

    Args:
        *args: array of shape (n, dim)
        max_seq_len: int, length of each sequence
        overlap_ratio: float in [0.0, 1.0], proportion of overlap between sequences

    Returns:
        *args: array of shape (num_windows, max_seq_len, dim)
    """
    n = args[0].shape[0]
    # проверить, что все массивы одной длины по времени
    for i, arr in enumerate(args):
        if arr.shape[0] != n:
            raise ValueError(
                f"Array 0 has length {n}, array {i} has length {arr.shape[0]}"
            )

    if n < max_seq_len:
        raise ValueError(f"Length {n} is shorter than max_seq_len {max_seq_len}")

    if not (0.0 <= overlap_ratio <= 1.0):
        raise ValueError("overlap_ratio must be in [0.0, 1.0]")

    step = int(max(1, max_seq_len * (1.0 - overlap_ratio)))
    starts = np.arange(0, n - max_seq_len + 1, step)
    idx = np.expand_dims(starts, 1) + np.arange(max_seq_len)  # (num_windows, max_seq_len)

    result = [arr[idx] for arr in args]  # (num_windows, max_seq_len, dim)
    if len(result) == 1:
        return result[0]
    return tuple(result)


def create_random_mask(time_series: torch.Tensor,  #(B, max_seq_len, num_features)
                       attention_mask: torch.Tensor,  # (B, max_seq_len)
                       mask_ratio: float = 0.15
                    ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Create random mask for time series patches, only masking valid sequence parts."""
    batch_size, seq_len, num_features = time_series.shape
    patch_size = 4 # DEFAULT VALUE HARDCODE

    mask = torch.zeros(batch_size, seq_len)  # (B, max_seq_len)

    for i in range(batch_size):
        # Get valid sequence length for this sample
        valid_length = attention_mask[i].sum().item()

        # Calculate number of patches in valid sequence
        num_valid_patches = (valid_length - 1) // patch_size + 1
        num_masked = int(num_valid_patches * mask_ratio)

        if num_masked > 0:
            # Only select patches from valid sequence
            masked_patches = torch.randperm(num_valid_patches)[:num_masked]
            for j in masked_patches:
                start_idx = j * patch_size
                end_idx = min((j + 1) * patch_size, valid_length)  # Don't exceed valid length
                mask[i, start_idx:end_idx] = 1

    # Create masked time series - only mask valid parts
    masked_time_series = time_series.clone()
    mask_indices = mask.bool() & attention_mask  # Only mask where both mask and attention_mask are True
    mask_expanded = mask_indices.unsqueeze(-1).expand(-1, -1, num_features)  # (B, max_seq_len, num_features)

    # full_reconstruction.py, TimeRCD_pretrain_multi.py
    '''
    Что делает: Заменяет замаскированные участки на случайный гауссовский шум с маленькой дисперсией.
    Зачем:
    Модель не может просто "скопировать" значение из соседних позиций (как при маскировании нулем)
    Вынуждена реально понимать структуру данных, чтобы восстановить правильное значение
    Шум — это "честная" маскировка: модель не знает, какое значение было, и должна его предсказать
    Типичное использование: В задачах, где важно, чтобы модель не "жульничала" (например, в MAE — Masked Autoencoders)
    '''
    masked_time_series[mask_expanded] = torch.randn_like(masked_time_series[mask_expanded]) * 0.1

    # training.py
    '''
    Что делает: Просто обнуляет замаскированные участки.
    Зачем:
    Проще и быстрее
    Модель может научиться, что "ноль = маска"
    Проблема: модель может использовать сам факт обнуления как признак, а не учиться восстанавливать данные
    Типичное использование: В простых baseline-моделях или когда маскировка используется не для реконструкции, а для других целей.

    Вероятно, более старая или упрощенная версия
    '''
    # masked_time_series[mask_expanded] = 0.0

    # Update mask to only include valid parts
    mask = mask * attention_mask.float()
    return masked_time_series, mask  # (B, max_seq_len, num_features), (B, max_seq_len)


def collate_fn(batch):
    """Collate function for pretraining dataset."""
    time_series_list, labels_list = zip(*batch)

    # Convert to tensors and pad sequences
    if time_series_list[0].ndim == 1:
        time_series_tensors = [ts.unsqueeze(-1) for ts in time_series_list]  # Add feature dimension
    else:
        time_series_tensors = [ts for ts in time_series_list]

    # Standardize time series (per batch) by features to stabilize training

    # TimeRCD_pretrain_multi.py, training.py
    concatenated = torch.cat(time_series_tensors, dim=0)  # (total_length, num_features)
    mean = concatenated.mean(dim=0, keepdim=True)  # (1, num_features)
    std = concatenated.std(dim=0, keepdim=True) + 1e-8  # (1, num_features)
    time_series_tensors_std = [(ts - mean) / std for ts in time_series_tensors]
    time_series_tensors = time_series_tensors_std

    # full_reconstruction.py
    # means = []
    # stds = []
    # for i in range(len(time_series_tensors)):
    #     ts = time_series_tensors[i]
    #     mean = ts.mean(dim=0, keepdim=True)
    #     std = ts.std(dim=0, keepdim=True) + 1e-4
    #     means.append(mean)
    #     stds.append(std)
    #     time_series_tensors[i] = (ts - mean) / std

    labels = [label for label in labels_list]
    # Pad time series to same length
    padded_time_series = torch.nn.utils.rnn.pad_sequence(
        time_series_tensors, batch_first=True, padding_value=0.0
    )  # (B, max_seq_len, num_features)
    padded_labels = torch.nn.utils.rnn.pad_sequence(
        labels, batch_first=True, padding_value=-100
    )  # (B, max_seq_len)

    sequence_lengths = [ts.size(0) for ts in time_series_tensors]
    B, max_seq_len, num_features = padded_time_series.shape
    attention_mask = torch.zeros(B, max_seq_len, dtype=torch.bool)  # (B, max_seq_len)
    for i, length in enumerate(sequence_lengths):
        attention_mask[i, :length] = True

    # Create random masks for reconstruction task - only mask valid sequence parts
    masked_time_series, mask = create_random_mask(padded_time_series, attention_mask)

    return {
        'time_series': padded_time_series,
        'masked_time_series': masked_time_series,
        'mask': mask,  # for reconstruction task
        'labels': padded_labels,
        'attention_mask': attention_mask,  # for padding
    }


# TimeRCD_pretrain_multi.py
# def test_collate_fn(batch):
#     """Collate function for pretraining dataset."""
#     # Unpack the batch correctly - batch is a list of (time_series, mask) tuples
#     time_series_list, mask_list = zip(*batch)

#     # Stack into batch format instead of concatenating
#     # This maintains the batch dimension: (B, seq_len, num_features)
#     batched_time_series = torch.stack(time_series_list, dim=0)
#     print(f"batched_time_series shape: {batched_time_series.shape}")
#     # Stack masks into batch format: (B, seq_len)
#     batched_mask = torch.stack(mask_list, dim=0)
#     print(f"batched_mask shape: {batched_mask.shape}")

#     return {
#         'time_series': batched_time_series,
#         'attention_mask': batched_mask,  # for padding
#     }


def get_DAE_loaders(folder, config, **kwargs):
    SCALERS_DEF = {
        'point_global': MaxAbsScaler,
        'point_contextual': MaxAbsScaler,
        'point_seasonal': MaxAbsScaler,
        'point_shapelet': MaxAbsScaler,
        'pattern_trendv2': MinMaxScaler,    
    }

    X_train = np.load(f"{folder}/train.npy")
    X_test  = np.load(f"{folder}/test.npy")
    X_valid = np.load(f"{folder}/validation.npy")
    y_train = np.load(f"{folder}/labels.npy")
    y_valid = np.load(f"{folder}/labels_validation.npy")

    # Если аномалия в одном измерении -> аномалия во всех измерениях
    y_train = y_train.max(axis=-1).flatten().astype(int)
    y_valid = y_valid.max(axis=-1).flatten().astype(int)

    name = os.path.split(folder)[-1]
    # scaler = SCALERS_DEF.get(name, MinMaxScaler)
    # scaler = scaler(**kwargs).fit(X_train)

    # X_train = scaler.transform(X_train)
    # X_valid = scaler.transform(X_valid)
    # X_test = scaler.transform(X_test)

    overlap_ratio = config.overlap_ratio
    max_seq_len = config.max_seq_len
    X_train_seq, y_train_seq = split_ts_sequences(
        X_train, y_train, max_seq_len=max_seq_len,
        overlap_ratio=overlap_ratio
    )
    X_valid_seq, y_valid_seq = split_ts_sequences(
        X_valid, y_valid, max_seq_len=max_seq_len,
        overlap_ratio=overlap_ratio
    )
    X_test_seq = split_ts_sequences(
        X_test, max_seq_len=max_seq_len,
        overlap_ratio=overlap_ratio
    )

    print(name)
    print(f"    X_train: {X_train_seq.shape}")
    print(f"    y_train: {y_train_seq.shape}")
    print(f"    X_valid: {X_valid_seq.shape}")
    print(f"    y_valid: {y_valid_seq.shape}")
    print(f"     X_test: {X_test_seq.shape}")

    X_train_seq = torch.Tensor(X_train_seq)
    X_valid_seq = torch.Tensor(X_valid_seq)
    X_test_seq  = torch.Tensor(X_test_seq)

    y_train_seq = torch.Tensor(y_train_seq).long()
    y_valid_seq = torch.Tensor(y_valid_seq).long()
    y_test_seq  = torch.Tensor(np.zeros_like(X_test_seq)).long()

    train_dataset = TensorDataset(X_train_seq, y_train_seq)
    valid_dataset = TensorDataset(X_valid_seq, y_valid_seq)
    test_dataset  = TensorDataset(X_test_seq,  y_test_seq)

    bs = config.batch_size
    loaders = {
        "train": DataLoader(
            train_dataset, batch_size=bs,
            shuffle=True,  num_workers=4,
            pin_memory=True, collate_fn=collate_fn
        ),
        "valid": DataLoader(
            valid_dataset, batch_size=bs,
            shuffle=False, num_workers=4,
            pin_memory=True, collate_fn=collate_fn
        ),
        "test":  DataLoader(
            test_dataset,  batch_size=bs,
            shuffle=False, num_workers=4,
            pin_memory=True, collate_fn=collate_fn
        ),
    }
    return loaders
