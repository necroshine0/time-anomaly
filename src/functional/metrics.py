import numpy as np
from sklearn.metrics import (
    f1_score, precision_score, recall_score, roc_auc_score
)


def calc_reconstruction_metrics(original_ts, reconstriction_ts):
    diff = original_ts - reconstriction_ts
    rmse = np.sqrt((diff ** 2).mean().mean())
    mae = np.abs(diff).mean().mean()
    return rmse, mae


def calc_detection_metrics(y_true, y_pred, y_scores):
    f1 = f1_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    rocauc = roc_auc_score(y_true, y_scores)
    return f1, recall, precision, rocauc


def point_adjustment(y_true, y_pred):
    """
    Apply point adjustment for predicted labels as described in https://arxiv.org/pdf/1802.03903 (Fig. 7).
    Args:
        y_true: numpy.array, shape = (n_samples, )
            True labels with values {0, 1}: 0 is for normal observations, 1 is for anomalies.
        y_pred: numpy.array, shape = (n_samples, )
            Predicted labels with values {0, 1}: 0 is for normal observations, 1 is for anomalies.

    Returns:
        y_pred_pa: numpy.array, shape = (n_samples, )
            Adjusted predicted labels with values {0, 1}: 0 is for normal observations, 1 is for anomalies/attacks.
    """
    y_pred_pa = np.copy(y_pred)

    # find all segmetns of 1
    seg_start = None
    seg_end = None
    segment_inds = []
    for i in range(len(y_true)):
        if y_true[i] == 1:
            if seg_start is None:
                seg_start = i
        elif y_true[i] == 0:
            if seg_start is not None:
                seg_end = i
                segment_inds.append([seg_start, seg_end])
            seg_start = None
            seg_end = None

    # adjust predictions
    for aseg in segment_inds:
        if np.sum(y_pred[aseg[0]:aseg[1]]) > 0:
            y_pred_pa[aseg[0]:aseg[1]] = 1

    return y_pred_pa


def get_best_threshold(y_true, y_scores, use_point_adjustment=True,
                         min_anomaly_rate=0.001, max_anomaly_rate=1.0, quantile_step=0.01):

    qs = np.arange(1 - max_anomaly_rate, 1 - min_anomaly_rate + quantile_step, quantile_step)
    if len(qs) == 0:
        mean = np.mean(y_scores)
        std = np.std(y_scores)
        best_thresh = (mean + 3 * std)
        return best_thresh

    best_f1 = -1
    best_thresh = None
    for aq in qs:
        thresh = np.quantile(y_scores, aq)
        y_pred = 1 * (y_scores > thresh)
        if use_point_adjustment:
            y_pred = point_adjustment(y_true, y_pred)

        f1, _, _, _ = calc_detection_metrics(y_true, y_pred, y_scores)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh

    return best_thresh


def compute_best_metrics(y_true, y_scores, original_ts, reconstructed_ts, best_thresh=None, **kwargs):
    # Reconstruction
    rmse, mae = calc_reconstruction_metrics(original_ts, reconstructed_ts)

    # Detection
    if best_thresh is None:
        best_thresh = get_best_threshold(y_true, y_scores, **kwargs)
    y_pred = 1 * (y_scores > best_thresh)

    f1, recall, precision, rocauc = calc_detection_metrics(y_true, y_pred, y_scores)

    return {
        "RMSE": rmse,
        "MAE": mae,
        "AUC-ROC": rocauc,
        "F1": f1,
        "Precision": precision,
        "Recall": recall,
        "t>": best_thresh,
    }
