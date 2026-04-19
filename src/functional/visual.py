import matplotlib.pyplot as plt

from .training import predict_batch


def plot_training_results(results: dict, title_prefix=None):
    epochs = results["epochs"]
    train_losses = results["train_losses"]
    valid_losses = results["valid_losses"]

    fig, axes = plt.subplots(ncols=3, figsize=(15, 4))
    axes[0].plot(epochs, train_losses["total"], label="train")
    axes[0].plot(epochs, valid_losses["total"], label="valid")
    if title_prefix:
        axes[0].set_title(f"{title_prefix}: total loss")
    else:
        axes[0].set_title("total loss")
    axes[0].legend()
    axes[0].grid()

    axes[1].plot(epochs, train_losses["reconstruction"], label="train")
    axes[1].plot(epochs, valid_losses["reconstruction"], label="valid")
    if title_prefix:
        axes[1].set_title(f"{title_prefix}: reconstruction loss")
    else:
        axes[1].set_title("reconstruction loss")
    axes[1].legend()
    axes[1].grid()

    axes[2].plot(epochs, train_losses["anomaly"], label="train")
    axes[2].plot(epochs, valid_losses["anomaly"], label="valid")
    if title_prefix:
        axes[2].set_title(f"{title_prefix}: anomaly loss")
    else:
        axes[2].set_title("anomaly loss")
    axes[2].legend()
    axes[2].grid()

    plt.tight_layout()
    plt.show()
    plt.close(fig)


def visualize_batch_sample(detector, batch, sample_idx=0):
    device = next(detector.parameters()).device
    pred = predict_batch(detector, batch, device)

    atn_mask = pred['attention_mask'][sample_idx].numpy()
    anomaly_scores = pred['anomaly_scores'][sample_idx][atn_mask].numpy()
    true_labels = pred['true_labels'][sample_idx][atn_mask].numpy()

    for dim in range(5):
        original = pred["original"][sample_idx, atn_mask, dim].numpy()
        reconstructed = pred["reconstructed"][sample_idx, atn_mask, dim].numpy()

        plt.figure(figsize=(14, 3))
        plt.title(f"sample_idx={sample_idx}: dim={dim}")
        plt.plot(original, label="original")
        plt.plot(reconstructed, label="reconstructed")
        plt.legend()
        plt.show()

    plt.figure(figsize=(14, 3))
    plt.title(f"sample_idx={sample_idx}: labels vs anomaly scores")
    plt.plot(true_labels, label="labels")
    plt.plot(anomaly_scores, label="anomaly scores")
    plt.legend()
    plt.show()
