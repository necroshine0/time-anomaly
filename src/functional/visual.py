import matplotlib.pyplot as plt

from .training import predict_batch


def plot_training_results(results: dict, title_prefix=None):
    epochs = results["epochs"]
    train_losses = results["train_losses"]
    valid_losses = results["valid_losses"]

    fig = plt.figure(figsize=(12, 4))
    plt.plot(epochs, train_losses["reconstruction"], label="train")
    plt.plot(epochs, valid_losses["reconstruction"], label="valid")
    if title_prefix:
        plt.title(f"{title_prefix}: reconstruction loss")
    else:
        plt.title("reconstruction loss")
    plt.legend()
    plt.grid()

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
