"""
SAR coherence change detection for disaster damage mapping.

What this script does:
1. Creates simulated pre-event and co-event coherence images
   (or loads real GeoTIFF files if you provide them).
2. Computes the coherence change between the two dates.
3. Finds the damaged area with an automatic threshold (Otsu).
4. Compares the result with a reference map and calculates accuracy.
5. Saves a figure and a CSV summary.

Requirements:
    pip install numpy matplotlib
    (optional for real data) pip install rasterio

Run:
    python sar_damage_mapping.py
"""

import os
import csv
import numpy as np
import matplotlib.pyplot as plt

OUTPUT_DIR = "outputs"
RANDOM_SEED = 42


# ---------------------------------------------------------------
# 1. Data
# ---------------------------------------------------------------

def simulate_data(size=300, seed=RANDOM_SEED):
    """Create a fake coherence pair with a damaged circular area."""
    rng = np.random.default_rng(seed)

    yy, xx = np.mgrid[0:size, 0:size]
    cy, cx, radius = size * 0.5, size * 0.45, size * 0.18
    truth = ((yy - cy) ** 2 + (xx - cx) ** 2) < radius ** 2

    # Before the event, coherence is high almost everywhere.
    coh_pre = np.clip(rng.normal(0.80, 0.05, (size, size)), 0, 1)

    # After the event, coherence drops in the damaged area.
    coh_post = coh_pre.copy()
    coh_post[truth] = np.clip(rng.normal(0.30, 0.08, truth.sum()), 0, 1)
    coh_post = np.clip(coh_post + rng.normal(0, 0.03, coh_post.shape), 0, 1)

    return coh_pre, coh_post, truth


def load_real_data(pre_path, post_path, reference_path):
    """
    Load real coherence GeoTIFFs.
    The reference file should be a binary raster (1 = damaged, 0 = not damaged).
    """
    import rasterio

    with rasterio.open(pre_path) as src:
        coh_pre = src.read(1).astype(float)
    with rasterio.open(post_path) as src:
        coh_post = src.read(1).astype(float)
    with rasterio.open(reference_path) as src:
        truth = src.read(1) == 1

    return coh_pre, coh_post, truth


# ---------------------------------------------------------------
# 2. Change detection
# ---------------------------------------------------------------

def coherence_change(coh_pre, coh_post):
    """A large drop in coherence means a possible change (damage)."""
    return coh_pre - coh_post


def otsu_threshold(values, bins=256):
    """Find the threshold that best separates two groups of values."""
    hist, edges = np.histogram(values, bins=bins)
    hist = hist.astype(float)
    centers = (edges[:-1] + edges[1:]) / 2

    w_low = np.cumsum(hist)
    w_high = np.cumsum(hist[::-1])[::-1]
    m_low = np.cumsum(hist * centers)
    m_high = np.cumsum((hist * centers)[::-1])[::-1]

    mean_low = m_low / np.maximum(w_low, 1e-9)
    mean_high = m_high / np.maximum(w_high, 1e-9)

    between = w_low[:-1] * w_high[1:] * (mean_low[:-1] - mean_high[1:]) ** 2
    return centers[np.argmax(between)]


def detect_damage(change, threshold=None):
    """Return a True/False damage map."""
    if threshold is None:
        threshold = otsu_threshold(change.ravel())
    return change > threshold, threshold


# ---------------------------------------------------------------
# 3. Accuracy
# ---------------------------------------------------------------

def accuracy_scores(pred, truth):
    tp = np.sum(pred & truth)
    fp = np.sum(pred & ~truth)
    fn = np.sum(~pred & truth)
    tn = np.sum(~pred & ~truth)
    total = tp + fp + fn + tn

    overall = (tp + tn) / total
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) else 0.0)
    iou = tp / (tp + fp + fn) if (tp + fp + fn) else 0.0

    return {
        "overall_accuracy": overall,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "iou": iou,
    }


# ---------------------------------------------------------------
# 4. Output
# ---------------------------------------------------------------

def save_figure(coh_pre, coh_post, change, pred, truth, path):
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))

    axes[0].imshow(coh_pre, cmap="gray", vmin=0, vmax=1)
    axes[0].set_title("Pre-event coherence")

    axes[1].imshow(coh_post, cmap="gray", vmin=0, vmax=1)
    axes[1].set_title("Co-event coherence")

    im = axes[2].imshow(change, cmap="RdBu_r")
    axes[2].set_title("Coherence change")
    fig.colorbar(im, ax=axes[2], fraction=0.046)

    overlay = np.zeros(truth.shape + (3,))
    overlay[truth & pred] = [0, 1, 0]        # correct damage: green
    overlay[pred & ~truth] = [1, 0, 0]       # false alarm: red
    overlay[~pred & truth] = [0, 0, 1]       # missed damage: blue
    axes[3].imshow(overlay)
    axes[3].set_title("Green = correct, Red = false alarm, Blue = missed")

    for ax in axes:
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close(fig)


def save_summary(scores, threshold, path):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        writer.writerow(["otsu_threshold", round(float(threshold), 4)])
        for key, value in scores.items():
            writer.writerow([key, round(float(value), 4)])


# ---------------------------------------------------------------
# Main
# ---------------------------------------------------------------

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # To use real data, replace simulate_data() with load_real_data(...)
    coh_pre, coh_post, truth = simulate_data()

    change = coherence_change(coh_pre, coh_post)
    pred, threshold = detect_damage(change)
    scores = accuracy_scores(pred, truth)

    figure_path = os.path.join(OUTPUT_DIR, "damage_map.png")
    csv_path = os.path.join(OUTPUT_DIR, "accuracy_summary.csv")

    save_figure(coh_pre, coh_post, change, pred, truth, figure_path)
    save_summary(scores, threshold, csv_path)

    print("Damage mapping complete")
    print(f"Otsu threshold: {threshold:.3f}")
    for key, value in scores.items():
        print(f"{key:>18}: {value:.3f}")
    print(f"\nFigure saved to: {figure_path}")
    print(f"Summary saved to: {csv_path}")


if __name__ == "__main__":
    main()