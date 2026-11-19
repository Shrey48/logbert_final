"""
src/report.py
Saves evaluation metrics to JSON and generates publication-quality plots:
  - Anomaly score distribution
  - ROC curve with AUC
  - Precision-Recall curve
  - Confusion matrix heatmap
  - Training loss curve
"""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve,
    confusion_matrix, f1_score,
)
from loguru import logger

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

# Consistent style
plt.rcParams.update({
    "font.family":     "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid":       True,
    "grid.alpha":      0.3,
    "figure.dpi":      150,
})
C_NORMAL  = "#378ADD"
C_ANOMALY = "#D85A30"
C_MAIN    = "#534AB7"
C_GREEN   = "#1D9E75"


def _find_best_threshold(scores, labels):
    best_t, best_f1 = 0.5, 0.0
    for t in np.linspace(0.0, 1.0, 200):
        f1 = f1_score(labels, (scores >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return round(best_t, 3)


def save_metrics(scores, labels, threshold, extra: dict = None) -> dict:
    preds   = (scores >= threshold).astype(int)
    fpr, tpr, _ = roc_curve(labels, scores)
    roc_auc = auc(fpr, tpr)
    prec, rec, _ = precision_recall_curve(labels, scores)
    pr_auc  = auc(rec, prec)
    cm      = confusion_matrix(labels, preds)
    tp, fn, fp, tn = cm[1,1], cm[1,0], cm[0,1], cm[0,0]

    metrics = {
        "threshold":             threshold,
        "precision":             round(tp / max(tp+fp, 1), 4),
        "recall":                round(tp / max(tp+fn, 1), 4),
        "f1_score":              round(2*tp / max(2*tp+fp+fn, 1), 4),
        "accuracy":              round((tp+tn) / max(len(labels), 1), 4),
        "roc_auc":               round(roc_auc, 4),
        "pr_auc":                round(pr_auc, 4),
        "true_positives":        int(tp),
        "true_negatives":        int(tn),
        "false_positives":       int(fp),
        "false_negatives":       int(fn),
        "total_sessions":        int(len(labels)),
        "anomalies_true":        int(labels.sum()),
        "anomalies_predicted":   int(preds.sum()),
    }
    if extra:
        metrics.update(extra)
    (RESULTS_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))
    logger.success(f"  Metrics saved → results/metrics.json")
    return metrics


def plot_score_distribution(scores, labels):
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.hist(scores[labels==0], bins=50, alpha=0.7, color=C_NORMAL,  label="Normal",  density=True)
    ax.hist(scores[labels==1], bins=50, alpha=0.7, color=C_ANOMALY, label="Anomaly", density=True)
    ax.set_xlabel("Anomaly Score",  fontsize=12)
    ax.set_ylabel("Density",        fontsize=12)
    ax.set_title("LogBERT — Anomaly Score Distribution\n"
                 "Normal sessions score low; anomalous sessions score high",
                 fontsize=12, pad=10)
    ax.legend(fontsize=11)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "score_distribution.png")
    plt.close(fig)
    logger.success("  Plot saved → results/score_distribution.png")


def plot_roc(scores, labels) -> float:
    fpr, tpr, _ = roc_curve(labels, scores)
    roc_auc     = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color=C_MAIN, lw=2.5, label=f"LogBERT  AUC = {roc_auc:.4f}")
    ax.plot([0,1],[0,1], "k--", lw=1, label="Random classifier")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate",  fontsize=12)
    ax.set_title("ROC Curve",            fontsize=13)
    ax.legend(fontsize=11)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "roc_curve.png")
    plt.close(fig)
    logger.success("  Plot saved → results/roc_curve.png")
    return roc_auc


def plot_pr_curve(scores, labels) -> float:
    prec, rec, _ = precision_recall_curve(labels, scores)
    pr_auc       = auc(rec, prec)
    baseline     = labels.mean()
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(rec, prec, color=C_GREEN, lw=2.5, label=f"LogBERT  AUC = {pr_auc:.4f}")
    ax.axhline(baseline, color="gray", linestyle="--", lw=1, label=f"Random ({baseline:.2f})")
    ax.set_xlabel("Recall",    fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title("Precision-Recall Curve", fontsize=13)
    ax.legend(fontsize=11)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "pr_curve.png")
    plt.close(fig)
    logger.success("  Plot saved → results/pr_curve.png")
    return pr_auc


def plot_confusion_matrix(labels, preds):
    cm  = confusion_matrix(labels, preds)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", linewidths=0.5,
                xticklabels=["Normal", "Anomaly"],
                yticklabels=["Normal", "Anomaly"],
                annot_kws={"size": 14}, ax=ax)
    ax.set_ylabel("True Label",      fontsize=12)
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_title("Confusion Matrix", fontsize=13)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "confusion_matrix.png")
    plt.close(fig)
    logger.success("  Plot saved → results/confusion_matrix.png")


def plot_training_history():
    hist_path = RESULTS_DIR / "history.json"
    if not hist_path.exists():
        return
    import json
    history = json.loads(hist_path.read_text())
    epochs  = [h["epoch"] for h in history]
    losses  = [h["loss"]  for h in history]
    f1s     = [h["f1"]    for h in history]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    ax1.plot(epochs, losses, color=C_MAIN, lw=2, marker="o", markersize=5)
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss"); ax1.set_title("Training Loss")
    ax2.plot(epochs, f1s,    color=C_GREEN, lw=2, marker="o", markersize=5)
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("F1 Score"); ax2.set_title("Test F1 Score")
    fig.suptitle("LogBERT Training Progress", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "training_history.png")
    plt.close(fig)
    logger.success("  Plot saved → results/training_history.png")


def generate_all(scores, labels, preds, threshold):
    plot_score_distribution(scores, labels)
    plot_roc(scores, labels)
    plot_pr_curve(scores, labels)
    plot_confusion_matrix(labels, preds)
    plot_training_history()
