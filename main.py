#!/usr/bin/env python3
"""
main.py
=======================================================
LogBERT: Log Anomaly Detection using Transformers
=======================================================
"""

import sys
import time
import json
import argparse
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from loguru import logger
from sklearn.model_selection import train_test_split

logger.remove()
logger.add(sys.stdout,
           format="<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | {message}",
           colorize=True)

# ── Create required directories BEFORE anything else ──────────────────────────
Path("results").mkdir(exist_ok=True)
Path("models").mkdir(exist_ok=True)
Path("data").mkdir(exist_ok=True)

logger.add("results/run.log", rotation="1 MB", retention=3,
           format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}")

from src.device  import DEVICE, device_info
from src.data    import load_dataset
from src.model   import build_model, load_model
from src.dataset import LogSessionDataset
from src.trainer import train, evaluate, THRESH
from src.report  import save_metrics, generate_all, _find_best_threshold


def _banner():
    info = device_info()
    gpu  = info.get("gpu_name", "CPU")
    print()
    print("  ╔══════════════════════════════════════════════════════════╗")
    print("  ║                                                          ║")
    print("  ║   LogBERT — Log Anomaly Detection                       ║")
    print("  ║   Transformer-Based Big Data Engineering Project        ║")
    print("  ║                                                          ║")
    print(f"  ║   Device  : {gpu:<44} ║")
    print(f"  ║   PyTorch : {info['torch']:<44} ║")
    print("  ║                                                          ║")
    print("  ╚══════════════════════════════════════════════════════════╝")
    print()


def _phase_header(n: int, title: str):
    print()
    print(f"  ┌──────────────────────────────────────────────────────┐")
    print(f"  │  PHASE {n}  {title:<45}│")
    print(f"  └──────────────────────────────────────────────────────┘")
    print()


def _phase_done(n: int, elapsed: float):
    logger.success(f"Phase {n} complete  [{elapsed:.1f}s]")


# ══════════════════════════════════════════════════════════════════════════════
# Phase 1 — Data
# ══════════════════════════════════════════════════════════════════════════════

def phase1() -> tuple[pd.DataFrame, dict]:
    _phase_header(1, "Download & Parse Real HDFS Dataset")
    t0 = time.time()

    sessions, vocab = load_dataset()

    logger.info(f"  Sessions  : {len(sessions):,}")
    logger.info(f"  Normal    : {(sessions['Label']==0).sum():,}")
    logger.info(f"  Anomaly   : {(sessions['Label']==1).sum():,}")
    logger.info(f"  Vocabulary: {len(vocab)} unique event tokens")

    _phase_done(1, time.time() - t0)
    return sessions, vocab


# ══════════════════════════════════════════════════════════════════════════════
# Phase 2 — Train
# ══════════════════════════════════════════════════════════════════════════════

def phase2(sessions: pd.DataFrame, vocab: dict) -> dict:
    _phase_header(2, "Train LogBERT (Masked Log-Key Prediction)")
    t0 = time.time()

    # Ensure models dir exists
    Path("models").mkdir(exist_ok=True)

    # Train on NORMAL sessions only (unsupervised anomaly detection)
    normal_df = sessions[sessions["Label"] == 0]
    train_df, _ = train_test_split(normal_df, test_size=0.1, random_state=42)

    # Test on all sessions (normal + anomaly)
    test_df = sessions

    logger.info(f"  Train sessions (normal only) : {len(train_df):,}")
    logger.info(f"  Test  sessions (all)          : {len(test_df):,}")

    train_ds = LogSessionDataset(train_df, vocab)
    test_ds  = LogSessionDataset(test_df,  vocab)
    model    = build_model(len(vocab))

    best_metrics = train(model, train_ds, test_ds)

    _phase_done(2, time.time() - t0)
    return best_metrics


# ══════════════════════════════════════════════════════════════════════════════
# Phase 3 — Evaluate
# ══════════════════════════════════════════════════════════════════════════════

def phase3(sessions: pd.DataFrame, vocab: dict, threshold: float = THRESH) -> dict:
    _phase_header(3, "Evaluate: Precision / Recall / F1 / ROC-AUC")
    t0 = time.time()

    model_path = Path("models/logbert_best.pt")
    if not model_path.exists():
        logger.error("No trained model found at models/logbert_best.pt")
        logger.error("Make sure Phase 2 completed successfully.")
        raise SystemExit(1)

    model, ckpt = load_model(model_path)
    logger.success(f"  Loaded model (epoch {ckpt.get('epoch','?')}, "
                   f"train F1={ckpt.get('f1', 0):.4f})")

    from src.trainer import _score_loader
    from src.dataset import LogSessionDataset
    from torch.utils.data import DataLoader

    nw      = 0 if str(DEVICE) == "mps" else 2
    dataset = LogSessionDataset(sessions, vocab)
    loader  = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=nw)
    scores, labels = _score_loader(model, loader)

    best_t = _find_best_threshold(scores, labels)
    preds  = (scores >= best_t).astype(int)
    logger.info(f"  Optimal threshold: {best_t}")

    metrics = save_metrics(scores, labels, best_t,
                           extra={"trained_epochs": ckpt.get("epoch", "?")})

    _phase_done(3, time.time() - t0)
    return {"scores": scores, "labels": labels, "preds": preds,
            "threshold": best_t, "metrics": metrics}


# ══════════════════════════════════════════════════════════════════════════════
# Phase 4 — Report
# ══════════════════════════════════════════════════════════════════════════════

def phase4(eval_result: dict, sessions: pd.DataFrame):
    _phase_header(4, "Save Results & Generate Plots")
    t0 = time.time()

    scores    = eval_result["scores"]
    labels    = eval_result["labels"]
    preds     = eval_result["preds"]
    threshold = eval_result["threshold"]
    metrics   = eval_result["metrics"]

    generate_all(scores, labels, preds, threshold)

    out = sessions[["BlockId", "Label"]].copy()
    out["AnomalyScore"] = scores.round(4)
    out["Predicted"]    = preds
    out["Correct"]      = (out["Label"] == out["Predicted"]).astype(int)
    out.to_csv("results/session_scores.csv", index=False)
    logger.success("  Saved → results/session_scores.csv")

    _phase_done(4, time.time() - t0)

    m = metrics
    print()
    print("  ╔══════════════════════════════════════════════════════════╗")
    print("  ║                  FINAL RESULTS                          ║")
    print("  ╠══════════════════════════════════════════════════════════╣")
    print(f"  ║  Precision      : {m['precision']:.4f}                               ║")
    print(f"  ║  Recall         : {m['recall']:.4f}                               ║")
    print(f"  ║  F1 Score       : {m['f1_score']:.4f}                               ║")
    print(f"  ║  ROC-AUC        : {m['roc_auc']:.4f}                               ║")
    print(f"  ║  PR-AUC         : {m['pr_auc']:.4f}                               ║")
    print(f"  ║  Accuracy       : {m['accuracy']:.4f}                               ║")
    print("  ╠══════════════════════════════════════════════════════════╣")
    print(f"  ║  True  Positives: {m['true_positives']:<5}  (anomalies correctly caught)  ║")
    print(f"  ║  False Positives: {m['false_positives']:<5}  (false alarms)                ║")
    print(f"  ║  False Negatives: {m['false_negatives']:<5}  (missed anomalies)            ║")
    print("  ╠══════════════════════════════════════════════════════════╣")
    print("  ║  Output files:                                          ║")
    print("  ║    results/metrics.json                                 ║")
    print("  ║    results/session_scores.csv                           ║")
    print("  ║    results/score_distribution.png                       ║")
    print("  ║    results/roc_curve.png                                ║")
    print("  ║    results/confusion_matrix.png                         ║")
    print("  ║    results/training_history.png                         ║")
    print("  ╚══════════════════════════════════════════════════════════╝")
    print()


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="LogBERT — Log Anomaly Detection Pipeline"
    )
    parser.add_argument("--skip-train", action="store_true",
                        help="Skip training; load saved model")
    parser.add_argument("--phase", type=int, default=None,
                        choices=[1, 2, 3, 4],
                        help="Run a single phase (1-4)")
    args = parser.parse_args()

    _banner()
    t_start = time.time()

    if args.phase == 1:
        phase1()
    elif args.phase == 2:
        sessions, vocab = phase1()
        phase2(sessions, vocab)
    elif args.phase in (3, 4):
        sessions, vocab = phase1()
        res = phase3(sessions, vocab)
        phase4(res, sessions)
    else:
        sessions, vocab = phase1()
        if args.skip_train:
            logger.info("--skip-train: using existing model weights")
        else:
            phase2(sessions, vocab)
        res = phase3(sessions, vocab)
        phase4(res, sessions)

    total = time.time() - t_start
    logger.success(f"Total runtime: {total/60:.1f} min")


if __name__ == "__main__":
    main()