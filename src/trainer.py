"""
src/trainer.py
Training and evaluation engine for LogBERT.
"""

import json
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from pathlib import Path
from loguru import logger
from sklearn.metrics import (
    classification_report, f1_score,
    roc_curve, auc, precision_recall_curve,
)

from src.device import DEVICE
from src.model  import LogBERT, save_model

# ── Config ────────────────────────────────────────────────────────────────────
BATCH_SIZE     = 64
EPOCHS         = 20
LR             = 3e-4
WARMUP_RATIO   = 0.1
TOP_K          = 50
THRESH         = 0.3
SEED           = 42
MODEL_PATH     = Path("models/logbert_best.pt")

torch.manual_seed(SEED)
np.random.seed(SEED)


# ── Training loop ─────────────────────────────────────────────────────────────

def _train_epoch(model, loader, optimizer, scheduler, epoch: int) -> float:
    model.train()
    total = 0.0
    for step, batch in enumerate(loader):
        ids  = batch["input_ids"].to(DEVICE)
        attn = batch["attention_mask"].to(DEVICE)
        lbl  = batch["labels"].to(DEVICE)
        optimizer.zero_grad()
        loss = model(ids, attn, labels=lbl).loss
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        total += loss.item()
        if step % 20 == 0:
            logger.info(f"    step {step:3d}/{len(loader)} | "
                        f"loss {loss.item():.4f} | "
                        f"lr {scheduler.get_last_lr()[0]:.2e}")
    return total / len(loader)


# ── Evaluation ────────────────────────────────────────────────────────────────

@torch.no_grad()
def _score_loader(model, loader) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    scores, labels = [], []
    for batch in loader:
        sc = model.anomaly_score(
            batch["input_ids"].to(DEVICE),
            batch["attention_mask"].to(DEVICE),
            batch["mask_positions"].to(DEVICE),
            batch["true_ids"].to(DEVICE),
            top_k=TOP_K,
        )
        scores.extend(sc.cpu().numpy())
        labels.extend(batch["gt_label"].numpy())
    return np.array(scores), np.array(labels)


def evaluate(model, loader, threshold=THRESH) -> dict:
    scores, labels = _score_loader(model, loader)
    preds  = (scores >= threshold).astype(int)
    f1     = f1_score(labels, preds, zero_division=0)
    report = classification_report(labels, preds,
                                   target_names=["Normal", "Anomaly"],
                                   zero_division=0)
    logger.info(f"\n{report}")
    return {"f1": f1, "scores": scores, "labels": labels, "preds": preds}


# ── Full training pipeline ────────────────────────────────────────────────────

def train(model: LogBERT, train_ds, test_ds) -> dict:
    # Guarantee models/ directory exists
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    nw = 0 if str(DEVICE) == "mps" else 2
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                              shuffle=True,  num_workers=nw, pin_memory=False)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE * 2,
                              shuffle=False, num_workers=nw, pin_memory=False)

    optimizer    = AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    total_steps  = len(train_loader) * EPOCHS
    warmup_steps = int(total_steps * WARMUP_RATIO)
    scheduler    = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    best_f1   = -1.0   # start at -1 so epoch-1 always saves
    best_meta = {}
    history   = []

    for epoch in range(1, EPOCHS + 1):
        t0       = time.time()
        avg_loss = _train_epoch(model, train_loader, optimizer, scheduler, epoch)
        elapsed  = time.time() - t0

        logger.info(f"  Epoch {epoch}/{EPOCHS} | avg_loss={avg_loss:.4f} | {elapsed:.0f}s")

        if epoch % 2 == 0 or epoch == EPOCHS:
            metrics = evaluate(model, test_loader)
            f1      = metrics["f1"]
            history.append({"epoch": epoch, "loss": round(avg_loss, 4), "f1": round(f1, 4)})

            # Always save on first eval; then only on improvement
            if f1 > best_f1:
                best_f1   = f1
                best_meta = {**metrics, "epoch": epoch, "threshold": THRESH, "top_k": TOP_K}
                save_model(model, MODEL_PATH,
                           {"epoch": epoch, "f1": f1,
                            "threshold": THRESH, "top_k": TOP_K})
                logger.success(f"  ✓ New best — F1={f1:.4f} → saved to {MODEL_PATH}")

        # Safety net: always ensure a checkpoint exists after epoch 1
        if epoch == 1 and not MODEL_PATH.exists():
            save_model(model, MODEL_PATH,
                       {"epoch": 1, "f1": 0.0,
                        "threshold": THRESH, "top_k": TOP_K})
            logger.info(f"  Checkpoint saved (epoch 1 safety) → {MODEL_PATH}")

    Path("results").mkdir(exist_ok=True)
    (Path("results") / "history.json").write_text(json.dumps(history, indent=2))
    return best_meta