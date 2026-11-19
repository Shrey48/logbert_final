"""
src/model.py
LogBERT: BERT adapted for log anomaly detection via
Masked Log-Key Prediction (MLKP).

Reference: Guo et al., "LogBERT: Log Anomaly Detection via BERT", 2021.
"""

import torch
import torch.nn as nn
from transformers import BertConfig, BertForMaskedLM
from pathlib import Path
from loguru import logger

from src.device import DEVICE


def _bert_config(vocab_size: int) -> BertConfig:
    """
    Compact BERT config tuned for log sequences.
    Runs efficiently on Apple MPS and consumer GPUs.
    Scale up hidden_size/num_hidden_layers for larger datasets.
    """
    return BertConfig(
        vocab_size                   = vocab_size,
        hidden_size                  = 256,
        num_hidden_layers            = 4,
        num_attention_heads          = 4,
        intermediate_size            = 1024,
        max_position_embeddings      = 512,
        hidden_dropout_prob          = 0.1,
        attention_probs_dropout_prob = 0.1,
        type_vocab_size              = 1,
        pad_token_id                 = 0,
    )


class LogBERT(nn.Module):
    """
    Transformer-based log anomaly detector.

    Training objective — Masked Log-Key Prediction (MLKP):
      15% of event tokens are masked at random. The model learns
      to reconstruct them using the surrounding context.
      Normal logs have predictable event sequences; anomalous logs
      produce unexpected events the model cannot predict.

    Inference — anomaly score ∈ [0, 1]:
      For each masked position, we check whether the true event
      appears in the model's top-k predictions.
        score = (# masks NOT in top-k) / (total masks)
      score → 0  : normal session
      score → 1  : highly anomalous session
    """

    def __init__(self, vocab_size: int):
        super().__init__()
        self.bert       = BertForMaskedLM(_bert_config(vocab_size))
        self.vocab_size = vocab_size

    def forward(self, input_ids, attention_mask, labels=None):
        return self.bert(input_ids=input_ids,
                         attention_mask=attention_mask,
                         labels=labels)

    @torch.no_grad()
    def anomaly_score(self, input_ids, attention_mask,
                      mask_positions, true_ids, top_k: int = 9) -> torch.Tensor:
        logits = self.bert(input_ids=input_ids,
                           attention_mask=attention_mask).logits   # (B, L, V)
        B      = input_ids.shape[0]
        scores = torch.zeros(B, device=input_ids.device)
        for b in range(B):
            pos = mask_positions[b].nonzero(as_tuple=True)[0]
            if len(pos) == 0:
                continue
            hits = sum(
                true_ids[b, p].item() in logits[b, p].topk(top_k).indices.tolist()
                for p in pos
            )
            scores[b] = 1.0 - hits / len(pos)
        return scores


def build_model(vocab_size: int) -> LogBERT:
    model    = LogBERT(vocab_size).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"  LogBERT | vocab={vocab_size} | params={n_params:,} | device={DEVICE}")
    return model


def save_model(model: LogBERT, path: Path, meta: dict):
    torch.save({"model_state": model.state_dict(),
                "vocab_size":  model.vocab_size, **meta}, path)


def load_model(path: Path) -> tuple[LogBERT, dict]:
    ckpt  = torch.load(path, map_location=DEVICE)
    model = LogBERT(ckpt["vocab_size"]).to(DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, ckpt
