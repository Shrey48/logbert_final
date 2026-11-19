"""
src/dataset.py
PyTorch Dataset for LogBERT.
Implements BERT-style Masked Log-Key Prediction (MLKP).
"""

import ast
import random
import torch
import pandas as pd
from torch.utils.data import Dataset

MAX_SEQ_LEN = 50
MASK_RATIO  = 0.15


class LogSessionDataset(Dataset):
    """
    One item = one log session (a sequence of event IDs for a single HDFS block).
    Applies dynamic masking: 15% of event tokens are randomly replaced
    with [MASK] at each __getitem__ call (different mask each epoch).
    """

    def __init__(self, sessions: pd.DataFrame, vocab: dict,
                 max_len: int = MAX_SEQ_LEN, mask_ratio: float = MASK_RATIO):
        self.sessions   = sessions.reset_index(drop=True)
        self.vocab      = vocab
        self.max_len    = max_len
        self.mask_ratio = mask_ratio

        self.PAD  = vocab["[PAD]"]
        self.UNK  = vocab["[UNK]"]
        self.CLS  = vocab["[CLS]"]
        self.SEP  = vocab["[SEP]"]
        self.MASK = vocab["[MASK]"]
        self._special = {self.PAD, self.UNK, self.CLS, self.SEP, self.MASK}

    def __len__(self): return len(self.sessions)

    def _parse(self, raw) -> list:
        if isinstance(raw, list): return raw
        try:    return ast.literal_eval(raw)
        except: return str(raw).split()

    def _encode(self, seq: list) -> list:
        ids = [self.vocab.get(e, self.UNK) for e in seq]
        ids = ids[:self.max_len - 2]
        return [self.CLS] + ids + [self.SEP]

    def _pad(self, ids: list):
        n   = self.max_len - len(ids)
        attn = [1]*len(ids) + [0]*n
        ids  = ids          + [self.PAD]*n
        return ids, attn

    def _mask(self, ids: list, attn: list):
        masked = ids.copy()
        labels = [-100] * len(ids)
        is_mask = [False] * len(ids)
        cands = [i for i, (t, a) in enumerate(zip(ids, attn))
                 if a == 1 and t not in self._special]
        n = max(1, int(len(cands) * self.mask_ratio))
        for i in random.sample(cands, min(n, len(cands))):
            labels[i]  = ids[i]
            is_mask[i] = True
            r = random.random()
            if   r < 0.80: masked[i] = self.MASK
            elif r < 0.90: masked[i] = random.randint(5, len(self.vocab)-1)
        return masked, labels, is_mask

    def __getitem__(self, idx):
        row     = self.sessions.iloc[idx]
        seq     = self._parse(row["Sequence"])
        ids     = self._encode(seq)
        true_ids = ids.copy()
        ids, attn = self._pad(ids)
        true_ids, _ = self._pad(true_ids)
        masked, labels, is_mask = self._mask(ids, attn)
        return {
            "input_ids":      torch.tensor(masked,    dtype=torch.long),
            "attention_mask": torch.tensor(attn,      dtype=torch.long),
            "labels":         torch.tensor(labels,    dtype=torch.long),
            "true_ids":       torch.tensor(true_ids,  dtype=torch.long),
            "mask_positions": torch.tensor(is_mask,   dtype=torch.bool),
            "gt_label":       torch.tensor(int(row.get("Label", 0)), dtype=torch.long),
        }
