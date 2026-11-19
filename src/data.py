"""
src/data.py
Loads the HDFS_v1 dataset (logpai/loghub Zenodo release).

Supports three data layouts (auto-detected):
  A) Event_traces.csv + anomaly_label.csv   ← HDFS_v1.zip preprocessed/
  B) HDFS.log_structured.csv + anomaly_label.csv
  C) Raw HDFS.log (parsed via Drain3)
  D) 2k sample downloaded from GitHub (dev/demo only)

Reference: Xu et al., SOSP 2009.
"""

import re
import json
import sys
import requests
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict
from loguru import logger
from tqdm import tqdm

DATA_DIR     = Path("data")
DATA_DIR.mkdir(exist_ok=True)

SESSIONS_CSV  = DATA_DIR / "sessions.csv"
VOCAB_JSON    = DATA_DIR / "vocabulary.json"
RAW_LOG       = DATA_DIR / "HDFS.log"
STRUCT_CSV    = DATA_DIR / "HDFS_structured.csv"
LABEL_CSV     = DATA_DIR / "anomaly_label.csv"
TRACES_CSV    = DATA_DIR / "Event_traces.csv"   # HDFS_v1 preprocessed

URLS = {
    "structured_2k": [
        "https://raw.githubusercontent.com/logpai/loghub/master/HDFS/HDFS_2k.log_structured.csv",
        "https://raw.githubusercontent.com/logpai/logparser/main/data/loghub-2k/HDFS/HDFS_2k.log_structured.csv",
    ],
    "labels": [
        "https://raw.githubusercontent.com/logpai/loghub/master/HDFS/HDFS_v1/anomaly_label.csv",
        "https://raw.githubusercontent.com/logpai/loghub/master/HDFS/anomaly_label.csv",
    ],
}


# ══════════════════════════════════════════════════════════════════════════════
# Layout A — Event_traces.csv  (best: already session-grouped)
# ══════════════════════════════════════════════════════════════════════════════

def _load_from_event_traces(traces_path: Path, label_path: Path) -> pd.DataFrame:
    """
    Event_traces.csv from HDFS_v1 preprocessed/:
      Each row = one block session, columns vary by version.
      Common formats:
        BlockId, E1 E2 E3 ...          (space-separated in one column)
        BlockId, E1, E2, E3, ...       (one event per column)
    """
    logger.info(f"  Loading Event_traces.csv "
                f"({traces_path.stat().st_size // 1024 // 1024} MB)...")
    df = pd.read_csv(traces_path)
    logger.info(f"  Raw columns: {list(df.columns)[:6]}  ({len(df.columns)} total)")
    logger.info(f"  Rows       : {len(df):,}")

    # ── Detect and normalise format ────────────────────────────────────────────
    col_lower = {c: c.lower().strip() for c in df.columns}

    block_col = next((c for c, lc in col_lower.items()
                      if "block" in lc), None)
    seq_col   = next((c for c, lc in col_lower.items()
                      if any(k in lc for k in
                             ("sequence","eventsequence","events","trace"))),
                     None)

    if seq_col:
        # Standard format: BlockId + one sequence column
        sessions_df = pd.DataFrame({
            "BlockId":  df[block_col].astype(str) if block_col else
                        [f"BLK_{i}" for i in range(len(df))],
            "Sequence": df[seq_col].astype(str),
        })
    elif block_col:
        # Wide format: BlockId + one event per column → join them
        event_cols = [c for c in df.columns if c != block_col]
        sessions_df = pd.DataFrame({
            "BlockId":  df[block_col].astype(str),
            "Sequence": df[event_cols].astype(str).apply(
                lambda r: " ".join(v for v in r if v not in ("nan","","-")), axis=1
            ),
        })
    else:
        # Single column of space-separated events
        sessions_df = pd.DataFrame({
            "BlockId":  [f"BLK_{i}" for i in range(len(df))],
            "Sequence": df.iloc[:, 0].astype(str),
        })

    # ── Merge anomaly labels ───────────────────────────────────────────────────
    label_map = {}
    if label_path.exists():
        ldf = pd.read_csv(label_path)
        ldf.columns = [c.strip() for c in ldf.columns]
        bc = next((c for c in ldf.columns if "block" in c.lower()), ldf.columns[0])
        lc = next((c for c in ldf.columns if "label" in c.lower()), ldf.columns[1])
        label_map = {str(r[bc]).strip(): str(r[lc]).strip().lower()
                     for _, r in ldf.iterrows()}
        logger.success(f"  Labels loaded: {len(label_map):,} entries "
                       f"({sum(1 for v in label_map.values() if v=='anomaly'):,} anomalies)")

    sessions_df["Label"]  = sessions_df["BlockId"].apply(
        lambda b: 1 if label_map.get(str(b).strip(), "normal") == "anomaly" else 0
    )
    sessions_df["Length"] = sessions_df["Sequence"].apply(
        lambda s: len(str(s).split())
    )
    return sessions_df[["BlockId", "Sequence", "Label", "Length"]]


# ══════════════════════════════════════════════════════════════════════════════
# Layout B — HDFS.log_structured.csv
# ══════════════════════════════════════════════════════════════════════════════

def _load_structured() -> pd.DataFrame:
    df = pd.read_csv(STRUCT_CSV)
    rename = {}
    for col in df.columns:
        lc = col.lower().replace(" ", "").replace("_", "")
        if lc in ("eventid", "event"):  rename[col] = "EventId"
        elif lc == "blockid":           rename[col] = "BlockId"
        elif lc == "content":           rename[col] = "Content"
    df.rename(columns=rename, inplace=True)
    if "BlockId" not in df.columns:
        blk_re = re.compile(r"blk_[-\d]+")
        df["BlockId"] = df.get("Content", "").apply(
            lambda t: (m := blk_re.search(str(t))) and m.group(0) or "UNKNOWN"
        )
    return df


def _build_sessions_from_structured(struct_df, label_df) -> pd.DataFrame:
    label_map = {}
    if label_df is not None:
        ldf = label_df.copy()
        ldf.columns = [c.strip() for c in ldf.columns]
        bc = next((c for c in ldf.columns if "block" in c.lower()), ldf.columns[0])
        lc = next((c for c in ldf.columns if "label" in c.lower()), ldf.columns[1])
        label_map = {str(r[bc]).strip(): str(r[lc]).strip().lower()
                     for _, r in ldf.iterrows()}
    groups = defaultdict(list)
    for _, row in struct_df.iterrows():
        blk = str(row.get("BlockId", "UNKNOWN")).strip()
        eid = str(row.get("EventId", "E0")).strip()
        if blk not in ("UNKNOWN", "nan", ""):
            groups[blk].append(eid)
    records = []
    for blk, seq in groups.items():
        lbl = 1 if label_map.get(blk, "normal") == "anomaly" else 0
        records.append({"BlockId": blk, "Sequence": " ".join(seq),
                         "Label": lbl, "Length": len(seq)})
    return pd.DataFrame(records)


# ══════════════════════════════════════════════════════════════════════════════
# Layout C — Raw HDFS.log → Drain3
# ══════════════════════════════════════════════════════════════════════════════

def _parse_with_drain3(raw_log: Path) -> pd.DataFrame:
    try:
        from drain3 import TemplateMiner
    except ImportError:
        logger.warning("drain3 not installed — skipping"); return pd.DataFrame()
    logger.info("  Parsing raw logs with Drain3 (may take a few minutes)...")
    miner  = TemplateMiner()
    blk_re = re.compile(r"blk_[-\d]+")
    records, lines = [], raw_log.read_text(errors="ignore").splitlines()
    for lid, line in enumerate(tqdm(lines, desc="  Drain3", unit="lines")):
        line = line.strip()
        if not line: continue
        result = miner.add_log_message(line)
        blks   = blk_re.findall(line)
        records.append({"LineId": lid, "EventId": f"E{result['cluster_id']}",
                         "BlockId": blks[0] if blks else "UNKNOWN",
                         "Content": line})
    df = pd.DataFrame(records)
    df.to_csv(STRUCT_CSV, index=False)
    logger.success(f"  Drain3: {len(df):,} lines → {df['EventId'].nunique()} templates")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# Synthetic anomaly fallback
# ══════════════════════════════════════════════════════════════════════════════

def _inject_synthetic_anomalies(sessions: pd.DataFrame,
                                  ratio: float = 0.067) -> pd.DataFrame:
    logger.warning(f"  No real labels — injecting synthetic anomalies ({ratio*100:.1f}%)")
    df  = sessions.copy()
    rng = np.random.default_rng(42)
    all_events = [e for seq in df["Sequence"] for e in str(seq).split()]
    uniq       = list(set(all_events))
    n_anom     = int(len(df) * ratio)
    for idx in rng.choice(len(df), size=n_anom, replace=False):
        seq  = df.at[idx, "Sequence"].split()
        kind = rng.integers(0, 3)
        if kind == 0 and len(seq) > 2:
            seq = seq[:max(1, len(seq)//3)]
        elif kind == 1 and len(seq) > 3:
            mid = seq[1:-1]; rng.shuffle(mid)
            seq = [seq[0]] + mid + [seq[-1]]
        else:
            pos = rng.integers(1, max(2, len(seq)))
            seq = seq[:pos] + [rng.choice(uniq)]*int(rng.integers(3,7)) + seq[pos:]
        df.at[idx, "Sequence"] = " ".join(seq)
        df.at[idx, "Label"]    = 1
    logger.success(f"  Injected {int(df['Label'].sum())} synthetic anomalies")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# Vocabulary
# ══════════════════════════════════════════════════════════════════════════════

def _build_vocab(sessions: pd.DataFrame) -> dict:
    special = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
    events  = set()
    for seq in sessions["Sequence"]:
        events.update(str(seq).split())
    vocab = {t: i for i, t in enumerate(special)}
    for ev in sorted(events):
        if ev not in vocab:
            vocab[ev] = len(vocab)
    VOCAB_JSON.write_text(json.dumps(vocab, indent=2))
    return vocab


# ══════════════════════════════════════════════════════════════════════════════
# Download helpers
# ══════════════════════════════════════════════════════════════════════════════

def _fetch(url, dest):
    try:
        r = requests.get(url, timeout=30); r.raise_for_status()
        dest.write_bytes(r.content); return True
    except Exception as e:
        logger.debug(f"  fetch failed: {e}"); return False

def _try_urls(urls, dest, label):
    for url in urls:
        logger.info(f"  Downloading {label}...")
        if _fetch(url, dest):
            logger.success(f"  {label} → {dest.name} ({dest.stat().st_size//1024} KB)")
            return True
    return False


# ══════════════════════════════════════════════════════════════════════════════
# Public entry point
# ══════════════════════════════════════════════════════════════════════════════

def load_dataset() -> tuple[pd.DataFrame, dict]:
    # ── Cache hit ──────────────────────────────────────────────────────────────
    if SESSIONS_CSV.exists() and VOCAB_JSON.exists():
        logger.info("  Loading cached sessions...")
        sessions = pd.read_csv(SESSIONS_CSV)
        vocab    = json.loads(VOCAB_JSON.read_text())
        n        = int(sessions["Label"].sum())
        logger.success(f"  {len(sessions):,} sessions | {n:,} anomaly | vocab {len(vocab)}")
        return sessions, vocab

    sessions = None

    # A — Event_traces.csv
    if TRACES_CSV.exists():
        logger.success("  Layout A: Event_traces.csv detected")
        sessions = _load_from_event_traces(TRACES_CSV, LABEL_CSV)

    # B — structured CSV
    if sessions is None:
        for p in [DATA_DIR/"HDFS.log_structured.csv",
                  DATA_DIR/"HDFS_1"/"HDFS.log_structured.csv"]:
            if p.exists():
                import shutil
                if p != STRUCT_CSV: shutil.copy(p, STRUCT_CSV)
                ldf = pd.read_csv(LABEL_CSV) if LABEL_CSV.exists() else None
                sessions = _build_sessions_from_structured(_load_structured(), ldf)
                logger.success("  Layout B: structured CSV detected")
                break

    # C — raw log + Drain3
    if sessions is None and RAW_LOG.exists():
        struct_df = _parse_with_drain3(RAW_LOG)
        if len(struct_df):
            ldf      = pd.read_csv(LABEL_CSV) if LABEL_CSV.exists() else None
            sessions = _build_sessions_from_structured(struct_df, ldf)
            logger.success("  Layout C: raw HDFS.log parsed with Drain3")

    # D — download 2k sample
    if sessions is None:
        if not STRUCT_CSV.exists():
            if not _try_urls(URLS["structured_2k"], STRUCT_CSV, "HDFS 2k sample"):
                logger.error("Cannot download dataset. Check internet connection.")
                sys.exit(1)
        if not LABEL_CSV.exists():
            _try_urls(URLS["labels"], LABEL_CSV, "anomaly labels")
        ldf      = pd.read_csv(LABEL_CSV) if LABEL_CSV.exists() else None
        sessions = _build_sessions_from_structured(_load_structured(), ldf)
        logger.info("  Layout D: using 2k sample (demo mode)")

    # Synthetic fallback
    if int(sessions["Label"].sum()) == 0:
        sessions = _inject_synthetic_anomalies(sessions)

    vocab = _build_vocab(sessions)
    sessions.to_csv(SESSIONS_CSV, index=False)

    n    = int(sessions["Label"].sum())
    tot  = len(sessions)
    logger.success(f"  Dataset ready: {tot:,} sessions | "
                   f"{tot-n:,} normal | {n:,} anomaly ({100*n/max(tot,1):.1f}%)"
                   f" | vocab {len(vocab)}")
    return sessions, vocab