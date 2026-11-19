# LogBERT: Log Anomaly Detection with a Transformer

**Teach a small BERT what normal system logs look like, then flag log sessions it cannot predict.**

Large distributed systems write millions of log lines, and failures leave faint traces in them: a block that is never replicated, a write that is retried and abandoned. Nobody can read all of it, and labelled failures are rare. This project takes the **unsupervised** route. A transformer learns the language of *normal* behaviour, and anything that departs from it is flagged.

It is a from-scratch PyTorch + Hugging Face implementation of the main idea in *LogBERT: Log Anomaly Detection via BERT* (Guo et al., 2021). It is applied to the **HDFS** benchmark, one of the standard datasets for log anomaly detection. A single command (`./run.sh`) does the following:

1. gets the data and parses the logs into per-block sessions,
2. trains the model on normal sessions only,
3. scores every session and picks a decision threshold,
4. writes metrics, per-session scores and plots.

It uses an Apple Silicon GPU (MPS) or an NVIDIA GPU (CUDA) when one is available, and runs on CPU otherwise.

---

## How it works

### From log lines to "sentences"

In HDFS, every log line that mentions a block ID (`blk_…`) belongs to that block's **session**. Each line is reduced to its **log key**, the ID of its event template, so a session becomes a short sentence such as `E5 E22 E5 E11 E9 …`. Each session is labelled *Normal* or *Anomaly*.

`src/data.py` builds these sentences from whatever is in `data/`, using the first layout it finds:

| Layout | Files in `data/` | What happens |
|---|---|---|
| **A** | `Event_traces.csv` (+ `anomaly_label.csv`) | Already grouped by block. Tokens come **only** from the event column: loghub's `Features` (`[E5,E22,...]`), an `EventSequence`-style column, or columns named `E1, E2, …`. `Label`, `Type`, `TimeInterval`, `Latency` and other columns are never used. If no event column is found, the loader stops with an error. |
| **B** | `HDFS.log_structured.csv` or `HDFS_1/HDFS.log_structured.csv` (+ `anomaly_label.csv`) | Groups structured lines by `BlockId` into event-ID sequences. |
| **C** | `HDFS.log` (raw) (+ `anomaly_label.csv`) | Parses raw lines into templates with **Drain3**, takes the block ID with a regex, then groups as in B. |
| **D** | nothing | Downloads the 2,000-line HDFS sample from the LogPAI GitHub repos. Demo only. |

If no labels can be found, the loader **injects synthetic anomalies** into 6.7% of sessions, chosen with a fixed seed. Each one is altered in one of three ways: cut to its first third, its middle events shuffled, or a random event repeated 3–6 times as a burst. It then builds a vocabulary of `[PAD] [UNK] [CLS] [SEP] [MASK]` plus every event key, and caches `data/sessions.csv` and `data/vocabulary.json`. **Delete those two files to force a re-parse.**

### The model

A compact `BertForMaskedLM` built from scratch, with no pretrained weights:

| Setting | Value |
|---|---|
| Hidden size | 256 |
| Layers / attention heads | 4 / 4 |
| Feed-forward size | 1024 |
| Dropout | 0.1 |
| Sequence length | 50 tokens (`[CLS]` + up to 48 events + `[SEP]`) |

### Training: Masked Log-Key Prediction

Each time a session is fetched, 15% of its event keys (at least one) are hidden. Of those, 80% become `[MASK]`, 10% become a random event and 10% stay unchanged. The model learns to recover them from context. Because it sees **only normal sessions** (a random 90% of them), it learns which keys are likely to come next in healthy behaviour.

- AdamW (lr 3e-4, weight decay 0.01), linear schedule with 10% warm-up, gradient clipping at 1.0
- Batch size 64, 20 epochs, evaluated every 2 epochs, and the checkpoint with the best F1 is saved to `models/logbert_best.pt`

### Scoring: "how surprised was the model?"

For each session, the hidden positions are predicted. A position is a **miss** if the true key is not in the model's **top-50** guesses:

```
anomaly score = misses / hidden positions        0 = fully predictable, 1 = nothing predicted
```

A session is flagged when its score is at or above a threshold. The final threshold is chosen by sweeping 200 values in [0, 1] and keeping the one with the best F1.

### Outputs (`results/`)

| File | Contents |
|---|---|
| `metrics.json` | precision, recall, F1, accuracy, ROC-AUC, PR-AUC, confusion counts |
| `session_scores.csv` | per block: `BlockId, Label, AnomalyScore, Predicted, Correct` |
| `score_distribution.png`, `roc_curve.png`, `pr_curve.png`, `confusion_matrix.png`, `training_history.png` | plots |
| `history.json`, `run.log` | per-evaluation loss/F1, and the full log |

---

## Dataset

The target is **HDFS_v1** from [LogPAI loghub](https://github.com/logpai/loghub) (Xu et al., SOSP 2009): about 11 million log lines, **575,061 block sessions, 16,838 labelled anomalous** (2.9%). Download the HDFS_v1 archive from the loghub Zenodo release linked in that repository and place the files in `data/`:

```
data/
├── Event_traces.csv      # layout A, preferred
├── anomaly_label.csv     # BlockId,Label  (Normal / Anomaly)
└── HDFS.log              # optional: raw logs for layout C (Drain3)
```

`data/`, `hdfs_extracted/` and `HDFS_v1.zip` are git-ignored.

With no data present, the 2k-line sample is downloaded automatically. Its label URLs currently return 404, so that run falls back to synthetic anomalies. That mode is fine for a smoke test but not for drawing conclusions.

---

## Setup and running

Requirements: Python 3.9+. Dependencies are in `requirements.txt`: torch, transformers, pandas, numpy, scikit-learn, matplotlib, seaborn, loguru, requests, tqdm, drain3.

```bash
git clone https://github.com/Shrey48/logbert_final.git
cd logbert_final
chmod +x run.sh
./run.sh                 # full pipeline: data -> train -> evaluate -> report
```

`run.sh` checks the Python version, creates and activates `venv/`, installs the requirements if they are missing, prints the detected GPU, and then runs `python3 main.py` with any flags you pass:

| Command | Runs |
|---|---|
| `./run.sh` | phases 1–4 |
| `./run.sh --skip-train` | phases 1, 3, 4 with the saved `models/logbert_best.pt` |
| `./run.sh --phase 1` | load and parse data only |
| `./run.sh --phase 2` | data + training |
| `./run.sh --phase 3` (or `4`) | data + evaluation + report (needs a trained model) |

Without the wrapper:

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python3 main.py [--skip-train] [--phase N]
```

Hyperparameters are module constants. Edit them in `src/trainer.py` (`EPOCHS`, `BATCH_SIZE`, `LR`, `TOP_K`, `THRESH`), `src/dataset.py` (`MAX_SEQ_LEN`, `MASK_RATIO`) and `src/model.py` (`_bert_config`).

---

## Results so far

Two captured terminal logs are kept in `results/logs/`. **Neither is a valid full-dataset result. The full HDFS_v1 run needs to be redone with the current code.**

**`results/logs/2k_sample_smoke_test.txt`: pipeline check on the 2k sample (Apple M-series, MPS).** The sample gave 1,994 sessions and a vocabulary of 19. The label download failed, so 133 synthetic anomalies (6.7%) were injected. The model (3.36 M parameters) trained for 10 epochs in about 36 s, and the best checkpoint came from epoch 6. At the tuned threshold of 0.005 the results were:

| Precision | Recall | F1 | ROC-AUC | PR-AUC | Accuracy | TP / FP / FN |
|---|---|---|---|---|---|---|
| 0.360 | 0.135 | 0.197 | 0.559 | 0.277 | 0.926 | 18 / 32 / 115 |

A ROC-AUC of 0.56 is close to chance. The anomalies were synthetic and the sample is tiny, so this run shows only that the pipeline works end to end. It came from an earlier revision of the code (10 epochs; today's default is 20).

**`results/logs/hdfs_full_epoch1_pre_fix.txt`: first epoch on the full HDFS_v1 data.** 575,061 sessions were loaded (558,223 normal / 16,838 anomalous). Training used 502,400 normal sessions, the model had 29.96 M parameters, and each epoch was 7,850 steps. Epoch 1 of 20 took **6,896 s (about 1 h 55 min)** on MPS. The log stops there, with no evaluation.

This run was produced **before the column fix in `src/data.py`**. At that time, layout A did not recognise loghub's `Features` column. It joined every column of `Event_traces.csv` (including `Label`, `Type`, `TimeInterval` and `Latency`) into the token sequence, and each whole `[E5,E22,...]` list became one token. That is why the log reports a vocabulary of **103,501**, when HDFS has only a few dozen event types. It also means the label value was in the model's input. The current loader takes tokens from the event column only.

---

## Project layout

```
logbert_final/
├── run.sh              # one-command wrapper: venv, deps, GPU check, main.py
├── main.py             # 4-phase pipeline + CLI (--phase, --skip-train)
├── requirements.txt
├── src/
│   ├── data.py         # HDFS loading (layouts A–D), Drain3, labels, synthetic anomalies, vocab
│   ├── dataset.py      # LogSessionDataset: encode, pad, dynamic 15% masking
│   ├── model.py        # LogBERT (BertForMaskedLM) + top-k anomaly score, save/load
│   ├── trainer.py      # training loop, scoring, checkpointing, hyperparameters
│   ├── report.py       # metrics.json, threshold sweep, plots
│   └── device.py       # MPS / CUDA / CPU selection
├── results/
│   └── logs/           # captured terminal logs (tracked); other results/ output is git-ignored
├── data/               # (git-ignored) datasets + cached sessions.csv / vocabulary.json
└── models/             # (git-ignored) logbert_best.pt
```

---

## Limitations and next steps

- **Re-run the full dataset.** Delete `data/sessions.csv` and `data/vocabulary.json` so the sessions are rebuilt with the fixed loader. Then train and evaluate on HDFS_v1 and confirm the vocabulary is a few dozen events.
- **Train and test overlap.** The evaluation set is *all* sessions, so it includes the normal sessions used for training. The best checkpoint and the final threshold are both chosen on that same set. A clean protocol would hold out normal and anomalous sessions that are never used for training or tuning.
- **Scores vary between runs.** The hidden positions are redrawn every time a session is fetched, including at evaluation. Fixed masks, or averaging over several draws, would make scores repeatable.
- **Only one training objective.** The paper also uses *Volume of Hypersphere Minimisation*. This implementation trains with masked-key prediction only.
- **Long sessions are truncated** to 48 events.
- **Batch, not streaming.** The pipeline scores a stored dataset. Streaming inference would be a natural extension.

## References

- Guo, H., Yuan, S., Wu, X. *LogBERT: Log Anomaly Detection via BERT.* IJCNN 2021.
- Xu, W., Huang, L., Fox, A., Patterson, D., Jordan, M. *Detecting Large-Scale System Problems by Mining Console Logs.* SOSP 2009.
- He, P. et al. *Drain: An Online Log Parsing Approach with Fixed Depth Tree.* ICWS 2017.
