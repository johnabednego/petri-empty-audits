"""Shared loading code. Every result in this repo is computed from the parquet files under DATA_DIR."""
import glob
import hashlib
import os
import re

import pandas as pd

DATA_DIR = os.environ.get("PETRI_DATA", "data")
SPLIT = re.compile(r"^(?P<behaviour>.+)_(?P<data>synth_docs|transcripts)_(?P<adv>adv_high|adv_kto)$")
NON_BEHAVIOUR = {"auditor_failure", "scenario_realism"}  # dimensions about the audit itself, not the target


def org_name(path):
    return os.path.basename(path).split("-0000")[0]


def files(subset="transcripts_all"):
    out = sorted(glob.glob(os.path.join(DATA_DIR, subset, "data", "*.parquet")))
    if not out:
        raise SystemExit(f"No parquet files under {DATA_DIR}/{subset}/data. Run src/download_data.py first.")
    return out


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def has_conversation(messages):
    roles = [m["role"] for m in messages]
    return ("user" in roles) or ("assistant" in roles)


def load_scores():
    """One row per audit transcript: the 37 judge scores plus structure and organism labels."""
    frames = []
    for path in files():
        org = org_name(path)
        labels = SPLIT.match(org).groupdict()
        df = pd.read_parquet(path)
        scores = pd.DataFrame(list(df["scores"]))
        msgs = df["messages"]
        scores["n_messages"] = msgs.apply(len).values
        scores["n_user"] = msgs.apply(lambda ms: sum(m["role"] == "user" for m in ms)).values
        scores["n_assistant"] = msgs.apply(lambda ms: sum(m["role"] == "assistant" for m in ms)).values
        scores["is_empty"] = ~msgs.apply(has_conversation).values
        scores["row"] = range(len(df))
        scores["organism"] = org
        for k, v in labels.items():
            scores[k] = v
        frames.append(scores)
    d = pd.concat(frames, ignore_index=True)
    d["config"] = d["data"].map({"synth_docs": "Synthetic documents", "transcripts": "Transcripts"}) + " + " + d["adv"].map({"adv_high": "SFT", "adv_kto": "KTO"})
    return d


def score_columns(d):
    meta = {"n_messages", "n_user", "n_assistant", "is_empty", "row", "organism", "behaviour", "data", "adv", "config"}
    return [c for c in d.columns if c not in meta]
