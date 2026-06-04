from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import pandas as pd

from cowp.core.constants import MechanismToken, TOKEN_NAMES
from cowp.data.cache_schema import validate_numeric_invariants, validate_schema


def validate_label_file(path: str | Path, cfg: dict) -> list[str]:
    data = np.load(path, allow_pickle=True)
    d = {k: data[k] for k in data.files}
    return validate_schema(d, cfg, strict=True) + validate_numeric_invariants(d, cfg)


def summarize_label_file(path: str | Path) -> dict[str, float | int | str]:
    data = np.load(path, allow_pickle=True)
    sid = str(data["scenario/id"].item()) if "scenario/id" in data else Path(path).stem
    cand_valid = data["cowp/candidates/valid"].astype(bool)
    crit_valid = data["cowp/critical/valid"].astype(bool)
    witness = data["cowp/witness/exists"].astype(bool)
    pair_mask = cand_valid[:, None] & crit_valid[None, :]
    tokens = data["cowp/witness/token"][witness]
    return {
        "scenario_id": sid,
        "candidate_valid": int(cand_valid.sum()),
        "critical_valid": int(crit_valid.sum()),
        "positive_pairs": int(np.sum(witness & pair_mask)),
        "valid_pairs": int(np.sum(pair_mask)),
        "positive_pair_ratio": float(np.sum(witness & pair_mask) / max(np.sum(pair_mask), 1)),
        "false_safe_candidates": int(np.sum(data["cowp/candidates/false_safe"].astype(bool) & cand_valid)),
        "ncf_candidates": int(np.sum(data["cowp/candidates/noncoercive_feasible"].astype(bool) & cand_valid)),
        "mean_opr": float(np.mean(data["cowp/witness/opr"][pair_mask])) if np.any(pair_mask) else 0.0,
        "max_cbs": float(np.max(data["cowp/witness/burden_total"][pair_mask])) if np.any(pair_mask) else 0.0,
        "tokens": ",".join(str(int(t)) for t in tokens),
    }


def diagnose_dataset(labels_dir: str | Path, cfg: dict, output_dir: str | Path) -> pd.DataFrame:
    labels_dir = Path(labels_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    validation_errors = {}
    for p in sorted(labels_dir.glob("*.npz")):
        errs = validate_label_file(p, cfg)
        if errs:
            validation_errors[p.name] = errs
        rows.append(summarize_label_file(p))
    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "dataset_diagnostics.csv", index=False)
    with (output_dir / "validation_errors.json").open("w", encoding="utf-8") as f:
        json.dump(validation_errors, f, indent=2, ensure_ascii=False)
    token_counter: Counter[str] = Counter()
    if not df.empty:
        for tokstr in df["tokens"].fillna(""):
            for tok in str(tokstr).split(","):
                if tok:
                    token_counter[TOKEN_NAMES.get(MechanismToken(int(tok)), tok)] += 1
    stats = {
        "num_scenes": int(len(df)),
        "mean_candidate_valid": float(df["candidate_valid"].mean()) if not df.empty else 0.0,
        "mean_critical_valid": float(df["critical_valid"].mean()) if not df.empty else 0.0,
        "positive_pair_ratio": float(df["positive_pairs"].sum() / max(df["valid_pairs"].sum(), 1)) if not df.empty else 0.0,
        "false_safe_candidate_ratio": float(df["false_safe_candidates"].sum() / max(df["candidate_valid"].sum(), 1)) if not df.empty else 0.0,
        "ncf_candidate_ratio": float(df["ncf_candidates"].sum() / max(df["candidate_valid"].sum(), 1)) if not df.empty else 0.0,
        "mean_opr": float(df["mean_opr"].mean()) if not df.empty else 0.0,
        "mean_max_cbs": float(df["max_cbs"].mean()) if not df.empty else 0.0,
        "mechanism_token_counts": dict(token_counter),
        "validation_error_files": int(len(validation_errors)),
    }
    with (output_dir / "dataset_diagnostics_summary.json").open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    return df
