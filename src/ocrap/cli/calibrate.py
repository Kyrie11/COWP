from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from ocrap.algorithms.lcv import finite_sample_upper_quantile
from ocrap.data.serialization import load_npz, write_json
from ocrap.models.data import expand_split_roles, iter_sample_paths_many, scalar_metadata_for_path
from ocrap.models.inference import load_model_bundle, predict_sample
from ocrap.evaluation.metrics import option_execution_semantics, predicted_option_success


def _calibration_score(d: dict, pred, cfg: dict) -> float:
    mode = str((cfg.get("calibration", {}) or {}).get("score", "r_dep")).strip().lower()
    if mode in {"r_dep", "rec", "recoverability"}:
        return float(pred.r_dep)
    if mode in {"rec_lcb", "lcb", "r_dep_lcb"}:
        beta = float((cfg.get("selection", {}) or {}).get("lcb_beta", 0.10))
        return float(pred.r_dep - beta * max(0.0, pred.gap))
    if mode in {"pred_drs", "drs", "shared_option_success", "option_drs"}:
        gamma = float((cfg.get("selection", {}) or {}).get("drs_success_gamma", 0.0))
        return float(predicted_option_success(
            pred.q, pred.root_probs, gamma=gamma,
            root_valid=d.get("root_valid", None), option_valid=d.get("option_valid", None),
            semantics=option_execution_semantics(cfg),
        ))
    raise ValueError(f"Unknown calibration.score={mode!r}; expected r_dep, rec_lcb, or pred_drs")


def _sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _prediction_cache_signature(checkpoint: str | None, cfg: dict) -> tuple[str, str]:
    """Return checkpoint/config signatures for an inference-score cache.

    v48.52 intentionally excludes only calibration bookkeeping fields that do
    not affect model inference or the score definition.  The cache is scoped to
    one temporary calibration directory, so it is never a cross-run source of
    model predictions.  Any checkpoint/config mismatch invalidates all entries.
    """
    checkpoint_sha = "teacher_fallback"
    if checkpoint:
        ckpt = Path(checkpoint)
        if not ckpt.is_file():
            raise FileNotFoundError(checkpoint)
        checkpoint_sha = _sha256_file(ckpt)

    inference_cfg = copy.deepcopy(cfg)
    cal = dict((inference_cfg.get("calibration", {}) or {}))
    for key in (
        "required_min_for_delta",
        "allowed_split_ids",
        "exact_split_ids",
        "allow_validation_fallback",
        "deltas",
        "strict",
        "numerical_margin",
        "prediction_cache_json",
    ):
        cal.pop(key, None)
    inference_cfg["calibration"] = cal
    payload = json.dumps(inference_cfg, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
    cfg_sha = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return checkpoint_sha, cfg_sha


def _load_prediction_cache(path: Path | None, checkpoint_sha: str, cfg_sha: str) -> dict[str, dict[str, float]]:
    if path is None or not path.is_file():
        return {}
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if (
        doc.get("schema") != "ocrap-standard-calibration-prediction-cache-v1"
        or doc.get("checkpoint_sha256") != checkpoint_sha
        or doc.get("inference_cfg_sha256") != cfg_sha
    ):
        return {}
    entries = doc.get("entries")
    return entries if isinstance(entries, dict) else {}


def _write_prediction_cache(
    path: Path | None,
    checkpoint_sha: str,
    cfg_sha: str,
    entries: dict[str, dict[str, float]],
) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = {
        "schema": "ocrap-standard-calibration-prediction-cache-v1",
        "checkpoint_sha256": checkpoint_sha,
        "inference_cfg_sha256": cfg_sha,
        "entries": entries,
    }
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def calibrate(dataset: str, checkpoint: str | None = None, output: str | None = None, cfg: dict | None = None) -> dict:
    cfg = cfg or {}
    deltas = cfg.get("calibration", {}).get("deltas", [0.01, 0.05, 0.10])
    strict = bool(cfg.get("calibration", {}).get("strict", True))
    numerical_margin = float(cfg.get("calibration", {}).get("numerical_margin", 0.0))
    cal_cfg = cfg.get("calibration", {}) or {}
    required = int(cal_cfg.get("required_min_for_delta", 100))
    allowed_raw = cal_cfg.get("allowed_split_ids", "calibration")
    if isinstance(allowed_raw, str):
        requested_splits = {x.strip() for x in allowed_raw.split(",") if x.strip()}
    else:
        requested_splits = {str(x).strip() for x in allowed_raw if str(x).strip()}
    exact_split_ids = bool(cal_cfg.get("exact_split_ids", False))
    allowed_splits = (requested_splits or {"calibration"}) if exact_split_ids else expand_split_roles(requested_splits or {"calibration"})
    allow_val_fallback = bool(cal_cfg.get("allow_validation_fallback", True))

    cache_raw = str(cal_cfg.get("prediction_cache_json", "") or "").strip()
    cache_path = Path(cache_raw) if cache_raw else None
    checkpoint_sha, cfg_sha = _prediction_cache_signature(checkpoint, cfg)
    cache_entries = _load_prediction_cache(cache_path, checkpoint_sha, cfg_sha)
    cache_hits = 0
    cache_misses = 0
    cache_dirty = False

    # v48.52 runtime optimization: model construction is delayed until the first
    # cache miss.  A cache hit was created earlier in the same atomic calibration
    # attempt under the exact same checkpoint/config signatures, so reusing the
    # stored float score is numerically identical to repeating predict_sample.
    bundle = None
    bundle_loaded = False

    def ensure_bundle():
        nonlocal bundle, bundle_loaded
        if not bundle_loaded:
            bundle = load_model_bundle(checkpoint, cfg)
            bundle_loaded = True
        return bundle

    def score_path(p: Path) -> tuple[float, float]:
        nonlocal cache_hits, cache_misses, cache_dirty
        key = str(Path(p).resolve(strict=False))
        cached = cache_entries.get(key)
        if isinstance(cached, dict) and "score" in cached and "teacher_r_dep" in cached:
            score = float(cached["score"])
            teacher_r_dep = float(cached["teacher_r_dep"])
            if np.isfinite(score) and np.isfinite(teacher_r_dep):
                cache_hits += 1
                return score, teacher_r_dep
        cache_misses += 1
        d = load_npz(p)
        pred = predict_sample(d, ensure_bundle(), cfg)
        score = float(_calibration_score(d, pred, cfg))
        teacher_r_dep = float(np.asarray(d["r_dep_star"]).item())
        if cache_path is not None:
            cache_entries[key] = {"score": score, "teacher_r_dep": teacher_r_dep}
            cache_dirty = True
        return score, teacher_r_dep

    paths = iter_sample_paths_many(dataset)
    print({"event": "calibrate_start", "num_npz_paths": len(paths), "dataset": str(dataset)}, flush=True)
    scores = []
    teacher = []
    used_splits = []
    for idx, p in enumerate(paths, 1):
        if idx == 1 or idx % 1000 == 0:
            print({"event": "calibrate_progress", "seen": idx, "kept": len(scores)}, flush=True)
        split = str(scalar_metadata_for_path(p, "split_id", ""))
        if split not in allowed_splits:
            continue
        score, teacher_r_dep = score_path(p)
        scores.append(score)
        teacher.append(teacher_r_dep)
        used_splits.append(split)
    warnings = []
    if not scores and allow_val_fallback:
        warnings.append("no requested calibration split found; falling back to validation role")
        val_splits = expand_split_roles({"val"})
        for idx, p in enumerate(paths, 1):
            if idx == 1 or idx % 1000 == 0:
                print({"event": "calibrate_val_fallback_progress", "seen": idx, "kept": len(scores)}, flush=True)
            split = str(scalar_metadata_for_path(p, "split_id", ""))
            if split not in val_splits:
                continue
            score, teacher_r_dep = score_path(p)
            scores.append(score)
            teacher.append(teacher_r_dep)
            used_splits.append(split)

    if cache_dirty:
        _write_prediction_cache(cache_path, checkpoint_sha, cfg_sha, cache_entries)

    scores = np.asarray(scores, dtype=float)
    teacher = np.asarray(teacher, dtype=float)
    neg = scores[teacher < 0]
    if len(scores) == 0:
        warnings.append("no samples matched the requested calibration split role(s)")
    if len(neg) < required:
        warnings.append(f"num_negative < required_min_for_delta ({len(neg)} < {required})")

    def _delta_key(x) -> str:
        try:
            return f"{float(x):g}"
        except Exception:
            return str(x)

    thresholds = {}
    for delta in deltas:
        try:
            thresholds[_delta_key(delta)] = finite_sample_upper_quantile(neg, float(delta), numerical_margin, strict)
        except ValueError:
            thresholds[_delta_key(delta)] = float("inf")
    eval_delta = (cfg.get("evaluation", {}) or {}).get("delta", None) if isinstance(cfg.get("evaluation", {}), dict) else None
    default_delta = _delta_key(eval_delta if eval_delta is not None else (deltas[0] if deltas else 0.05))
    if default_delta not in thresholds and thresholds:
        default_delta = next(iter(thresholds.keys()))
    source = "model" if checkpoint is not None else ("model" if bundle_loaded and bundle is not None else "teacher_fallback")
    result = {
        "num_samples": int(len(scores)),
        "num_negative": int(len(neg)),
        "source": source,
        "score_mode": str((cfg.get("calibration", {}) or {}).get("score", "r_dep")),
        "splits": sorted(set(used_splits)),
        "requested_split_roles": sorted(requested_splits),
        "allowed_split_ids": sorted(allowed_splits),
        "exact_split_ids": exact_split_ids,
        "allow_validation_fallback": allow_val_fallback,
        "thresholds": thresholds,
        "default_delta": default_delta,
        "gamma_rec": thresholds.get(default_delta, 0.0),
        "warnings": warnings,
        "prediction_cache": {
            "enabled": cache_path is not None,
            "path": str(cache_path) if cache_path is not None else None,
            "checkpoint_sha256": checkpoint_sha if cache_path is not None else None,
            "inference_cfg_sha256": cfg_sha if cache_path is not None else None,
            "hits": int(cache_hits),
            "misses": int(cache_misses),
        },
    }
    if output:
        write_json(result, output)
    return result
