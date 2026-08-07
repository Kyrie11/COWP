from __future__ import annotations

import argparse
import hashlib
import os
import pickle
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from cowp.core.config import load_config
from cowp.utils.progress import tqdm_iter
from cowp.waymax_eval.baselines import planner_for_method
from cowp.waymax_eval.metrics_cowp import (
    metrics_from_labels,
    module_effect_metrics,
    stress_acceptance_metrics,
    witness_table_from_labels,
)

# v16.8.6: 05_make_tables is a label/certificate-space diagnostic.  The old
# implementation materialized every array in every label NPZ, including the very
# large natural/response trajectory banks that no table metric or planner uses.
# Keep only the semantic tensors actually consumed below.  Candidate trajectory
# is compacted to its first/last sample because label-space progress only needs
# the endpoint displacement and planner selection never inspects intermediate
# points.
TABLE_KEYS = {
    "cowp/candidates/trajectory",
    "cowp/candidates/valid",
    "cowp/candidates/conventional_safe",
    "cowp/candidates/noncoercive_feasible",
    "cowp/candidates/false_safe",
    "cowp/candidates/ego_utility_prior",
    "cowp/candidates/is_neutral",
    "cowp/candidates/macro_type",
    "cowp/critical/valid",
    "cowp/natural/beta",
    "cowp/witness/exists",
    "cowp/witness/token",
    "cowp/witness/burden_total",
    "cowp/witness/min_safe_burden",
    "cowp/witness/opr",
    "cowp/witness/c_i",
    "cowp/witness/natural_conflict_mass_by_source",
    "cowp/witness/natural_mass_by_source",
    "cowp/witness/low_safe_mass_by_source",
}


def _fingerprint(paths: list[Path]) -> str:
    h = hashlib.sha256()
    for p in paths:
        st = p.stat()
        h.update(p.name.encode("utf-8"))
        h.update(str(int(st.st_size)).encode())
        h.update(str(int(st.st_mtime_ns)).encode())
    return h.hexdigest()


def _read_compact_label(path: Path) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    with np.load(path, allow_pickle=False) as data:
        available = set(data.files)
        for key in TABLE_KEYS & available:
            arr = np.asarray(data[key])
            if key == "cowp/candidates/trajectory" and arr.ndim >= 3 and arr.shape[1] > 1:
                # [K,T,D] -> [K,2,D]; preserves exact endpoint progress and a
                # valid trajectory object for PlannerDecision without holding T=80.
                arr = np.stack((arr[:, 0], arr[:, -1]), axis=1)
            out[key] = arr
    missing = {
        "cowp/candidates/trajectory",
        "cowp/candidates/valid",
        "cowp/candidates/conventional_safe",
        "cowp/candidates/ego_utility_prior",
        "cowp/critical/valid",
        "cowp/witness/exists",
        "cowp/witness/opr",
        "cowp/witness/c_i",
    } - set(out)
    if missing:
        raise KeyError(f"{path}: missing table-required label keys: {sorted(missing)}")
    return out


def _load_labels(paths: list[Path], cache_path: Path | None, load_workers: int) -> list[dict[str, np.ndarray]]:
    fp = _fingerprint(paths)
    if cache_path is not None and cache_path.is_file():
        try:
            with cache_path.open("rb") as f:
                payload = pickle.load(f)
            if payload.get("schema_version") == "cowp_v16_8_6_compact_table_cache_v1" and payload.get("fingerprint") == fp:
                rows = payload.get("labels")
                if isinstance(rows, list) and len(rows) == len(paths):
                    print(f"Loaded compact table cache: {cache_path} ({len(rows)} scenes)")
                    return rows
        except Exception as exc:
            print(f"Ignoring stale/unreadable compact table cache {cache_path}: {type(exc).__name__}: {exc}")

    workers = max(1, int(load_workers))
    if workers == 1:
        rows = [_read_compact_label(p) for p in tqdm_iter(paths, enabled=True, total=len(paths), desc="Load compact labels for tables", unit="file")]
    else:
        rows = []
        with ThreadPoolExecutor(max_workers=workers) as ex:
            iterator = ex.map(_read_compact_label, paths)
            rows.extend(tqdm_iter(iterator, enabled=True, total=len(paths), desc="Load compact labels for tables", unit="file"))

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = cache_path.with_suffix(cache_path.suffix + ".tmp")
        with tmp.open("wb") as f:
            pickle.dump({
                "schema_version": "cowp_v16_8_6_compact_table_cache_v1",
                "fingerprint": fp,
                "num_scenes": len(rows),
                "labels": rows,
            }, f, protocol=pickle.HIGHEST_PROTOCOL)
        tmp.replace(cache_path)
        print(f"Wrote compact table cache: {cache_path}")
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate label-space COWP tables without loading response/natural trajectory banks.")
    ap.add_argument("--data-config", default="configs/data.yaml")
    ap.add_argument("--label-config", default="configs/label.yaml")
    ap.add_argument("--eval-config", default="configs/eval.yaml")
    ap.add_argument("--labels-dir", default=None)
    ap.add_argument("--output-dir", default="outputs/tables")
    ap.add_argument("--load-workers", type=int, default=int(os.environ.get("LABEL_TABLE_LOAD_WORKERS", "8")))
    ap.add_argument("--compact-cache", default=None, help="Persistent compact pickle cache. Default: <output-dir>/compact_label_table_cache.pkl")
    ap.add_argument("--no-compact-cache", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.label_config, args.data_config, args.eval_config)
    labels_dir = args.labels_dir or cfg["outputs"]["labels_dir"]
    methods = cfg.get("eval", {}).get("methods", ["cowp"])
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    label_paths = sorted(Path(labels_dir).glob("*.npz"))
    if not label_paths:
        raise FileNotFoundError(f"No label NPZ files under {labels_dir}")
    cache_path = None if args.no_compact_cache else Path(args.compact_cache or (out / "compact_label_table_cache.pkl"))
    label_dicts = _load_labels(label_paths, cache_path, args.load_workers)

    rows = []
    stress_rows = []
    method_decisions: dict[str, list[tuple[int, np.ndarray]]] = {}
    method_selected: dict[str, list[int]] = {}
    method_metrics: dict[str, dict[str, float]] = {}
    for method in methods:
        planner = planner_for_method(method, cfg)
        decisions: list[tuple[int, np.ndarray]] = []
        selected: list[int] = []
        for label in label_dicts:
            dec = planner.select_from_labels(label)
            decisions.append((int(dec.candidate_index), np.asarray(dec.accepted_mask, dtype=bool)))
            selected.append(int(dec.candidate_index))
        method_decisions[method] = decisions
        method_selected[method] = selected
        mm = metrics_from_labels(selected, label_dicts)
        method_metrics[method] = mm
        rows.append({"Method": method, **mm})
        stress_rows.append({"Method": method, **stress_acceptance_metrics(decisions, label_dicts)})

    df = pd.DataFrame(rows)
    df.to_csv(out / "main_results.csv", index=False)
    df[["Method", "FSR", "CBS", "OPR", "EP"]].to_csv(out / "ablation.csv", index=False)
    pd.DataFrame(stress_rows).to_csv(out / "stress_test.csv", index=False)
    witness_rows = [{"Method": "COWP rule certificate", **witness_table_from_labels(label_dicts)}]
    pd.DataFrame(witness_rows).to_csv(out / "witness_quality.csv", index=False)
    # Reuse the decisions already computed above.  v16.8.5 accidentally called
    # every label-space planner a second time inside module_effect_metrics.
    module_metrics = module_effect_metrics(
        label_dicts,
        cfg,
        methods=list(methods),
        precomputed_decisions=method_decisions,
        precomputed_selected=method_selected,
        precomputed_metrics=method_metrics,
    )
    pd.DataFrame([{"Method": k, **v} for k, v in module_metrics.items()]).to_csv(out / "module_effects.csv", index=False)
    print(df.to_string(index=False))
    print(f"Wrote module-effect diagnostics to {out / 'module_effects.csv'}")


if __name__ == "__main__":
    main()
