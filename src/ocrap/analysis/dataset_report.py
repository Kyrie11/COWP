from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from ocrap.data.serialization import ensure_dir, load_npz, write_json
from ocrap.models.data import iter_sample_paths_many


def _scalar(d: dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        return float(np.asarray(d.get(key, default)).item())
    except Exception:
        return default


def _str_scalar(d: dict[str, Any], key: str, default: str = "") -> str:
    try:
        return str(np.asarray(d.get(key, default)).item())
    except Exception:
        return default


def _json_field(x: Any, default: Any) -> Any:
    try:
        if isinstance(x, (bytes, bytearray)):
            return json.loads(x.decode("utf-8"))
        arr = np.asarray(x)
        if arr.shape == ():
            val = arr.item()
            if isinstance(val, (bytes, bytearray)):
                val = val.decode("utf-8")
            if isinstance(val, str):
                return json.loads(val)
        if isinstance(x, str):
            return json.loads(x)
    except Exception:
        pass
    return default


def _stats(x: list[float]) -> dict[str, float | None]:
    arr = np.asarray(x, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {k: None for k in ["mean", "std", "min", "p05", "p25", "p50", "p75", "p95", "max"]}
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "p05": float(np.quantile(arr, 0.05)),
        "p25": float(np.quantile(arr, 0.25)),
        "p50": float(np.quantile(arr, 0.50)),
        "p75": float(np.quantile(arr, 0.75)),
        "p95": float(np.quantile(arr, 0.95)),
        "max": float(arr.max()),
    }


def _maybe_plot(out_dir: Path, rows: list[dict[str, Any]]) -> list[str]:
    paths: list[str] = []
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        return paths
    if not rows:
        return paths
    r_dep = np.array([r["r_dep"] for r in rows], dtype=float)
    r_orc = np.array([r["r_orc"] for r in rows], dtype=float)
    gap = np.array([r["gap"] for r in rows], dtype=float)
    art = np.array([r["artifact"] for r in rows], dtype=bool)

    # Backward-compatible simple plots.
    fig = plt.figure(figsize=(8.0, 4.8))
    plt.hist(r_dep[np.isfinite(r_dep)], bins=50, alpha=0.85)
    plt.axvline(0.0, color="k", lw=1.0)
    plt.xlabel("Deployable recovery score after the candidate prefix\n(higher is safer; 0 means barely recoverable)")
    plt.ylabel("Number of candidate prefixes")
    plt.title("How much deployable recovery headroom do candidate prefixes preserve?")
    p = out_dir / "hist_r_dep.png"
    fig.savefig(p, dpi=180, bbox_inches="tight")
    plt.close(fig)
    paths.append(str(p))

    fig = plt.figure(figsize=(8.0, 4.8))
    plt.hist(gap[np.isfinite(gap)], bins=50, alpha=0.85)
    plt.axvline(0.0, color="k", lw=1.0)
    plt.xlabel("Oracle-to-deployable recovery gap\n(hindsight score minus deployable score)")
    plt.ylabel("Number of candidate prefixes")
    plt.title("How often does hindsight recovery overestimate deployable recovery?")
    p = out_dir / "hist_oracle_gap.png"
    fig.savefig(p, dpi=180, bbox_inches="tight")
    plt.close(fig)
    paths.append(str(p))

    fig = plt.figure(figsize=(8.0, 5.4))
    plt.scatter(r_dep[~art], r_orc[~art], s=6, alpha=0.5, label="non-artifact")
    if art.any():
        plt.scatter(r_dep[art], r_orc[art], s=8, alpha=0.7, label="oracle artifact")
    plt.axvline(0.0, color="k", lw=1.0)
    plt.axhline(0.0, color="k", lw=1.0)
    plt.xlabel("Deployable recovery score after the prefix")
    plt.ylabel("Hindsight/oracle recovery score")
    plt.title("Oracle recovery vs. actually deployable recovery")
    plt.legend(loc="best")
    p = out_dir / "scatter_oracle_vs_deployable.png"
    fig.savefig(p, dpi=180, bbox_inches="tight")
    plt.close(fig)
    paths.append(str(p))

    # Presentation-oriented figures.  Keep each figure isolated: a failure in one
    # optional visualization should not prevent later figures (especially the toy
    # gallery) from being written.
    warnings: list[str] = []
    try:
        from ocrap.analysis.visualization import (
            plot_criticality_ladder,
            plot_gap_by_category,
            plot_recoverability_story,
            plot_regime_breakdown,
            write_toy_gallery,
        )

        for fn in (
            plot_recoverability_story,
            plot_criticality_ladder,
            plot_regime_breakdown,
            plot_gap_by_category,
        ):
            try:
                maybe = fn(rows, out_dir)
                if maybe:
                    paths.append(maybe)
            except Exception as exc:
                warnings.append(f"{fn.__name__}: {exc}")

        try:
            paths.extend(write_toy_gallery(rows, out_dir, max_examples=4))
        except Exception as exc:
            warnings.append(f"write_toy_gallery: {exc}")
    except Exception as exc:
        # Keep analyze-dataset robust on machines without a display/matplotlib extras.
        warnings.append(f"presentation_visualizations: {exc}")

    if warnings:
        (out_dir / "visualization_warning.txt").write_text("\n".join(warnings), encoding="utf-8")
    return paths


def analyze_dataset(dataset: str | Path, output: str | Path, max_samples: int | None = None, plots: bool = True) -> dict[str, Any]:
    out_dir = ensure_dir(output)
    paths = iter_sample_paths_many(dataset, max_samples=max_samples)
    rows: list[dict[str, Any]] = []
    splits: Counter[str] = Counter()
    regimes: Counter[str] = Counter()
    macro_counts: Counter[str] = Counter()
    scene_time: set[tuple[str, int]] = set()
    scenes: set[str] = set()
    group_rows: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)

    for p in paths:
        d = load_npz(p)
        scene = _str_scalar(d, "scene_id")
        t = int(round(_scalar(d, "time_index", -1)))
        split = _str_scalar(d, "split_id", "unknown")
        row = {
            "path": str(p),
            "scene_id": scene,
            "time_index": t,
            "candidate_index": int(round(_scalar(d, "candidate_index", -1))),
            "split": split,
            "macro_id": int(round(_scalar(d, "prefix_macro_id", -1))),
            "utility": _scalar(d, "utility"),
            "hard": _scalar(d, "hard_violation"),
            "harm": _scalar(d, "harm_proxy"),
            "r_dep": _scalar(d, "r_dep_star"),
            "r_orc": _scalar(d, "r_orc_star"),
            "gap": _scalar(d, "oracle_gap_star", _scalar(d, "r_orc_star") - _scalar(d, "r_dep_star")),
            "artifact": bool(round(_scalar(d, "i_art_star"))),
            "is_nominal": bool(round(_scalar(d, "is_nominal"))),
            "post_contact": False,
            "normal": False,
            "occluded": False,
            "near_contact": False,
            "low_headroom": False,
        }
        reg = _json_field(d.get("regime_label", "{}"), {})
        if isinstance(reg, dict):
            for k, v in reg.items():
                if bool(v):
                    regimes[str(k)] += 1
                    if str(k) in row:
                        row[str(k)] = True
        rows.append(row)
        splits[split] += 1
        macro_counts[str(row["macro_id"])] += 1
        scenes.add(scene)
        scene_time.add((scene, t))
        group_rows[(scene, t)].append(row)

    group_min_dep = [min(r["r_dep"] for r in rs) for rs in group_rows.values() if rs]
    group_max_gap = [max(r["gap"] for r in rs) for rs in group_rows.values() if rs]
    critical_bins = Counter()
    for r in rows:
        if r["r_dep"] >= 0.50 and r["gap"] < 0.25 and not r["artifact"]:
            critical_bins["normal_high_headroom"] += 1
        elif r["r_dep"] >= 0.0:
            critical_bins["recoverable_low_or_mixed_headroom"] += 1
        elif r["r_orc"] >= 0.0:
            critical_bins["oracle_only_artifact"] += 1
        else:
            critical_bins["unrecoverable_or_critical"] += 1

    def top_rows(pred, n=8):
        arr = [r for r in rows if pred(r)]
        arr.sort(key=lambda r: (r["gap"], -r["r_dep"], r["utility"]), reverse=True)
        keep_keys = ["scene_id", "time_index", "candidate_index", "split", "macro_id", "utility", "r_dep", "r_orc", "gap", "artifact", "normal", "low_headroom", "near_contact", "post_contact", "occluded"]
        return [{k: r[k] for k in keep_keys} for r in arr[:n]]

    report = {
        "num_samples": int(len(rows)),
        "num_scenes": int(len(scenes)),
        "num_scene_time_groups": int(len(scene_time)),
        "splits": dict(splits),
        "regimes": dict(regimes),
        "criticality_bins": dict(critical_bins),
        "artifact_fraction": float(np.mean([r["artifact"] for r in rows])) if rows else 0.0,
        "normal_fraction": float(np.mean([r["normal"] for r in rows])) if rows else 0.0,
        "post_contact_fraction": float(np.mean([r["post_contact"] for r in rows])) if rows else 0.0,
        "macro_counts": dict(macro_counts),
        "r_dep_stats": _stats([r["r_dep"] for r in rows]),
        "r_orc_stats": _stats([r["r_orc"] for r in rows]),
        "oracle_gap_stats": _stats([r["gap"] for r in rows]),
        "group_min_r_dep_stats": _stats(group_min_dep),
        "group_max_oracle_gap_stats": _stats(group_max_gap),
        "toy_examples": {
            "oracle_artifacts_high_gap": top_rows(lambda r: r["r_orc"] >= 0 and r["r_dep"] < 0, 10),
            "normal_high_headroom": top_rows(lambda r: r["normal"] and r["r_dep"] >= 0.5 and r["gap"] < 0.25, 10),
            "post_contact_or_secondary": top_rows(lambda r: r["post_contact"], 10),
            "near_contact_low_headroom": top_rows(lambda r: r["near_contact"] or r["low_headroom"], 10),
        },
    }
    if plots:
        report["plot_files"] = _maybe_plot(out_dir, rows)
    write_json(report, out_dir / "dataset_report.json")
    write_json(report["toy_examples"], out_dir / "toy_examples.json")
    return report
