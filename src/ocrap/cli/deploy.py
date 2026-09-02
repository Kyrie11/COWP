from __future__ import annotations

from pathlib import Path

import numpy as np

from ocrap.data.serialization import load_npz, write_json
from ocrap.models.data import iter_sample_paths_many
from ocrap.models.inference import load_model_bundle, predict_sample
from ocrap.planning.selector import crisp_select


def deploy(dataset: str, checkpoint: str | None, scene_id: str, time_index: int, output: str | None = None, calibration_json: str | None = None, delta: float | None = None, cfg: dict | None = None) -> dict:
    cfg = cfg or {}
    bundle = load_model_bundle(checkpoint, cfg)
    items = []
    for p in iter_sample_paths_many(dataset):
        d = load_npz(p)
        if str(np.asarray(d["scene_id"]).item()) == scene_id and int(np.asarray(d["time_index"]).item()) == int(time_index):
            items.append((p, d, predict_sample(d, bundle, cfg)))
    if not items:
        raise ValueError(f"No candidates found for scene_id={scene_id} time_index={time_index}")
    items.sort(key=lambda x: int(np.asarray(x[1]["candidate_index"]).item()))
    gamma = float((cfg or {}).get("selection", {}).get("gamma_rec", 0.0))
    if calibration_json:
        import json

        with Path(calibration_json).open("r", encoding="utf-8") as f:
            cal = json.load(f)
        if delta is not None and "thresholds" in cal:
            gamma = float(cal["thresholds"].get(str(delta), gamma))
        else:
            gamma = float(cal.get("gamma_rec", gamma))
    sel_cfg = cfg.get("selection", {}) if isinstance(cfg.get("selection", {}), dict) else {}
    utility = np.array([float(np.asarray(d["utility"]).item()) for _, d, _ in items])
    r_dep = np.array([float(pred.r_dep) for _, _, pred in items])
    hard = np.array([float(np.asarray(d["hard_violation"]).item()) for _, d, _ in items])
    harm = np.array([float(np.asarray(d["harm_proxy"]).item()) for _, d, _ in items])
    feasible = np.array([bool(int(np.asarray(d["feasible"]).item())) for _, d, _ in items])
    sel = crisp_select(utility, r_dep, hard, harm, feasible, gamma_rec=gamma, gamma_H=float(sel_cfg.get("gamma_H", 0.0)), gamma_D=float(sel_cfg.get("gamma_D", 5.0)))
    selected = items[sel.selected_index][1]
    selected_pred = items[sel.selected_index][2]
    result = {
        "scene_id": scene_id,
        "time_index": int(time_index),
        "selected_candidate_index": int(np.asarray(selected["candidate_index"]).item()),
        "reason": sel.reason,
        "source": "model" if bundle is not None else "teacher_fallback",
        "gamma_rec": gamma,
        "admitted_indices": [int(np.asarray(items[i][1]["candidate_index"]).item()) for i, ok in enumerate(sel.admitted) if ok],
        "pred_r_dep": float(selected_pred.r_dep),
        "pred_r_orc": float(selected_pred.r_orc),
        "pred_gap": float(selected_pred.gap),
        "teacher_r_dep_star": float(np.asarray(selected["r_dep_star"]).item()) if "r_dep_star" in selected else None,
        "utility": float(np.asarray(selected["utility"]).item()),
    }
    if output:
        write_json(result, output)
    return result
