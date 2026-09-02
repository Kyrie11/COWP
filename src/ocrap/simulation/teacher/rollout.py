from __future__ import annotations

import numpy as np

from ocrap.data.schema import CandidatePrefix, CounterfactualFuture, RecoveryOption, SceneHistory

from .controllers import rollout_recovery_controller
from .margins import TeacherDiagnostics, teacher_margin


def compute_future_option_margins(history: SceneHistory, prefix: CandidatePrefix, futures: list[CounterfactualFuture], options: list[RecoveryOption], cfg: dict) -> tuple[np.ndarray, list[list[TeacherDiagnostics]]]:
    if str(cfg.get("simulation_backend", "ocrap_surrogate")) == "waymax_closed_loop":
        from ocrap.simulation.waymax_rollout import compute_waymax_future_option_margins

        return compute_waymax_future_option_margins(history, prefix, futures, options, cfg)
    horizon_steps = max(2, int(round(float(cfg.get("recovery_horizon_s", 4.0)) * float(cfg.get("sample_rate_hz", 10.0)))))
    M = np.zeros((len(futures), len(options)), dtype=np.float32)
    all_diag: list[list[TeacherDiagnostics]] = []
    controller_cache = [rollout_recovery_controller(prefix, opt, horizon_steps, cfg) for opt in options]
    for j, fut in enumerate(futures):
        row_diag: list[TeacherDiagnostics] = []
        for l, opt in enumerate(options):
            rec_states, rec_controls, cdiag = controller_cache[l]
            m, diag = teacher_margin(history, prefix, fut, opt, rec_states, rec_controls, cfg, cdiag)
            M[j, l] = m
            row_diag.append(diag)
        all_diag.append(row_diag)
    return M, all_diag
