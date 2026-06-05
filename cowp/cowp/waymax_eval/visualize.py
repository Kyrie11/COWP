from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from cowp.core.constants import MechanismToken, TOKEN_NAMES


def _token_name(value: int) -> str:
    try:
        return TOKEN_NAMES.get(MechanismToken(int(value)), str(int(value)))
    except ValueError:
        return str(int(value))


def plot_witness_scene(label: dict[str, np.ndarray], candidate_idx: int, output_path: str | Path) -> None:
    """Plot one candidate and its most informative critical-agent evidence.

    The label cache does not store full HD-map geometry, so this diagnostic plot
    focuses on trajectories, min-burden safe responses, conflict-region centers,
    and witness token/burden values.  It is intended for selective inspection of
    the highest-signal scenes selected by ``06_diagnose_dataset.py``.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ego = label["cowp/candidates/trajectory"][candidate_idx]
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot(ego[:, 0], ego[:, 1], label=f"ego candidate {candidate_idx}")
    if "map/conflict_regions" in label and "map/conflict_region_valid" in label:
        valid_regions = label["map/conflict_region_valid"].astype(bool)
        regions = label["map/conflict_regions"][valid_regions]
        if len(regions):
            ax.scatter(regions[:, 1], regions[:, 2], marker="x", s=24, label="conflict regions")
    crit = label["cowp/critical/valid"].astype(bool)
    witness_rows: list[str] = []
    for a in np.where(crit)[0]:
        nat_valid = label["cowp/natural/valid"][a].astype(bool)
        if np.any(nat_valid):
            nat_scores = label["cowp/natural/weight"][a].copy()
            nat_scores[~nat_valid] = -1.0
            m = int(np.argmax(nat_scores))
            nat = label["cowp/natural/traj"][a, m]
            ax.plot(nat[:, 0], nat[:, 1], linestyle="--", label=f"critical {a} natural")
        if label["cowp/witness/exists"][candidate_idx, a]:
            token = _token_name(int(label["cowp/witness/token"][candidate_idx, a]))
            burden = float(label["cowp/witness/min_safe_burden"][candidate_idx, a])
            opr = float(label["cowp/witness/opr"][candidate_idx, a])
            witness_rows.append(f"a{a}: {token}, minB={burden:.2f}, OPR={opr:.2f}")
            resp_valid = label["cowp/response/valid"][candidate_idx, a].astype(bool)
            safe = label["cowp/response/is_safe"][candidate_idx, a].astype(bool)
            idxs = np.where(resp_valid & safe)[0]
            if len(idxs):
                burdens = label["cowp/response/burden_total"][candidate_idx, a, idxs]
                r = int(idxs[int(np.argmin(burdens))])
                resp = label["cowp/response/traj"][candidate_idx, a, r]
                ax.plot(resp[:, 0], resp[:, 1], linestyle=":", label=f"critical {a} min-safe response")
    ax.axis("equal")
    ax.legend(loc="best", fontsize=8)
    subtitle = " | ".join(witness_rows[:3]) if witness_rows else "no positive witness for selected candidate"
    ax.set_title(f"COWP witness inspection\n{subtitle}")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
