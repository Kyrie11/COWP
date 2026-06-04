from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_witness_scene(label: dict[str, np.ndarray], candidate_idx: int, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ego = label["cowp/candidates/trajectory"][candidate_idx]
    plt.figure(figsize=(7, 7))
    plt.plot(ego[:, 0], ego[:, 1], label=f"ego candidate {candidate_idx}")
    crit = label["cowp/critical/valid"].astype(bool)
    for a in np.where(crit)[0]:
        nat_valid = label["cowp/natural/valid"][a].astype(bool)
        if np.any(nat_valid):
            nat = label["cowp/natural/traj"][a, np.where(nat_valid)[0][0]]
            plt.plot(nat[:, 0], nat[:, 1], linestyle="--", label=f"critical {a} natural")
        if label["cowp/witness/exists"][candidate_idx, a]:
            resp_valid = label["cowp/response/valid"][candidate_idx, a].astype(bool)
            safe = label["cowp/response/is_safe"][candidate_idx, a].astype(bool)
            idxs = np.where(resp_valid & safe)[0]
            if len(idxs):
                r = idxs[int(np.argmin(label["cowp/response/burden_total"][candidate_idx, a, idxs]))]
                resp = label["cowp/response/traj"][candidate_idx, a, r]
                plt.plot(resp[:, 0], resp[:, 1], linestyle=":", label=f"critical {a} min-safe response")
    plt.axis("equal")
    plt.legend(loc="best", fontsize=8)
    plt.title("COWP witness inspection")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()
