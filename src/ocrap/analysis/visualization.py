from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np

from ocrap.data.serialization import load_npz


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(np.asarray(x).item())
    except Exception:
        return default


def _safe_str(x: Any, default: str = "") -> str:
    try:
        return str(np.asarray(x).item())
    except Exception:
        return default


def _mpl():
    import matplotlib.pyplot as plt  # type: ignore
    from matplotlib.patches import Rectangle  # type: ignore
    from matplotlib.transforms import Affine2D  # type: ignore
    return plt, Rectangle, Affine2D


def _boxplot_with_labels(ax, data, labels, **kwargs):
    """Draw a boxplot across Matplotlib versions.

    Matplotlib versions before the tick_labels rename expect `labels=...`;
    newer versions accept `tick_labels=...`.  Try the new spelling first,
    then fall back only for that compatibility TypeError.
    """
    try:
        return ax.boxplot(data, tick_labels=labels, **kwargs)
    except TypeError as exc:
        if "tick_labels" not in str(exc):
            raise
        return ax.boxplot(data, labels=labels, **kwargs)


def _finite_arr(rows: list[dict[str, Any]], key: str) -> np.ndarray:
    a = np.asarray([r.get(key, np.nan) for r in rows], dtype=float)
    return a[np.isfinite(a)]


def plot_recoverability_story(rows: list[dict[str, Any]], out_dir: Path) -> str | None:
    if not rows:
        return None
    plt, _, _ = _mpl()
    r_dep = np.asarray([r["r_dep"] for r in rows], dtype=float)
    r_orc = np.asarray([r["r_orc"] for r in rows], dtype=float)
    gap = np.asarray([r["gap"] for r in rows], dtype=float)
    art = np.asarray([r["artifact"] for r in rows], dtype=bool)
    finite = np.isfinite(r_dep) & np.isfinite(r_orc) & np.isfinite(gap)
    r_dep, r_orc, gap, art = r_dep[finite], r_orc[finite], gap[finite], art[finite]
    if r_dep.size == 0:
        return None
    x_min = min(float(r_dep.min()) - 0.25, -0.5)
    x_max = max(float(r_dep.max()) + 0.25, 0.8)
    y_min = min(float(r_orc.min()) - 0.25, -0.5)
    y_max = max(float(r_orc.max()) + 0.25, 0.8)
    oracle_only = (r_dep < 0.0) & (r_orc >= 0.0)
    deployable = r_dep >= 0.0
    critical = r_orc < 0.0

    fig = plt.figure(figsize=(10.6, 6.8))
    ax = fig.add_subplot(111)
    # Regions are named in plain language for slides.
    ax.axvspan(x_min, 0, ymin=max(0, (0 - y_min) / max(y_max - y_min, 1e-8)), ymax=1, alpha=0.10, color="tab:red", label="Oracle-only trap: not deployably recoverable")
    ax.axvspan(0, x_max, alpha=0.06, color="tab:green", label="Deployable-safe side")
    ax.axhspan(y_min, 0, alpha=0.06, color="0.2", label="No branch-wise recovery")
    ax.scatter(r_dep[~art], r_orc[~art], s=10, alpha=0.34, c="tab:blue", edgecolors="none", label="ordinary candidates")
    if art.any():
        ax.scatter(r_dep[art], r_orc[art], s=22, alpha=0.82, c="tab:orange", edgecolors="k", linewidths=0.15, label="oracle-artifact candidates")
    ax.axvline(0, color="k", lw=1.1)
    ax.axhline(0, color="k", lw=1.1)
    lo = min(x_min, y_min)
    hi = max(x_max, y_max)
    ax.plot([lo, hi], [lo, hi], ls="--", color="0.35", lw=1.0, label="oracle score = deployable score")
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel("Deployable recovery score after executing the prefix\n(higher is safer; 0 means barely recoverable)")
    ax.set_ylabel("Hindsight/oracle recovery score\n(higher is safer if the hidden future were known)")
    ax.set_title("Core motivation: hindsight recovery can overestimate what the vehicle can safely deploy")
    txt = (
        f"Oracle-only traps: {oracle_only.mean()*100:.1f}%\n"
        f"Deployable-safe candidates: {deployable.mean()*100:.1f}%\n"
        f"No branch-wise recovery: {critical.mean()*100:.1f}%\n"
        f"Median oracle-to-deployable gap: {np.median(gap):.2f}"
    )
    ax.text(0.02, 0.98, txt, transform=ax.transAxes, va="top", ha="left", fontsize=10,
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="0.75", alpha=0.94))
    if oracle_only.any():
        idx = np.argmax(gap * oracle_only)
        xy = (float(r_dep[idx]), float(r_orc[idx]))
    else:
        xy = (min(-0.4, float(np.percentile(r_dep, 5))), max(0.2, float(np.percentile(r_orc, 70))))
    ax.annotate("Oracle-only trap:\nrecovery depends on hidden branch identity",
                xy=xy,
                xycoords="data", xytext=(0.43, 0.22), textcoords="axes fraction",
                arrowprops=dict(arrowstyle="->", lw=1.2), fontsize=10,
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.8", alpha=0.92))
    ax.legend(loc="lower right", fontsize=8.5, framealpha=0.95)
    ax.grid(True, alpha=0.22)
    p = out_dir / "story_oracle_vs_deployable.png"
    fig.savefig(p, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return str(p)


def plot_criticality_ladder(rows: list[dict[str, Any]], out_dir: Path) -> str | None:
    if not rows:
        return None
    plt, _, _ = _mpl()
    names = [
        "Clearly safe\n(high recovery headroom)",
        "Recoverable but tight\n(low/mixed headroom)",
        "Oracle-only trap\n(not deployable online)",
        "Critical\n(no recovery)",
    ]
    counts = np.zeros((4, 2), dtype=int)  # non-art, artifact
    for r in rows:
        if r["r_dep"] >= 0.50 and r["gap"] < 0.25 and not r["artifact"]:
            b = 0
        elif r["r_dep"] >= 0.0:
            b = 1
        elif r["r_orc"] >= 0.0:
            b = 2
        else:
            b = 3
        counts[b, int(bool(r["artifact"]))] += 1
    fig = plt.figure(figsize=(10.8, 5.3))
    ax = fig.add_subplot(111)
    x = np.arange(len(names))
    ax.bar(x, counts[:, 0], label="ordinary candidates", color="tab:blue", alpha=0.76)
    ax.bar(x, counts[:, 1], bottom=counts[:, 0], label="oracle-artifact candidates", color="tab:orange", alpha=0.88)
    total = max(int(counts.sum()), 1)
    ymax = max(int(counts.sum(axis=1).max()), 1)
    for i in range(len(names)):
        c = int(counts[i].sum())
        ax.text(i, c + ymax * 0.025, f"{c:,}\n{c/total*100:.1f}%", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel("Number of candidate prefixes")
    ax.set_title("Dataset difficulty ladder: from easy recovery to oracle-only traps")
    ax.set_ylim(0, ymax * 1.18)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="upper right")
    p = out_dir / "story_criticality_ladder.png"
    fig.savefig(p, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return str(p)


def plot_regime_breakdown(rows: list[dict[str, Any]], out_dir: Path) -> str | None:
    if not rows:
        return None
    plt, _, _ = _mpl()
    keys = [
        ("normal", "Normal"),
        ("low_headroom", "Low-headroom"),
        ("near_contact", "Near-contact"),
        ("post_contact", "Post-contact"),
        ("occluded", "Occluded"),
        ("artifact", "Oracle artifact"),
    ]
    vals = [100.0 * np.mean([bool(r.get(k, False)) for r in rows]) for k, _ in keys]
    fig = plt.figure(figsize=(9.8, 4.8))
    ax = fig.add_subplot(111)
    x = np.arange(len(keys))
    ax.bar(x, vals, color=["tab:green", "tab:olive", "tab:red", "tab:purple", "tab:gray", "tab:orange"], alpha=0.84)
    for i, v in enumerate(vals):
        ax.text(i, v + max(1.2, max(vals) * 0.02), f"{v:.1f}%", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels([lab for _, lab in keys], rotation=15, ha="right")
    ax.set_ylabel("Fraction of candidate prefixes [%]")
    ax.set_title("What scenario regimes are represented in this OC-RAP dataset?")
    ax.set_ylim(0, max(100.0, max(vals) + 8.0))
    ax.grid(axis="y", alpha=0.25)
    p = out_dir / "story_regime_breakdown.png"
    fig.savefig(p, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return str(p)


def plot_gap_by_category(rows: list[dict[str, Any]], out_dir: Path) -> str | None:
    if not rows:
        return None
    plt, _, _ = _mpl()
    cats = [
        ("normal", "Normal"),
        ("low_headroom", "Low-headroom"),
        ("near_contact", "Near-contact"),
        ("post_contact", "Post-contact"),
        ("artifact", "Oracle artifact"),
    ]
    data, labels = [], []
    for k, lab in cats:
        vals = [float(r.get("gap", 0.0)) for r in rows if bool(r.get(k, False)) and np.isfinite(float(r.get("gap", 0.0)))]
        if vals:
            data.append(vals)
            labels.append(lab)
    if not data:
        return None
    fig = plt.figure(figsize=(9.8, 4.8))
    ax = fig.add_subplot(111)
    _boxplot_with_labels(ax, data, labels, showfliers=False)
    ax.set_ylabel("Oracle-to-deployable recovery gap")
    ax.set_title("Where is hindsight recovery most misleading?")
    ax.grid(axis="y", alpha=0.25)
    p = out_dir / "story_gap_by_category.png"
    fig.savefig(p, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return str(p)


def _collect_xy(a: np.ndarray) -> np.ndarray:
    arr = np.asarray(a, dtype=float)
    if arr.size == 0:
        return np.zeros((0, 2), dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    else:
        arr = arr.reshape(-1, arr.shape[-1])
    if arr.shape[-1] < 2:
        return np.zeros((0, 2), dtype=float)
    pts = arr[:, :2]
    mask = np.isfinite(pts[:, 0]) & np.isfinite(pts[:, 1])
    return pts[mask]


def _robust_view_limits(hist: np.ndarray, valid: np.ndarray, prefix: np.ndarray, route: np.ndarray, *, radius: float = 90.0):
    pts_list = []
    if prefix.ndim >= 2 and prefix.shape[0] > 0:
        pts_list.append(_collect_xy(prefix[:, :2]))
    if hist.ndim == 3 and valid.ndim == 2 and hist.shape[:2] == valid.shape:
        cur = hist[-1, valid[-1].astype(bool), :2]
        # Keep only reasonably local agents, otherwise global map artifacts can flatten the plot.
        if cur.size:
            d = np.linalg.norm(cur, axis=1)
            pts_list.append(cur[d <= max(radius * 1.3, 120.0)])
        ev = valid[:, 0].astype(bool) if valid.shape[1] > 0 else np.zeros(hist.shape[0], dtype=bool)
        if ev.any():
            pts_list.append(hist[ev, 0, :2])
    if route.ndim >= 2 and route.shape[0] > 0:
        rpts = _collect_xy(route[:, :2])
        if rpts.size:
            d = np.linalg.norm(rpts, axis=1)
            # Include route only when it is in the same local coordinate frame.
            local = rpts[d <= max(radius * 2.0, 160.0)]
            if local.size:
                pts_list.append(local)
    pts = np.concatenate([p for p in pts_list if p.size], axis=0) if any(p.size for p in pts_list) else np.zeros((0, 2))
    if pts.size == 0:
        return (-radius, radius), (-radius, radius)
    cx, cy = np.median(pts[:, 0]), np.median(pts[:, 1])
    x0, x1 = np.percentile(pts[:, 0], [2, 98])
    y0, y1 = np.percentile(pts[:, 1], [2, 98])
    half = max(20.0, min(radius, max(x1 - x0, y1 - y0) / 2 + 14.0))
    return (float(cx - half), float(cx + half)), (float(cy - half), float(cy + half))


def _segment_in_view(pts: np.ndarray, xlim, ylim, margin: float = 8.0) -> bool:
    if pts.shape[0] < 2:
        return False
    x0, x1 = xlim[0] - margin, xlim[1] + margin
    y0, y1 = ylim[0] - margin, ylim[1] + margin
    m = (pts[:, 0] >= x0) & (pts[:, 0] <= x1) & (pts[:, 1] >= y0) & (pts[:, 1] <= y1)
    return bool(m.any())


def _draw_vehicle(ax, x: float, y: float, heading: float, length: float, width: float, *, color: str, alpha: float, label: str | None = None, zorder: int = 6):
    _, Rectangle, Affine2D = _mpl()
    length = float(length) if np.isfinite(length) and length > 0 else 4.8
    width = float(width) if np.isfinite(width) and width > 0 else 2.0
    trans = Affine2D().rotate_around(x, y, heading) + ax.transData
    patch = Rectangle((x - length / 2, y - width / 2), length, width, transform=trans,
                      fc=color, ec="k", lw=0.8, alpha=alpha, label=label, zorder=zorder)
    ax.add_patch(patch)
    ax.plot([x, x + 0.55 * length * math.cos(heading)], [y, y + 0.55 * length * math.sin(heading)], color="k", lw=0.9, alpha=alpha, zorder=zorder + 1)


def plot_sample_scene(sample_path: str | Path, out_dir: Path, *, name: str = "toy_scene") -> str | None:
    plt, _, _ = _mpl()
    d = load_npz(sample_path)
    hist = np.asarray(d.get("agent_history", np.zeros((0, 0, 16))), dtype=float)
    valid = np.asarray(d.get("agent_valid", np.zeros(hist.shape[:2] if hist.ndim >= 2 else (0, 0))), dtype=bool)
    if valid.ndim == 2 and hist.ndim == 3 and valid.shape != hist.shape[:2] and valid.T.shape == hist.shape[:2]:
        valid = valid.T
    prefix = np.asarray(d.get("prefix_states", np.zeros((0, 9))), dtype=float)
    maps = np.asarray(d.get("map_polylines", np.zeros((0, 0, 2))), dtype=float)
    map_valid = np.asarray(d.get("map_valid", np.zeros(maps.shape[:2] if maps.ndim >= 2 else (0, 0))), dtype=bool)
    route = np.asarray(d.get("route", np.zeros((0, 2))), dtype=float)
    scene = _safe_str(d.get("scene_id", ""))
    t = int(round(_safe_float(d.get("time_index", -1), -1)))
    cand = int(round(_safe_float(d.get("candidate_index", -1), -1)))
    macro = _safe_str(d.get("prefix_macro_name", ""))
    r_dep = _safe_float(d.get("r_dep_star", 0.0))
    r_orc = _safe_float(d.get("r_orc_star", 0.0))
    gap = _safe_float(d.get("oracle_gap_star", r_orc - r_dep))
    art = bool(round(_safe_float(d.get("i_art_star", 0.0))))

    xlim, ylim = _robust_view_limits(hist, valid, prefix, route)
    fig = plt.figure(figsize=(9.0, 8.2))
    ax = fig.add_subplot(111)

    if maps.ndim >= 3:
        for i in range(min(maps.shape[0], 320)):
            mv = map_valid[i] if map_valid.ndim >= 2 and i < map_valid.shape[0] else np.ones(maps.shape[1], dtype=bool)
            pts = maps[i, mv]
            if pts.shape[0] >= 2 and _segment_in_view(pts[:, :2], xlim, ylim):
                ax.plot(pts[:, 0], pts[:, 1], color="0.78", lw=0.65, alpha=0.82, zorder=0)
    if route.ndim >= 2 and route.shape[0] >= 2 and _segment_in_view(route[:, :2], xlim, ylim):
        ax.plot(route[:, 0], route[:, 1], color="tab:green", lw=2.2, alpha=0.75, label="Reference route", zorder=1)

    n_drawn = 0
    if hist.ndim >= 3 and valid.ndim == 2 and hist.shape[:2] == valid.shape and hist.shape[0] > 0:
        T, A = hist.shape[:2]
        # Historical traces.
        for a in range(min(A, 24)):
            av = valid[:, a].astype(bool)
            if av.sum() < 2:
                continue
            pts = hist[av, a, :2]
            if not _segment_in_view(pts, xlim, ylim, margin=0):
                continue
            is_ego = a == 0
            ax.plot(pts[:, 0], pts[:, 1], color=("tab:red" if is_ego else "tab:blue"), lw=(1.5 if is_ego else 0.8), ls=("--" if is_ego else "-"), alpha=(0.85 if is_ego else 0.22), zorder=(3 if is_ego else 2), label=("Ego history" if is_ego else None))
        current = hist[-1]
        current_valid = valid[-1].astype(bool)
        for a in range(min(A, 40)):
            if not current_valid[a]:
                continue
            s = current[a]
            if s.shape[0] < 12:
                continue
            if not (xlim[0] - 10 <= s[0] <= xlim[1] + 10 and ylim[0] - 10 <= s[1] <= ylim[1] + 10):
                continue
            is_ego = a == 0
            color = "tab:red" if is_ego else "tab:blue"
            label = "Ego at decision time" if is_ego else ("Other traffic participants" if n_drawn == 1 else None)
            heading = float(s[7]) if np.isfinite(s[7]) else 0.0
            _draw_vehicle(ax, float(s[0]), float(s[1]), heading, float(s[10]) if s.size > 10 else 4.8, float(s[11]) if s.size > 11 else 2.0, color=color, alpha=0.95 if is_ego else 0.50, label=label, zorder=8 if is_ego else 6)
            n_drawn += 1
    ego_state = np.asarray(d.get("ego_state", np.zeros((9,))), dtype=float).reshape(-1)
    if n_drawn == 0 and ego_state.size >= 9:
        _draw_vehicle(ax, float(ego_state[0]), float(ego_state[1]), float(ego_state[4]), float(ego_state[7]), float(ego_state[8]), color="tab:red", alpha=0.95, label="Ego at decision time", zorder=8)
    if prefix.ndim >= 2 and prefix.shape[0] >= 2:
        ax.plot(prefix[:, 0], prefix[:, 1], color="k", lw=2.6, marker="o", ms=2.4, label=f"Candidate prefix ({macro})", zorder=9)
        if prefix.shape[1] >= 9:
            _draw_vehicle(ax, float(prefix[-1, 0]), float(prefix[-1, 1]), float(prefix[-1, 4]), float(prefix[-1, 7]), float(prefix[-1, 8]), color="tab:purple", alpha=0.60, label="Ego after executing prefix", zorder=10)

    ax.set_xlim(*xlim); ax.set_ylim(*ylim)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.18)
    ax.set_xlabel("Local x position around ego [m]")
    ax.set_ylabel("Local y position around ego [m]")
    headline = "Toy example: candidate prefix under hidden-future ambiguity"
    if art and r_dep < 0 <= r_orc:
        headline = "Toy example: oracle-only trap — hindsight recovery is not deployable online"
    ax.set_title(headline)
    subtitle = (
        f"scene={scene}, t={t}, candidate={cand}, macro={macro}\n"
        f"deployable recovery={r_dep:.2f}, oracle recovery={r_orc:.2f}, oracle-to-deployable gap={gap:.2f}, artifact={art}"
    )
    ax.text(0.02, 0.02, subtitle, transform=ax.transAxes, ha="left", va="bottom", fontsize=8.7,
            bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="0.75", alpha=0.94))
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    if by_label:
        ax.legend(by_label.values(), by_label.keys(), loc="upper right", fontsize=8, framealpha=0.92)
    p = out_dir / f"{name}_topdown.png"
    fig.savefig(p, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return str(p)


def plot_sample_ambiguity(sample_path: str | Path, out_dir: Path, *, name: str = "toy_scene") -> str | None:
    plt, _, _ = _mpl()
    d = load_npz(sample_path)
    c = np.asarray(d.get("c_star", np.zeros((0, 0))), dtype=float)
    m = np.asarray(d.get("m_star", np.zeros((0, 0))), dtype=float)
    rv = np.asarray(d.get("root_valid", np.ones(c.shape[0] if c.ndim >= 2 else 0)), dtype=bool)
    ov = np.asarray(d.get("option_valid", np.ones(m.shape[1] if m.ndim >= 2 else 0)), dtype=bool)
    if c.ndim != 2 or c.size == 0:
        return None
    fig = plt.figure(figsize=(12.2, 4.9))
    ax1 = fig.add_subplot(121)
    im = ax1.imshow(c, vmin=0, vmax=1, cmap="viridis")
    ax1.set_title("Post-prefix observation similarity between hidden futures")
    ax1.set_xlabel("Hidden-future root j"); ax1.set_ylabel("Hidden-future root i")
    fig.colorbar(im, ax=ax1, fraction=0.046, pad=0.04, label="similarity: 1 means indistinguishable")
    ax2 = fig.add_subplot(122)
    mm = np.asarray(m, dtype=float).copy()
    if m.ndim == 2:
        invalid = ~(rv[:, None] & ov[None, :])
        mm[invalid] = np.nan
    im2 = ax2.imshow(mm, cmap="coolwarm", aspect="auto")
    ax2.set_title("Recovery margin for each hidden future and recovery option")
    ax2.set_xlabel("Recovery option index"); ax2.set_ylabel("Hidden-future root")
    fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04, label="signed margin: >0 recoverable, <0 unsafe")
    fig.suptitle("Why branch-wise recovery can be misleading: ambiguous observations + one shared recovery decision", y=1.03, fontsize=12)
    p = out_dir / f"{name}_ambiguity_matrix.png"
    fig.savefig(p, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return str(p)


def _diverse_candidates(rows: list[dict[str, Any]]) -> list[tuple[str, list[dict[str, Any]]]]:
    return [
        ("oracle_trap", [r for r in rows if r.get("r_orc", -1) >= 0 and r.get("r_dep", 1) < 0]),
        ("post_contact", [r for r in rows if r.get("post_contact", False)]),
        ("near_contact", [r for r in rows if r.get("near_contact", False) or r.get("low_headroom", False)]),
        ("normal", [r for r in rows if r.get("normal", False) and r.get("r_dep", 0) >= 0.5 and r.get("gap", 9) < 0.25]),
    ]


def write_toy_gallery(rows: list[dict[str, Any]], out_dir: Path, *, max_examples: int = 4) -> list[str]:
    paths: list[str] = []
    if not rows:
        return paths
    selected: list[tuple[str, dict[str, Any]]] = []
    seen_scene_time: set[tuple[Any, Any]] = set()
    for label, bucket in _diverse_candidates(rows):
        bucket = sorted(bucket, key=lambda r: float(r.get("gap", 0.0)), reverse=True)
        for r in bucket:
            key = (r.get("scene_id"), r.get("time_index"))
            if key in seen_scene_time or not r.get("path"):
                continue
            selected.append((label, r)); seen_scene_time.add(key)
            break
    if len(selected) < max_examples:
        fallback = sorted(rows, key=lambda r: float(r.get("gap", 0.0)), reverse=True)
        for r in fallback:
            key = (r.get("scene_id"), r.get("time_index"))
            if key not in seen_scene_time and r.get("path"):
                selected.append(("high_gap", r)); seen_scene_time.add(key)
            if len(selected) >= max_examples:
                break
    for k, (label, r) in enumerate(selected[:max_examples]):
        name = f"toy_{k:02d}_{label}_t{r.get('time_index', 'x')}_c{r.get('candidate_index', 'x')}"
        for p in (plot_sample_scene(r["path"], out_dir, name=name), plot_sample_ambiguity(r["path"], out_dir, name=name)):
            if p:
                paths.append(p)
    return paths
