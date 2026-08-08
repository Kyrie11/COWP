from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np


def _restore_key(key: str) -> str:
    return key.replace("__", "/")


def _canonicalize_state_aliases(data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Normalize WOMD tf.Example state keys to the model-facing ``state/*`` namespace.

    Tensor caches produced by this project store raw tf.Example tensors as
    ``womd/state/...`` while some older/debug caches store already-shaped tensors
    as ``state/...``. Mixing those files in one DataLoader batch is dangerous if
    collation intersects keys: the effective state input can disappear. We
    canonicalize to ``state/...`` immediately after NPZ loading so equivalent
    caches expose the same keys without duplicating tensors.
    """
    for key in list(data.keys()):
        if not key.startswith("womd/state/"):
            continue
        canon = "state/" + key[len("womd/state/") :]
        if canon not in data:
            data[canon] = data[key]
        del data[key]
    return data


def _has_model_state(data: dict[str, np.ndarray]) -> bool:
    """Whether a sample contains enough encoder input for the COWP graph."""
    if "state/history" in data or "state/all" in data:
        return True
    return _first_array(data, ("state/past/x",)) is not None and _first_array(data, ("state/current/x",)) is not None


def _missing_required_for_stage(data: dict[str, np.ndarray], stage: str | None) -> list[str]:
    """Return human-readable missing fields for a training stage.

    Invalid/partial cache files should be skipped before batching. Otherwise one
    incomplete item can remove required tensors from the whole batch and cause a
    late ``KeyError`` inside the compiled model.
    """
    stage = stage or "all"
    missing: list[str] = []
    if not _has_model_state(data):
        missing.append("state/history or state/all or state/{past,current}/x")
    for key in ("cowp/critical/track_index", "cowp/critical/valid"):
        if key not in data:
            missing.append(key)
    if stage in ("representation", "natural", "all"):
        for key in (
            "cowp/natural/traj",
            "cowp/natural/valid",
            "cowp/natural/weight",
            "cowp/natural/source",
            "cowp/natural/priority_preserved",
        ):
            if key not in data:
                missing.append(key)
    if stage in ("response", "witness", "planner", "planner_eval", "all"):
        for key in ("cowp/candidates/trajectory", "cowp/candidates/macro_type", "cowp/candidates/valid"):
            if key not in data:
                missing.append(key)
    if stage in ("response", "all"):
        for key in (
            "cowp/response/valid",
            "cowp/response/is_safe",
            "cowp/response/is_low_burden",
            "cowp/response/burden_total",
        ):
            if key not in data:
                missing.append(key)
    if stage in ("witness", "planner", "planner_eval", "all"):
        for key in (
            "cowp/witness/exists",
            "cowp/witness/token",
            "cowp/witness/burden_total",
            "cowp/witness/conflict_interval",
            "cowp/witness/opr",
            "cowp/witness/c_i",
        ):
            if key not in data:
                missing.append(key)
    if stage in ("planner", "planner_eval", "all"):
        for key in ("cowp/candidates/noncoercive_feasible", "cowp/candidates/false_safe"):
            if key not in data:
                missing.append(key)
    if stage == "planner_eval":
        # RIOT is a root-indexed transport claim.  Evaluation must fail rather
        # than silently omit the direct mechanism metric when the augmented
        # response-root labels are absent.
        for key in (
            "cowp/response/valid",
            "cowp/response/is_safe",
            "cowp/response/is_low_burden",
            "cowp/transport/response_root_index",
            "cowp/transport/mode_valid",
            "cowp/transport/mode_conflict",
        ):
            if key not in data:
                missing.append(key)
    return missing


def _first_array(data: dict[str, np.ndarray], names: tuple[str, ...]) -> np.ndarray | None:
    for name in names:
        arr = data.get(name)
        if arr is not None:
            return np.asarray(arr)
    return None


def _infer_agent_count(data: dict[str, np.ndarray]) -> int | None:
    """Infer the number of model-visible WOMD agents in one cache item.

    Tensor-cache files may store temporal WOMD fields flattened as [N*T].  Agent
    count must therefore be inferred primarily from current/id fields ([N]), not
    from past arrays ([N*T]).
    """
    hist = _first_array(data, ("state/history", "womd/state/history"))
    if hist is not None and hist.ndim >= 2:
        return int(hist.shape[0])

    for key in (
        "state/current/x",
        "womd/state/current/x",
        "state/current/valid",
        "womd/state/current/valid",
        "state/id",
        "womd/state/id",
        "state/agent_valid",
        "womd/state/agent_valid",
        "state/is_sdc",
        "womd/state/is_sdc",
    ):
        arr = data.get(key)
        if arr is None:
            continue
        arr = np.asarray(arr)
        if arr.ndim == 0:
            continue
        return int(arr.shape[0])

    # Last-resort temporal-only inference.  WOMD past has 10 steps; if the flat
    # size is divisible by 10, recover N instead of returning N*T.
    for key in ("state/past/x", "womd/state/past/x"):
        arr = data.get(key)
        if arr is None:
            continue
        arr = np.asarray(arr)
        if arr.ndim >= 2:
            return int(arr.shape[0])
        if arr.ndim == 1 and arr.size % 10 == 0:
            return int(arr.size // 10)
        if arr.ndim == 1:
            return int(arr.size)
    return None


def _infer_agent_visible_mask(data: dict[str, np.ndarray], num_agents: int | None = None) -> np.ndarray | None:
    """Infer which agent rows are visible to the model encoder.

    This mirrors ``build_agent_history_from_womd`` as closely as possible without
    importing torch: current-valid is preferred; otherwise any known agent row is
    considered visible.  The SDC row is always kept visible.
    """
    if num_agents is None:
        num_agents = _infer_agent_count(data)
    if num_agents is None or num_agents <= 0:
        return None
    visible = np.ones(int(num_agents), dtype=bool)
    cur_valid = _first_array(data, ("state/current/valid", "womd/state/current/valid", "state/agent_valid", "womd/state/agent_valid"))
    if cur_valid is not None and cur_valid.size >= num_agents:
        visible = np.asarray(cur_valid).reshape(-1)[:num_agents].astype(float) > 0.5
    is_sdc = _first_array(data, ("state/is_sdc", "womd/state/is_sdc"))
    if is_sdc is not None and is_sdc.size >= num_agents:
        visible = visible | (np.asarray(is_sdc).reshape(-1)[:num_agents].astype(float) > 0.5)
    elif num_agents > 0:
        visible[0] = True
    return visible


def _critical_gather_index(data: dict[str, np.ndarray]) -> tuple[str, np.ndarray] | tuple[None, None]:
    """Return the agent-row index used by the model.

    ``cowp/critical/track_index`` is the original Scenario proto track index.
    In real WOMD tf.Example tensors the model-visible agent row should be found
    by matching the object id.  New tensor caches therefore store
    ``cowp/critical/input_index`` and the model/dataset should prefer it.
    """
    if "cowp/critical/input_index" in data:
        return "cowp/critical/input_index", np.asarray(data["cowp/critical/input_index"]).astype(np.int64, copy=False)
    if "cowp/critical/track_index" in data:
        return "cowp/critical/track_index", np.asarray(data["cowp/critical/track_index"]).astype(np.int64, copy=False)
    return None, None


def align_critical_agents_to_womd_input(data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Map Scenario critical tracks to WOMD tf.Example/model input rows.

    Label construction is performed from Scenario protos, where critical agents
    are naturally indexed by Scenario track index.  The training model reads raw
    WOMD tf.Example state tensors whose first dimension is an input row and may
    be capped/reordered.  Matching by object id avoids supervising/gathering the
    wrong agent.  This function is intentionally conservative: if object ids are
    unavailable it falls back to the legacy track-index convention and runtime
    masks still prevent crashes.
    """
    track_index = data.get("cowp/critical/track_index")
    if track_index is None:
        return data
    track_index = np.asarray(track_index).astype(np.int64, copy=False).reshape(-1)
    input_index = np.array(track_index, dtype=np.int64, copy=True)
    mapped_by_id = np.zeros_like(track_index, dtype=bool)

    track_id = data.get("cowp/critical/track_id")
    state_id = _first_array(data, ("state/id", "womd/state/id"))
    n = _infer_agent_count(data)
    if track_id is not None and state_id is not None:
        ids = np.asarray(state_id).reshape(-1)
        if n is not None:
            ids = ids[: int(n)]
        id_to_row: dict[int, int] = {}
        visible = _infer_agent_visible_mask(data, len(ids))
        for row, tid in enumerate(ids):
            try:
                tid_int = int(tid)
            except Exception:
                continue
            # Ignore repeated padded ids when the corresponding row is invalid.
            if visible is not None and row < len(visible) and not bool(visible[row]) and tid_int in id_to_row:
                continue
            if tid_int not in id_to_row:
                id_to_row[tid_int] = row
        tids = np.asarray(track_id).reshape(-1)
        if tids.size == track_index.size:
            input_index = np.full_like(track_index, -1, dtype=np.int64)
            for a, tid in enumerate(tids):
                try:
                    row = id_to_row.get(int(tid), -1)
                except Exception:
                    row = -1
                input_index[a] = int(row)
                mapped_by_id[a] = row >= 0
            data["cowp/critical/track_index_original"] = np.array(track_index, dtype=np.int64, copy=True)
            data["cowp/critical/mapped_by_id"] = mapped_by_id.astype(bool)

    data["cowp/critical/input_index"] = input_index.astype(np.int64)
    return data


def mask_out_of_range_critical_agents(data: dict[str, np.ndarray], num_agents: int | None = None) -> dict[str, np.ndarray]:
    """Mask critical slots whose model input row is invisible.

    This prevents both hard crashes (e.g. gather index 149 for a 128-agent tensor)
    and silent wrong supervision.  New caches should first call
    ``align_critical_agents_to_womd_input`` so the check is performed on
    ``cowp/critical/input_index`` rather than the original Scenario track index.
    """
    if num_agents is None:
        num_agents = _infer_agent_count(data)
    idx_key, idx = _critical_gather_index(data)
    if num_agents is None or idx is None:
        return data

    idx = np.asarray(idx).astype(np.int64, copy=False)
    if idx.ndim != 1:
        return data
    in_range = (idx >= 0) & (idx < int(num_agents))
    agent_visible = _infer_agent_visible_mask(data, num_agents)
    visible = in_range.copy()
    if agent_visible is not None and len(agent_visible) > 0:
        safe_idx = np.clip(idx, 0, len(agent_visible) - 1)
        visible = visible & agent_visible[safe_idx]
    old_valid = np.asarray(data.get("cowp/critical/valid", np.ones_like(visible, dtype=bool))).astype(bool, copy=False)
    new_valid = old_valid & visible
    data["cowp/critical/input_visible"] = visible.astype(bool)
    if np.array_equal(new_valid, old_valid):
        return data

    data["cowp/critical/valid"] = new_valid.astype(bool)
    bad = ~new_valid
    if not bad.any():
        return data

    # Natural labels are [A,...].
    for key in list(data.keys()):
        arr0 = np.asarray(data[key])
        if key.startswith("cowp/natural/") and arr0.ndim >= 1 and arr0.shape[0] == len(idx):
            arr = np.array(arr0, copy=True)
            if key.endswith("/valid") or key.endswith("/priority_preserved"):
                arr[bad] = False
            else:
                arr[bad] = 0
            data[key] = arr

    # Response / witness pair labels are [K,A,...].
    for key in list(data.keys()):
        arr0 = np.asarray(data[key])
        if key.startswith("cowp/response/") and arr0.ndim >= 2 and arr0.shape[1] == len(idx):
            arr = np.array(arr0, copy=True)
            if key.endswith("/valid") or key.endswith("/is_safe") or key.endswith("/is_low_burden"):
                arr[:, bad] = False
            else:
                arr[:, bad] = 0
            data[key] = arr
        elif key.startswith("cowp/witness/") and arr0.ndim >= 2 and arr0.shape[1] == len(idx):
            arr = np.array(arr0, copy=True)
            if key.endswith("/exists"):
                arr[:, bad] = False
            elif key.endswith("/opr"):
                arr[:, bad] = 1.0
            else:
                arr[:, bad] = 0
            data[key] = arr
        elif key in {"cowp/witness/critical_agent_track_index"} and arr0.ndim == 1 and arr0.shape[0] == len(idx):
            arr = np.array(arr0, copy=True)
            arr[bad] = -1
            data[key] = arr
    return data


def _wanted_keys_for_stage(
    stage: str | None,
    *,
    include_response_traj: bool = True,
    include_response_components: bool = True,
    include_waymax_outcomes: bool = False,
) -> set[str] | None:
    """Return the minimal NPZ keys needed by a training/eval stage.

    ``stage=all`` should not mean "read the entire NPZ".  It is the union of
    model heads, while dense optional tensors remain controlled by flags.  For
    ``planner_eval`` we also load attached scalar Waymax candidate outcomes when
    present, so learned-offline evaluation can report replay outcome metrics.
    """
    if stage is None:
        return None
    stage = str(stage)
    wanted: set[str] = {
        "womd/state/",
        "state/",
        "cowp/critical/",
        "map/conflict_regions",
        "map/conflict_region_valid",
    }

    if stage in ("representation", "natural", "all"):
        wanted.add("cowp/natural/")

    # Response-set supervision is required not only by the dedicated response
    # stage, but also by witness/planner training because the mechanistic
    # Set-Transport certificate consumes the compact response bank.  v8 only
    # loaded these keys for stage=response/all, so response_aux and
    # set_transport/response silently stayed at zero during planner training.
    if stage in ("response", "witness", "planner", "planner_eval", "all"):
        wanted.update({
            "cowp/candidates/trajectory",
            "cowp/candidates/macro_type",
            "cowp/candidates/valid",
            "cowp/response/valid",
            "cowp/response/source",
            "cowp/response/is_safe",
            "cowp/response/is_low_burden",
            "cowp/response/burden_total",
            "cowp/transport/",
        })
        # Learned-offline planner evaluation needs compact response/root labels
        # for direct RIOT mechanism metrics, but never needs the dense response
        # trajectory or component tensors.
        if stage != "planner_eval" and include_response_components:
            wanted.add("cowp/response/burden_components")
        if stage != "planner_eval" and include_response_traj:
            wanted.add("cowp/response/traj")

    if stage in ("witness", "planner", "planner_eval", "all"):
        # Natural trajectories are modest compared with the candidate-conditioned
        # response bank and are needed to align unordered predicted natural modes
        # with explicit per-mode transport labels.
        wanted.add("cowp/natural/")
        wanted.update({
            "cowp/candidates/trajectory",
            "cowp/candidates/macro_type",
            "cowp/candidates/valid",
            "cowp/witness/",
        })

    # Candidate-level NCF/false-safe supervision is consumed directly by the
    # Set-Transport candidate-budget objective.  The dedicated transport stage
    # is named ``witness`` in the training CLI, so these labels must be loaded
    # there as well as for planner fine-tuning.  v11 omitted them for witness,
    # which made set_transport/candidate_budget identically zero.
    if stage in ("witness", "planner", "planner_eval", "all"):
        load_waymax = (stage == "planner_eval") or (stage in ("planner", "all") and include_waymax_outcomes)
        if load_waymax:
            wanted.update({
                "waymax/candidate_rollout_valid",
                "waymax/candidate_collision",
                "waymax/candidate_offroad",
                "waymax/candidate_log_divergence",
            })
        wanted.update({
            "cowp/candidates/conventional_safe",
            "cowp/candidates/false_safe",
            "cowp/candidates/noncoercive_feasible",
            "cowp/candidates/ego_utility_prior",
            "cowp/candidates/is_logged",
            "cowp/candidates/is_neutral",
            "cowp/natural/beta",
        })
    return wanted


def _key_allowed(key: str, wanted: set[str] | None) -> bool:
    if wanted is None:
        return True
    for item in wanted:
        if item.endswith("/") and key.startswith(item):
            return True
        if key == item:
            return True
    return False


class COWPNpzDataset:
    """NPZ dataset with optional transparent transport-label sidecars.

    A v9 overlay cache contains symlinks to the original large NPZ files and a
    hidden ``.transport_v9`` directory with small files that only store the new
    ``cowp/transport/*`` tensors.  Loading both sources here keeps every existing
    training/evaluation CLI unchanged while avoiding a full duplicate cache.
    """

    def __init__(self, cache_dir: str | Path, pattern: str = "*.npz"):
        self.cache_dir = Path(cache_dir)
        # Exclude hidden NPZ metadata files such as sampler-weight caches.
        # They share the .npz suffix but are not scenario samples and must never
        # enter dataset indexing, augmentation, signatures, or DDP sampling.
        self.paths = sorted(
            p for p in self.cache_dir.glob(pattern) if not p.name.startswith(".")
        )
        if not self.paths:
            raise FileNotFoundError(f"No cache npz files found in {self.cache_dir} matching {pattern}")
        # Overlay caches intentionally expose base scenario files as symlinks.
        # A directory listing can therefore look healthy even after its backing
        # cache was deleted: pathlib/glob still returns the broken link, while
        # np.load fails much later with a misleading FileNotFoundError.  Fail at
        # dataset construction with an actionable lineage diagnostic instead of
        # letting a random sample crash a long experiment.
        broken = [p for p in self.paths if p.is_symlink() and not p.exists()]
        if broken:
            examples = []
            for bp in broken[:5]:
                try:
                    target = bp.readlink()
                except OSError:
                    target = "<unreadable>"
                examples.append(f"{bp.name}->{target}")
            raise FileNotFoundError(
                f"Cache {self.cache_dir} contains {len(broken)} broken NPZ symlink(s); "
                f"examples={examples}. This usually means an overlay backing cache "
                "was deleted. Use the original base tensor cache, or rebase/rebuild "
                "the transport overlay; do not treat the visible symlink names as data files."
            )
        sidecar_name = ".transport_v9"
        summary = self.cache_dir / "transport_augmentation_summary.json"
        if summary.is_file():
            try:
                meta = json.loads(summary.read_text(encoding="utf-8"))
                if str(meta.get("storage_mode", "")) == "overlay" and meta.get("sidecar_subdir"):
                    sidecar_name = str(meta["sidecar_subdir"])
            except Exception:
                pass
        self.transport_sidecar_dir = self.cache_dir / sidecar_name

    def __len__(self) -> int:
        return len(self.paths)

    @staticmethod
    def _load_into(path: Path, out: dict[str, np.ndarray], wanted: set[str] | None) -> None:
        with np.load(path, allow_pickle=True) as data:
            for raw_key in data.files:
                key = _restore_key(raw_key)
                if _key_allowed(key, wanted):
                    out[key] = data[raw_key]

    @staticmethod
    def _wants_transport(wanted: set[str] | None) -> bool:
        if wanted is None:
            return True
        return any(
            item == "cowp/transport"
            or item == "cowp/transport/"
            or item.startswith("cowp/transport/")
            for item in wanted
        )

    def load(self, idx: int, wanted: set[str] | None = None) -> dict[str, np.ndarray]:
        path = self.paths[idx]
        out: dict[str, np.ndarray] = {}
        self._load_into(path, out, wanted)
        if self._wants_transport(wanted):
            sidecar = self.transport_sidecar_dir / path.name
            if sidecar.is_file():
                self._load_into(sidecar, out, wanted)
        _canonicalize_state_aliases(out)
        align_critical_agents_to_womd_input(out)
        return mask_out_of_range_critical_agents(out)

    def __getitem__(self, idx: int) -> dict[str, np.ndarray]:
        return self.load(idx, None)


class TorchCOWPDataset:
    def __init__(
        self,
        cache_dir: str | Path,
        pattern: str = "*.npz",
        stage: str | None = None,
        *,
        skip_invalid: bool = True,
        include_response_traj: bool = True,
        include_response_components: bool = True,
        include_waymax_outcomes: bool = False,
    ):
        self.base = COWPNpzDataset(cache_dir, pattern)
        self.stage = stage
        self._wanted = _wanted_keys_for_stage(
            stage,
            include_response_traj=include_response_traj,
            include_response_components=include_response_components,
            include_waymax_outcomes=include_waymax_outcomes,
        )
        self.skip_invalid = bool(skip_invalid)

    def __len__(self) -> int:
        return len(self.base)

    def _load_valid_np(self, idx: int) -> dict[str, np.ndarray]:
        last_missing: list[str] = []
        last_path: Path | None = None
        n = len(self.base)
        for off in range(n if self.skip_invalid else 1):
            j = (int(idx) + off) % n
            d = self.base.load(j, self._wanted)
            missing = _missing_required_for_stage(d, self.stage)
            if not missing:
                return d
            last_missing = missing
            last_path = self.base.paths[j]
            if not self.skip_invalid:
                break
        raise KeyError(f"No valid COWP sample found for stage={self.stage!r}; last_path={last_path}; missing={last_missing}")

    def __getitem__(self, idx: int):
        import torch

        d = self._load_valid_np(idx)
        out = {}
        for k, v in d.items():
            if not _key_allowed(k, self._wanted):
                continue
            arr = np.asarray(v)
            if arr.dtype.kind in "fiu" or arr.dtype == np.bool_:
                # np.load may return non-writable views; copying avoids rare pin-memory
                # and from_numpy warnings while keeping per-stage I/O bounded.
                if arr.dtype == np.float64:
                    arr = arr.astype(np.float32)
                elif not arr.flags.c_contiguous:
                    arr = np.ascontiguousarray(arr)
                out[k] = torch.from_numpy(arr)
        return out


def collate_torch(batch):
    import torch

    batch = [x for x in batch if isinstance(x, dict) and x]
    if not batch:
        return {}
    out = {}
    # Dataset-level validation makes stage-required tensors present for every
    # item. Intersecting keys is therefore safe and avoids fabricating labels.
    keys = set.intersection(*(set(x.keys()) for x in batch))
    optional_prefixes = ("scenario/", "dataset/", "womd/scenario/", "womd/roadgraph/", "roadgraph/")
    for k in keys:
        vals = [x[k] for x in batch]
        try:
            out[k] = torch.stack(vals, dim=0)
        except Exception as exc:
            if k.startswith(optional_prefixes):
                continue
            shapes = [tuple(v.shape) for v in vals if torch.is_tensor(v)]
            raise RuntimeError(f"Failed to collate tensor key {k!r}; shapes={shapes}") from exc
    return out
