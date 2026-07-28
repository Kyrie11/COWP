from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch


_EPOCH_RE = re.compile(r"_epoch(-?\d+)\.pt$")
_KIND_PRIORITY = {"best": 0, "epoch": 1, "last": 2}


@dataclass(frozen=True)
class CheckpointResolution:
    checkpoint: str
    stage: str
    epoch: int
    next_epoch: int
    target_epochs: int
    complete: bool
    kind: str


def _torch_load_metadata(path: Path) -> dict[str, Any]:
    """Load a checkpoint on CPU while avoiding eager tensor copies when possible."""
    try:
        return torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    except (TypeError, RuntimeError):
        try:
            return torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            # Older supported PyTorch releases do not expose ``weights_only``.
            return torch.load(path, map_location="cpu")


def _checkpoint_stage(payload: dict[str, Any]) -> str | None:
    stage = payload.get("stage")
    if isinstance(stage, str) and stage:
        return stage
    cfg = payload.get("config")
    if isinstance(cfg, dict):
        train = cfg.get("train")
        if isinstance(train, dict):
            nested = train.get("stage")
            if isinstance(nested, str) and nested:
                return nested
    return None


def _checkpoint_epoch(payload: dict[str, Any]) -> int | None:
    value = payload.get("epoch")
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _kind(path: Path) -> str:
    if path.name.endswith("_last.pt"):
        return "last"
    if _EPOCH_RE.search(path.name):
        return "epoch"
    return "best"


def _candidate_paths(output_dir: Path, stage: str) -> list[Path]:
    prefix = f"cowp_{stage}"
    paths: list[Path] = []
    last = output_dir / f"{prefix}_last.pt"
    if last.is_file():
        paths.append(last)

    epoch_paths: list[tuple[int, Path]] = []
    for path in output_dir.glob(f"{prefix}_epoch*.pt"):
        match = _EPOCH_RE.search(path.name)
        if match:
            epoch_paths.append((int(match.group(1)), path))
    epoch_paths.sort(key=lambda item: (item[0], item[1].stat().st_mtime_ns), reverse=True)
    paths.extend(path for _, path in epoch_paths)

    best = output_dir / f"{prefix}_best.pt"
    if best.is_file():
        paths.append(best)
    return paths


def resolve_latest_checkpoint(
    output_dir: str | os.PathLike[str],
    stage: str,
    target_epochs: int,
) -> CheckpointResolution:
    """Resolve the newest valid same-stage checkpoint.

    Epochs in ``03_train.py`` are zero-based.  Therefore a checkpoint at epoch
    ``target_epochs - 1`` has completed the requested number of epochs.
    """
    root = Path(output_dir)
    target = max(int(target_epochs), 0)
    valid: list[tuple[int, int, int, Path, str]] = []
    seen_kinds: set[str] = set()

    # Loading every epoch checkpoint can be prohibitively expensive.  Inspect the
    # authoritative post-scheduler ``last`` file, the newest valid numbered file,
    # and ``best``.  If a numbered file is corrupt, continue descending until one
    # can be loaded.
    for path in _candidate_paths(root, stage):
        kind = _kind(path)
        if kind in seen_kinds:
            continue
        try:
            payload = _torch_load_metadata(path)
            if not isinstance(payload, dict):
                raise TypeError("checkpoint payload is not a dictionary")
            actual_stage = _checkpoint_stage(payload)
            epoch = _checkpoint_epoch(payload)
            if actual_stage != stage:
                raise ValueError(f"stage={actual_stage!r}, expected {stage!r}")
            if epoch is None or epoch < -1:
                raise ValueError(f"invalid epoch={epoch!r}")
            if kind == "epoch":
                match = _EPOCH_RE.search(path.name)
                filename_epoch = int(match.group(1)) if match else None
                if filename_epoch != epoch:
                    raise ValueError(
                        f"filename epoch={filename_epoch!r} disagrees with payload epoch={epoch!r}"
                    )
            valid.append(
                (epoch, _KIND_PRIORITY[kind], path.stat().st_mtime_ns, path, kind)
            )
            seen_kinds.add(kind)
            if kind == "epoch":
                # Epoch files are ordered newest first; after this successful load,
                # older numbered files are skipped by ``seen_kinds``.
                pass
        except Exception as exc:
            print(f"[checkpoint-resolver] ignore {path}: {exc}", file=sys.stderr)
            continue

        # Non-epoch kinds are unique.  Keep scanning to compare last/epoch/best.

    if not valid:
        return CheckpointResolution(
            checkpoint="",
            stage=stage,
            epoch=-1,
            next_epoch=0,
            target_epochs=target,
            complete=False,
            kind="none",
        )

    epoch, _, _, path, kind = max(valid, key=lambda row: (row[0], row[1], row[2]))
    next_epoch = epoch + 1
    complete = target > 0 and next_epoch >= target
    return CheckpointResolution(
        checkpoint=str(path),
        stage=stage,
        epoch=epoch,
        next_epoch=next_epoch,
        target_epochs=target,
        complete=complete,
        kind=kind,
    )


def _write_nul(result: CheckpointResolution) -> None:
    fields = (
        result.checkpoint,
        str(result.epoch),
        "1" if result.complete else "0",
        str(result.next_epoch),
        result.kind,
    )
    sys.stdout.buffer.write(b"\0".join(field.encode("utf-8") for field in fields) + b"\0")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Find the newest valid same-stage COWP checkpoint and determine whether the target epoch count is complete."
    )
    ap.add_argument("--output-dir", required=True, type=Path)
    ap.add_argument("--stage", required=True, choices=["natural", "witness", "planner"])
    ap.add_argument("--target-epochs", required=True, type=int)
    ap.add_argument("--format", choices=["json", "nul"], default="json")
    args = ap.parse_args()

    result = resolve_latest_checkpoint(args.output_dir, args.stage, args.target_epochs)
    if args.format == "nul":
        _write_nul(result)
    else:
        print(json.dumps(asdict(result), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
