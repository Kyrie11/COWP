"""Regression test for the Hugging Face ReactSim checkpoint downloader.

The test uses a fake ``hf`` executable and never downloads model bytes.  It
guards the repository type distinction that previously caused the public model
repo to be queried as a dataset repo.
"""
from __future__ import annotations

import hashlib
import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_reactsim_weights_use_model_repo_and_record_checksum(tmp_path):
    carla = tmp_path / "CARLA_0.9.16"
    (carla / "PythonAPI" / "carla").mkdir(parents=True)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    args_file = tmp_path / "hf.args"
    fake_hf = bin_dir / "hf"
    fake_hf.write_text(
        "#!/usr/bin/env bash\n"
        "printf '%s\\n' \"$@\" > \"$HF_ARGS_FILE\"\n"
        "out=''\n"
        "for ((i=1;i<=$#;i++)); do\n"
        "  a=${!i}\n"
        "  if [[ $a == --local-dir ]]; then j=$((i+1)); out=${!j}; fi\n"
        "done\n"
        "printf 'test checkpoint bytes' > \"$out/smart.ckpt\"\n",
        encoding="utf8",
    )
    fake_hf.chmod(0o755)
    payload = b"test checkpoint bytes"
    digest = hashlib.sha256(payload).hexdigest()
    out = tmp_path / "ckpts"
    env = os.environ.copy()
    env.update(
        RIDE_CARLA_ROOT=str(carla),
        DATASET_ROOT=str(tmp_path),
        RIDE_REACTSIM_CHECKPOINTS=str(out),
        RIDE_REACTSIM_EXPECTED_SHA256=digest,
        HF_ARGS_FILE=str(args_file),
        PATH=f"{bin_dir}{os.pathsep}{env['PATH']}",
    )
    result = subprocess.run(
        ["bash", "scripts/14_download_reactsim_weights.sh", "smart.ckpt"],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    args = args_file.read_text(encoding="utf8").splitlines()
    assert "--repo-type" in args
    assert args[args.index("--repo-type") + 1] == "model"
    assert "dataset" not in args
    assert (out / "smart.ckpt").read_bytes() == payload
    assert (out / "smart.ckpt.sha256").read_text(encoding="utf8").startswith(digest)
