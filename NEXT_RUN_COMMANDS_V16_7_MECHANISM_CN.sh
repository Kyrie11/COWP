#!/usr/bin/env bash
set -euo pipefail
ROOT="${COWP_CODE_ROOT:-$(cd "$(dirname "$0")" && pwd)}"
cd "$ROOT"
PYTHON_BIN="${PYTHON_BIN:-python}"
SOURCE_NATURAL_ROOT="${SOURCE_NATURAL_ROOT:-outputs/cowp_v16_6_natural_recovery_v9labels_seed2026}"
ATTR_GATE="${ATTR_GATE:-outputs/cowp_v16_6_natural_attribution_aligned_v9labels_seed2026/natural_component_attribution_gate.json}"
export OUT_ROOT="${OUT_ROOT:-outputs/cowp_v16_7_mechanism_v9labels_seed2026}"
NAT_BASIS_GATE="$SOURCE_NATURAL_ROOT/eval/learned_offline/natural_basis_gate.json"
NAT_EFFECT_GATE="$SOURCE_NATURAL_ROOT/eval/learned_offline/natural_effectiveness_gate.json"
MAIN_REPORT="$SOURCE_NATURAL_ROOT/eval/learned_offline/learned_natural_effectiveness.json"

"$PYTHON_BIN" - "$NAT_BASIS_GATE" "$NAT_EFFECT_GATE" "$ATTR_GATE" <<'PY'
import json,sys
for p in sys.argv[1:]:
    x=json.load(open(p,encoding='utf-8'))
    assert bool(x.get('pass',x.get('passed',False))), f'gate failed: {p}'
print('validated v16.6 natural basis/effectiveness and aligned attribution')
PY
TARGET_EPOCH="$($PYTHON_BIN - "$MAIN_REPORT" <<'PY'
import json,sys
x=json.load(open(sys.argv[1],encoding='utf-8'))
e=x.get('checkpoint_epoch')
assert isinstance(e,int) and e>=0
print(e)
PY
)"
printf -v TARGET_TAG '%03d' "$TARGET_EPOCH"
CANDIDATE="$SOURCE_NATURAL_ROOT/checkpoints/natural/cowp_natural_epoch${TARGET_TAG}.pt"
BEST="$SOURCE_NATURAL_ROOT/checkpoints/natural/cowp_natural_best.pt"
if [[ -s "$CANDIDATE" ]]; then
  export NATURAL_CKPT="$CANDIDATE"
elif [[ -s "$BEST" ]]; then
  "$PYTHON_BIN" - "$BEST" "$TARGET_EPOCH" <<'PY'
import sys,torch
x=torch.load(sys.argv[1],map_location='cpu')
assert int(x.get('epoch',-1))==int(sys.argv[2]), (x.get('epoch'),sys.argv[2])
PY
  export NATURAL_CKPT="$BEST"
else
  echo "missing selected natural checkpoint under $SOURCE_NATURAL_ROOT/checkpoints/natural" >&2; exit 2
fi
export NATURAL_HISTORY="$SOURCE_NATURAL_ROOT/checkpoints/natural/history_natural.json"
[[ -s "$NATURAL_HISTORY" ]] || { echo "missing NATURAL_HISTORY=$NATURAL_HISTORY" >&2; exit 2; }

# Record exactly which natural checkpoint and attribution evidence are being
# transferred.  The aligned v16.6 attribution package may have evaluated an
# earlier checkpoint epoch; that is valid development evidence for the same
# architecture, but it is not silently treated as exact-checkpoint paper proof.
mkdir -p "$OUT_ROOT/configs"
"$PYTHON_BIN" - "$NATURAL_CKPT" "$NATURAL_HISTORY" "$MAIN_REPORT" "$ATTR_GATE" "$OUT_ROOT/configs/natural_attribution_transfer_manifest.json" <<'PY'
import hashlib,json,sys
from pathlib import Path

def sha256(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):
            h.update(chunk)
    return h.hexdigest()

ckpt,history_path,report_path,attr_path,out_path=map(Path,sys.argv[1:])
report=json.loads(report_path.read_text(encoding='utf-8'))
attr=json.loads(attr_path.read_text(encoding='utf-8'))
selected_epoch=int(report.get('checkpoint_epoch',-1))
attr_epoch=attr.get('checkpoint_epoch', attr.get('aligned_checkpoint_epoch'))
if attr_epoch is None:
    # v16.6 aligned report stores per-arm provenance under main/checkpoints.
    attr_epoch=(attr.get('main') or {}).get('checkpoint_epoch')
exact=(attr_epoch is not None and int(attr_epoch)==selected_epoch)
payload={
    'schema_version':'cowp_natural_attribution_transfer_v1',
    'natural_checkpoint':str(ckpt),
    'natural_checkpoint_sha256':sha256(ckpt),
    'natural_history':str(history_path),
    'natural_history_sha256':sha256(history_path),
    'natural_effectiveness_report':str(report_path),
    'natural_selected_epoch':selected_epoch,
    'attribution_gate':str(attr_path),
    'attribution_gate_sha256':sha256(attr_path),
    'attribution_pass':bool(attr.get('pass',False)),
    'attribution_paper_claim_ready':bool(attr.get('paper_claim_ready',False)),
    'attribution_checkpoint_epoch':attr_epoch,
    'exact_checkpoint_attribution':exact,
    'interpretation':(
        'exact selected-checkpoint attribution' if exact else
        'architecture-level development attribution; repeat multi-seed exact-checkpoint attribution before paper claims'
    ),
}
out_path.write_text(json.dumps(payload,indent=2,ensure_ascii=False),encoding='utf-8')
print(json.dumps(payload,indent=2,ensure_ascii=False))
PY

# v16.7 changes transport/root/candidate certification heads.  Train them from
# the validated natural checkpoint in a fresh provenance root; do not resume a
# v16.6 transport/planner optimizer state.
export STOP_AFTER_STAGE=offline
export RUN_DIAGNOSE="${RUN_DIAGNOSE:-1}"
export RUN_NATURAL=0
export RUN_TRANSPORT=1
export RUN_PLANNER=1
export RUN_OFFLINE=1
export RUN_PROBE=0
export RUN_FULL=0
export FORCE_TRAIN="${FORCE_TRAIN:-1}"
export FORCE_EVAL="${FORCE_EVAL:-1}"
export FREEZE_BACKBONE_EPOCHS="${FREEZE_BACKBONE_EPOCHS:-0}"
export REQUIRE_INIT_CKPT=0
export REQUIRE_WAYMAX_PREFLIGHT=0
export ALLOW_QUALITY_GATE_FAILURE=0
export BACKGROUND="${BACKGROUND:-0}"

# The reported failure happens before any v16.7 transport/planner checkpoint is
# written, but the strict provenance manifest has already been created.  Permit
# this specific pre-training hotfix to amend (not overwrite) provenance so the
# user's original OUT_ROOT can be resumed safely.  Once a downstream checkpoint
# exists, signature changes remain blocked by default.
if [[ -s "$OUT_ROOT/configs/run_provenance.json" \
      && ! -s "$OUT_ROOT/checkpoints/transport/cowp_witness_best.pt" \
      && ! -s "$OUT_ROOT/checkpoints/planner/cowp_planner_best.pt" ]]; then
  export ALLOW_COMPATIBLE_RESUME="${ALLOW_COMPATIBLE_RESUME:-1}"
  export PROVENANCE_RESUME_REASON="${PROVENANCE_RESUME_REASON:-v16.7 natural-checkpoint compatibility hotfix before transport training}"
fi
exec bash "$ROOT/NEXT_RUN_COMMANDS_V16_7_CN.sh"
