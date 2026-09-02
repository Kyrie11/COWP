#!/usr/bin/env bash
set -euo pipefail
REPO="${OCRAP_REPO:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO"; export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"
SOURCE_RUN="${SOURCE_RUN:?SOURCE_RUN is required}"
OUTPUTDIR="${OUTPUTDIR:?OUTPUTDIR is required and must differ from SOURCE_RUN}"
PROTOCOL_ROOT="${PROTOCOL_ROOT:?PROTOCOL_ROOT is required}"
CAL_SAFE="${CAL_SAFE:?CAL_SAFE is required}"
[[ "$OUTPUTDIR" != "$SOURCE_RUN" ]] || { echo "refusing to overwrite source run" >&2; exit 2; }
[[ -f "$PROTOCOL_ROOT/CALIBRATION_PROTOCOL_COMPLETE.json" ]] || { echo "missing protocol manifest" >&2; exit 2; }
rm -rf "$OUTPUTDIR"; mkdir -p "$OUTPUTDIR/candidates" "$OUTPUTDIR/logs"
for variant in balanced precision; do
  src="$SOURCE_RUN/candidates/$variant"
  [[ -f "$src/model_v48_trac_sr/best.pt" ]] || continue
  dst="$OUTPUTDIR/candidates/$variant"; mkdir -p "$dst"
  ln -s "$(realpath --relative-to="$dst" "$src/model_v48_trac_sr")" "$dst/model_v48_trac_sr"
  if [[ -f "$src/POLICY_CONTRACT.env" ]]; then
    ln -s "$(realpath --relative-to="$dst" "$src/POLICY_CONTRACT.env")" "$dst/POLICY_CONTRACT.env"
  else
    cat > "$dst/POLICY_CONTRACT.env" <<EOF
RISK_SOURCE=ordinal_evidence
CONDITIONAL_RECOVERY_RANKING=true
POLICY_FIRST_NO_FALLBACK=false
PROPOSAL_TOP_K=3
EVIDENCE_RERANK_TOP_K=true
MACRO_CONSTRAINT_MODE=opportunity_normalized
MAX_MACRO_EXCESS_SHARE=0.15
EOF
  fi
done
OUTPUTDIR="$OUTPUTDIR" CAL_SAFE="$CAL_SAFE" \
CERT_NEAR="$PROTOCOL_ROOT/certificate_pool_near_contact" \
CERT_CONTACT="$PROTOCOL_ROOT/certificate_pool_contact" \
GPU0="${GPU0:-0}" GPU1="${GPU1:-1}" \
  bash scripts/calibrate_v48_14_certificate_pool.sh
