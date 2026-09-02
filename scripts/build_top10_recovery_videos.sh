#!/usr/bin/env bash
set -euo pipefail

# Render exactly ten already-selected positive qualitative examples: 5 Near + 5 Contact.
: "${NEAR_OCRAP_SCENES:?set NEAR_OCRAP_SCENES to selective OC-RAP near .scenes.jsonl}"
: "${NEAR_BASELINE_SCENES:?set NEAR_BASELINE_SCENES to selective paired near baseline scenes}"
: "${CONTACT_OCRAP_SCENES:?set CONTACT_OCRAP_SCENES to selective OC-RAP contact scenes}"
: "${CONTACT_BASELINE_SCENES:?set CONTACT_BASELINE_SCENES to selective paired contact baseline scenes}"
: "${NEAR_SELECTION:?set NEAR_SELECTION generated from full metric journals}"
: "${CONTACT_SELECTION:?set CONTACT_SELECTION generated from full metric journals}"
: "${OUT:=runs/external_comparison/top10_recovery_videos}"
: "${NEAR_BASELINE_NAME:=Best external baseline}"
: "${CONTACT_BASELINE_NAME:=Best external baseline}"
: "${FPS:=10}"
: "${VIEW_RADIUS_M:=35}"
: "${CAMERA_MODE:=fixed}"

mkdir -p "$OUT/near" "$OUT/contact"
PYTHONPATH=src python tools/render_critical_scenes_v48_34.py \
  --method-scenes "$NEAR_OCRAP_SCENES" --control-scenes "$NEAR_BASELINE_SCENES" \
  --selection "$NEAR_SELECTION" --output-dir "$OUT/near" --fps "$FPS" --format mp4 --view-radius-m "$VIEW_RADIUS_M" \
  --camera-mode "$CAMERA_MODE" --method-name "OC-RAP" --control-name "$NEAR_BASELINE_NAME"
PYTHONPATH=src python tools/render_critical_scenes_v48_34.py \
  --method-scenes "$CONTACT_OCRAP_SCENES" --control-scenes "$CONTACT_BASELINE_SCENES" \
  --selection "$CONTACT_SELECTION" --output-dir "$OUT/contact" --fps "$FPS" --format mp4 --view-radius-m "$VIEW_RADIUS_M" \
  --camera-mode "$CAMERA_MODE" --method-name "OC-RAP" --control-name "$CONTACT_BASELINE_NAME"

python - "$OUT" <<'PY'
import json,pathlib,sys
out=pathlib.Path(sys.argv[1]); videos=[]
for regime in ('near','contact'):
    d=json.load((out/regime/'VIDEO_INDEX.json').open())
    rows=d.get('videos',[])
    if len(rows)!=5: raise SystemExit(f'expected 5 {regime} videos, got {len(rows)}')
    videos.extend([{**x,'regime':regime} for x in rows])
if len(videos)!=10: raise SystemExit(f'expected exactly 10 videos, got {len(videos)}')
doc={'event':'top10_recovery_videos_v50','num_videos':10,'exploratory_qualitative_only':True,'selection_note':'deterministic post-hoc selection; not population-level evidence','videos':videos}
(out/'TOP10_VIDEO_INDEX.json').write_text(json.dumps(doc,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'event':doc['event'],'num_videos':10,'output':str(out)}))
PY
