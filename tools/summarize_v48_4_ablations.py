from __future__ import annotations
import argparse, json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True)
    ap.add_argument('--output', required=True)
    args = ap.parse_args()
    root = Path(args.root)
    rows = []
    for run in sorted(p for p in root.iterdir() if p.is_dir()):
        status_path = run / 'screening_status.json'
        if not status_path.exists():
            rows.append({'run': run.name, 'missing': 'screening_status.json'})
            continue
        status = json.loads(status_path.read_text())
        for variant, by_bucket in (status.get('candidates') or {}).items():
            for bucket, d in (by_bucket or {}).items():
                verify = d.get('verify') or {}
                rows.append({
                    'run': run.name,
                    'variant': variant,
                    'bucket': bucket,
                    'valid': bool(d.get('valid')),
                    'candidate_positive_auc': d.get('candidate_auc'),
                    'group_top1_correlation': d.get('top1_corr'),
                    'verify_selected': verify.get('selected'),
                    'verify_precision': verify.get('precision_selected'),
                    'verify_harmful_rate': verify.get('harmful_rate_selected'),
                    'verify_positive_recall': verify.get('positive_recall'),
                })
    payload = {'root': str(root), 'rows': rows}
    Path(args.output).write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(payload, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()
