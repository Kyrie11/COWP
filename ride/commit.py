"""Evidence integrity: pre-evaluation commitment to baseline action choices."""
import argparse,hashlib,json,datetime
from pathlib import Path


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--choices',required=True);p.add_argument('--output',required=True)
    a=p.parse_args()
    if not Path(a.choices).exists():raise FileNotFoundError(a.choices)
    commitment={'sha256':digest(a.choices),'choices_file':str(Path(a.choices).resolve()),
                'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
                'note':'Integrity commitment only; not cryptographic proof of pre-outcome timestamp.'}
    dest=Path(a.output);dest.parent.mkdir(parents=True,exist_ok=True)
    if dest.exists():raise FileExistsError('Refuse to silently overwrite existing locked manifest: '+str(dest))
    dest.write_text(json.dumps(commitment,indent=2),encoding='utf8')
    print('LOCKED',commitment['sha256'],dest)
if __name__=='__main__':main()
