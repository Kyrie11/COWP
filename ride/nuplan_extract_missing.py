"""Selective extraction of missing nuPlan DBs from a *verified local* nuPlan ZIP.

DRY RUN by default; writes only requested .db files into a chosen split under
an explicit --apply flag. ZIP must be matched to its true original split by
user; neither content nor filename availability establishes scenario validity.
"""
from __future__ import annotations
import argparse,json,os,shutil,sqlite3,tempfile,zipfile
from pathlib import Path

ALLOWED={'val','train_boston','train_pittsburgh','train_singapore','train_vegas'}


def catalog(archive:Path,missing_list:Path,split:str):
    if split not in ALLOWED:raise ValueError('Only original train/val split DBs are allowed, never test')
    targets={Path(x.strip()).name for x in missing_list.read_text().splitlines() if x.strip()}
    if any(not x.endswith('.db') or '/' in x or '\\' in x for x in targets):
        raise ValueError('Invalid target filename list')
    matches={};duplicates=[]
    with zipfile.ZipFile(archive) as z:
        for item in z.infolist():
            if item.is_dir() or Path(item.filename).name not in targets:continue
            name=Path(item.filename).name
            if name in matches:duplicates.append(name)
            else:matches[name]={'zip_member':item.filename,'bytes':item.file_size}
    if duplicates:raise ValueError('Ambiguous duplicate DB names inside archive: '+repr(duplicates[:10]))
    return matches


def extract(archive:Path,missing_list:Path,output_splits:Path,split:str,apply=False):
    matches=catalog(archive,missing_list,split)
    dest=output_splits/split
    if apply and not dest.is_dir():raise FileNotFoundError('Target original split absent: '+str(dest))
    statuses={};
    for name,data in matches.items():
        p=dest/name
        if p.exists():statuses[name]='ALREADY_EXISTS_REFUSE_OVERWRITE'
        else:statuses[name]='WOULD_EXTRACT' if not apply else 'PENDING'
    if apply:
        with zipfile.ZipFile(archive) as z:
            for name,data in matches.items():
                if statuses[name]!='PENDING':continue
                member=z.getinfo(data['zip_member'])
                if member.file_size<4096:raise ValueError('Suspicious short DB: '+name)
                fd,tmppath=tempfile.mkstemp(prefix='.ride_unverified_',suffix='.db',dir=str(dest))
                os.close(fd)
                try:
                    with z.open(member) as src,open(tmppath,'wb') as dst:shutil.copyfileobj(src,dst)
                    with sqlite3.connect(f'file:{Path(tmppath).resolve()}?mode=ro',uri=True) as con:
                        ok=con.execute('PRAGMA quick_check(1)').fetchone()[0]
                        if ok!='ok':raise ValueError('Corrupt sqlite entry '+name+': '+str(ok))
                    # Requires nonexistence; refuse concurrent overwrite.
                    if p.exists():raise FileExistsError(str(p))
                    os.link(tmppath,p)
                    statuses[name]='EXTRACTED_SQLITE_QUICK_CHECK_OK'
                finally:
                    Path(tmppath).unlink(missing_ok=True)
    return {'archive':str(archive),'split':split,'target_dir':str(dest),
            'apply':apply,'matched_in_archive':len(matches),
            'status':statuses,'note':'This cannot establish original split provenance; verify archive origin before --apply.'}


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--archive',type=Path,required=True)
    p.add_argument('--missing-list',type=Path,required=True)
    p.add_argument('--splits-root',type=Path,required=True)
    p.add_argument('--split',choices=sorted(ALLOWED),required=True)
    p.add_argument('--apply',action='store_true')
    p.add_argument('--report',type=Path,required=True)
    a=p.parse_args(argv)
    res=extract(a.archive,a.missing_list,a.splits_root,a.split,a.apply)
    a.report.parent.mkdir(parents=True,exist_ok=True)
    a.report.write_text(json.dumps(res,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'archive':str(a.archive),'matched_in_archive':res['matched_in_archive'],
                      'apply':a.apply,'report':str(a.report)},indent=2))
if __name__=='__main__':main()
