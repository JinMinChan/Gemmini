#!/usr/bin/env python3
"""
Migrate records layout:

  app/records/YYYY-MM-DD/HH/{images,json,message}/...
    -> app/records/YYYY-MM-DD/{images,json,message}/...

Notes:
- This only touches date folders matching YYYY-MM-DD and hour folders "00".."23".
- It does NOT touch app/records/previous (kept as-is).
- By default this is destructive (moves files/directories). Use --copy for a safer run.
"""

from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path


DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
HOUR_RE = re.compile(r"^\d{2}$")
KINDS = ("images", "json", "message")


def _safe_move(src: Path, dst: Path, *, copy: bool) -> Path:
    if not dst.exists():
        if copy:
            if src.is_dir():
                shutil.copytree(src, dst)
            else:
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
        else:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src), str(dst))
        return dst

    # Collision (should be rare). Keep both by suffixing.
    stem = dst.name
    for i in range(1, 10_000):
        cand = dst.parent / f"{stem}.dup{i}"
        if not cand.exists():
            if copy:
                if src.is_dir():
                    shutil.copytree(src, cand)
                else:
                    cand.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src, cand)
            else:
                cand.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(src), str(cand))
            return cand
    raise RuntimeError(f"too many collisions for {dst}")


def migrate(records_dir: Path, *, dry_run: bool, copy: bool) -> dict:
    records_dir = records_dir.resolve()
    if not records_dir.exists():
        raise FileNotFoundError(f"records_dir not found: {records_dir}")

    moved = 0
    skipped = 0
    removed_dirs = 0

    for date_dir in sorted(records_dir.iterdir()):
        if not date_dir.is_dir():
            continue
        if date_dir.name == "previous":
            continue
        if not DATE_RE.match(date_dir.name):
            continue

        dest_by_kind = {k: (date_dir / k) for k in KINDS}
        for kdir in dest_by_kind.values():
            if not dry_run:
                kdir.mkdir(parents=True, exist_ok=True)

        for hour_dir in sorted(date_dir.iterdir()):
            if not hour_dir.is_dir():
                continue
            if not HOUR_RE.match(hour_dir.name):
                continue
            hour_int = int(hour_dir.name)
            if hour_int < 0 or hour_int > 23:
                continue

            any_moved_from_hour = False
            for kind in KINDS:
                src_kind = hour_dir / kind
                if not src_kind.exists():
                    continue
                if not src_kind.is_dir():
                    skipped += 1
                    continue

                for entry in sorted(src_kind.iterdir()):
                    dst = dest_by_kind[kind] / entry.name
                    if dry_run:
                        moved += 1
                        any_moved_from_hour = True
                        continue
                    _safe_move(entry, dst, copy=copy)
                    moved += 1
                    any_moved_from_hour = True

            # Cleanup empty hour buckets.
            if not dry_run and any_moved_from_hour:
                # Remove leftover empty kind dirs then hour dir.
                for kind in KINDS:
                    src_kind = hour_dir / kind
                    if src_kind.exists():
                        try:
                            src_kind.rmdir()
                            removed_dirs += 1
                        except OSError:
                            pass
                try:
                    hour_dir.rmdir()
                    removed_dirs += 1
                except OSError:
                    pass

    return {"moved_items": moved, "skipped_items": skipped, "removed_dirs": removed_dirs, "dry_run": dry_run, "copy": copy}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--records-dir", default="app/records", help="records root (default: app/records)")
    ap.add_argument("--dry-run", action="store_true", help="scan only; do not modify filesystem")
    ap.add_argument("--copy", action="store_true", help="copy instead of move (safer, but uses more disk)")
    args = ap.parse_args()

    res = migrate(Path(args.records_dir), dry_run=bool(args.dry_run), copy=bool(args.copy))
    print(res)


if __name__ == "__main__":
    main()

