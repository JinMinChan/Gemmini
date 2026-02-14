#!/usr/bin/env python3
"""
Admin helper for "마음의 편지" (letters) stored under app/records/YYYY-MM-DD/letters/*.json.

Examples:
  conda run -n gem python codex/letters_admin.py list --limit 30
  conda run -n gem python codex/letters_admin.py reply --id 2026-02-15_12-34-56_123456_abcd --text "감사합니다!"
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RECORDS = ROOT / "app" / "records"
DATE_DIR_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def _iter_letter_files() -> list[Path]:
    if not RECORDS.exists():
        return []
    date_dirs = [p for p in RECORDS.iterdir() if p.is_dir() and DATE_DIR_RE.match(p.name or "")]
    out: list[Path] = []
    for day in sorted(date_dirs, key=lambda p: p.name, reverse=True):
        letters = day / "letters"
        if not letters.exists():
            continue
        out.extend(sorted(letters.glob("*.json"), key=lambda p: p.name, reverse=True))
    return out


def cmd_list(limit: int) -> int:
    files = _iter_letter_files()
    if not files:
        print("No letters found.")
        return 0

    shown = 0
    for fp in files:
        try:
            obj = json.loads(fp.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(obj, dict):
            continue
        lid = str(obj.get("id") or fp.stem)
        created = str(obj.get("created_at") or "")
        msg = str(obj.get("message") or "").replace("\n", " ").strip()
        reply = str(obj.get("reply") or "").strip()
        msg_short = (msg[:80] + "...") if len(msg) > 80 else msg
        replied = "Y" if reply else "N"
        print(f"- id={lid} replied={replied} created_at={created} msg={msg_short}")
        shown += 1
        if shown >= limit:
            break
    return 0


def _find_letter_path(letter_id: str) -> Path | None:
    target_name = f"{letter_id}.json"
    for fp in _iter_letter_files():
        if fp.name == target_name:
            return fp
    return None


def cmd_reply(letter_id: str, text: str) -> int:
    fp = _find_letter_path(letter_id)
    if fp is None:
        print(f"Letter not found: {letter_id}")
        return 2
    try:
        obj = json.loads(fp.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"Failed to read JSON: {fp} ({type(e).__name__}: {e})")
        return 2
    if not isinstance(obj, dict):
        print(f"Invalid letter JSON: {fp}")
        return 2

    now = datetime.now(timezone.utc).astimezone().isoformat()
    obj["reply"] = str(text)
    obj["replied_at"] = now
    fp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"OK: replied to {letter_id} ({fp})")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(prog="letters_admin.py")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_list = sub.add_parser("list", help="List recent letters")
    p_list.add_argument("--limit", type=int, default=30)

    p_reply = sub.add_parser("reply", help="Write a reply to a letter")
    p_reply.add_argument("--id", required=True, dest="letter_id")
    p_reply.add_argument("--text", required=True, dest="text")

    args = parser.parse_args()
    if args.cmd == "list":
        lim = max(1, min(200, int(args.limit)))
        return cmd_list(lim)
    if args.cmd == "reply":
        return cmd_reply(str(args.letter_id).strip(), str(args.text).strip())
    parser.error("unknown command")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

