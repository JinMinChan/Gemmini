"""
Summarize "stop override" frequency from stored analyze records.

This script is read-only: it scans app/records/**/json/*.json and aggregates:
  - how often rl.action_overridden_by == "goal_success_zero"
  - which targets / remaining attempts appear in those cases

Usage:
  conda run -n gem python codex/summarize_stop_overrides.py
  conda run -n gem python codex/summarize_stop_overrides.py --limit 50
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


def _parse_count(count_text: str | None) -> tuple[int | None, int | None]:
    if not count_text:
        return None, None
    m = re.search(r"(\d+)\s*/\s*(\d+)", str(count_text))
    if not m:
        return None, None
    return int(m.group(1)), int(m.group(2))


def _safe_int(value, default: int | None = None) -> int | None:
    try:
        return int(value)
    except Exception:
        return default


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="app/records", help="records root dir")
    ap.add_argument("--limit", type=int, default=0, help="limit number of files to scan (0=all)")
    args = ap.parse_args(argv)

    root = Path(args.root)
    # Support both legacy per-IP layout and newer date/hour buckets.
    files = sorted(root.glob("**/json/*.json"))
    if args.limit and args.limit > 0:
        files = files[: int(args.limit)]

    total = 0
    by_override = Counter()
    by_action = Counter()
    by_reason = Counter()
    by_target = Counter()
    start_like = 0
    start_like_overridden = 0
    overridden_with_attempts = Counter()
    overridden_with_rerolls = Counter()

    # Keep a few examples for debugging / sanity.
    examples: dict[str, list[str]] = defaultdict(list)

    for p in files:
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        total += 1

        goal = (data.get("goal_success") or {}) if isinstance(data.get("goal_success"), dict) else {}
        rl = (data.get("rl") or {}) if isinstance(data.get("rl"), dict) else {}
        ui = (data.get("ui_state") or {}) if isinstance(data.get("ui_state"), dict) else {}
        ocr = (data.get("ocr_result") or {}) if isinstance(data.get("ocr_result"), dict) else {}

        override = str(rl.get("action_overridden_by") or "")
        action_name = str(rl.get("action_name") or "")
        reason = str(goal.get("reason") or "")
        success_prob = goal.get("success_prob")

        by_override[override or "(none)"] += 1
        by_action[action_name or "(none)"] += 1
        by_reason[reason or "(none)"] += 1

        target = goal.get("target") if isinstance(goal.get("target"), dict) else {}
        if target:
            key = (
                _safe_int(target.get("willpower")),
                _safe_int(target.get("points")),
                _safe_int(target.get("effect1_level")),
                _safe_int(target.get("effect2_level")),
            )
            by_target[key] += 1

        # "Start-like" = attempts_left == max_attempts from OCR count (e.g. 7/7, 9/9)
        left, right = _parse_count(ocr.get("count"))
        attempts_left = left if left is not None else _safe_int(ui.get("attempts_left"), 0) or 0
        rerolls = _safe_int(ui.get("rerolls"), 0) or 0
        if right in (7, 9) and left == right and int(left) > 0:
            start_like += 1
            if override == "goal_success_zero":
                start_like_overridden += 1

        if override == "goal_success_zero":
            overridden_with_attempts[int(attempts_left)] += 1
            overridden_with_rerolls[int(rerolls)] += 1
            if success_prob in (0, 0.0, "0", "0.0"):
                if len(examples["zero"]) < 8:
                    examples["zero"].append(str(p))
            else:
                if len(examples["nonzero"]) < 8:
                    examples["nonzero"].append(str(p))

    print(f"Scanned: {total} files (root={root})")
    print()

    print("== Action Distribution ==")
    for k, v in by_action.most_common():
        print(f"{k:>16}: {v}")
    print()

    print("== Override Distribution (rl.action_overridden_by) ==")
    for k, v in by_override.most_common():
        print(f"{k:>24}: {v}")
    print()

    print("== Goal Success Reason Distribution ==")
    for k, v in by_reason.most_common(10):
        print(f"{k:>24}: {v}")
    print()

    print("== Top Targets (willpower, points, e1, e2) ==")
    for k, v in by_target.most_common(10):
        print(f"{k}: {v}")
    print()

    if start_like:
        rate = (start_like_overridden / float(start_like)) * 100.0
        print(f"Start-like (7/7 or 9/9) records: {start_like}")
        print(f"  overridden_by goal_success_zero: {start_like_overridden} ({rate:.1f}%)")
        print()

    if by_override.get("goal_success_zero", 0) > 0:
        print("== Overridden AttemptsLeft (approx) ==")
        for k, v in overridden_with_attempts.most_common():
            print(f"attempts_left={k:>2}: {v}")
        print()

        print("== Overridden Rerolls ==")
        for k, v in overridden_with_rerolls.most_common():
            print(f"rerolls={k:>2}: {v}")
        print()

    if examples:
        print("== Example Files ==")
        for k, paths in examples.items():
            print(f"[{k}]")
            for pp in paths:
                print(f"  {pp}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
