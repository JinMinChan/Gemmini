#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2

# Ensure repo root is importable when running from ./codex.
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Reuse the server's homography runtime to decide whether a fullframe capture
# actually contains the Lost Ark gem-processing UI.
from app.server import HOMOGRAPHY_TEMPLATE_PATH, HomographyRoiRuntime


@dataclass
class HomographyResult:
    ok: bool
    reason: str
    inliers: int
    good_matches: int


def _read_json(path: Path) -> Optional[dict]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _homography_on_image(rt: HomographyRoiRuntime, img_path: Path) -> Optional[HomographyResult]:
    if not img_path.exists():
        return None
    img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if img is None:
        return None
    _rois, dbg = rt.extract_rois(img)
    ok = bool(dbg.get("ok"))
    reason = str(dbg.get("reason") or ("ok" if ok else "unknown"))
    inliers = int(dbg.get("inliers") or 0)
    good_matches = int(dbg.get("good_matches") or 0)
    return HomographyResult(ok=ok, reason=reason, inliers=inliers, good_matches=good_matches)


def _short(text: str, n: int = 120) -> str:
    t = " ".join(str(text or "").split())
    return t if len(t) <= n else (t[: n - 3] + "...")


def main() -> int:
    ap = argparse.ArgumentParser(description="Analyze Gemmini bug reports (messages) for a given date bucket.")
    ap.add_argument("--date", required=True, help="Date bucket (YYYY-MM-DD), e.g. 2026-02-14")
    ap.add_argument(
        "--out",
        default="codex/bug_report_analysis.md",
        help="Output markdown path (default: codex/bug_report_analysis.md)",
    )
    args = ap.parse_args()

    date = str(args.date).strip()
    msg_dir = Path("app") / "records" / date / "message"
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not msg_dir.exists():
        raise SystemExit(f"message dir not found: {msg_dir}")

    rt = HomographyRoiRuntime(template_path=Path(HOMOGRAPHY_TEMPLATE_PATH))

    rows: List[str] = []
    rows.append(f"# Bug Report Analysis: {date}\n")
    rows.append(f"- message dir: `{msg_dir}`")
    rows.append(f"- homography template: `{Path(HOMOGRAPHY_TEMPLATE_PATH)}`\n")

    msg_paths = sorted(p for p in msg_dir.glob("*.json") if p.is_file())
    rows.append(f"## Summary\n")
    rows.append(f"- messages: {len(msg_paths)}\n")

    low_inliers_hits = 0
    count_total_attachments = 0
    count_fullframe_images = 0

    details: List[str] = []
    for p in msg_paths:
        msg = _read_json(p) or {}
        message = str(msg.get("message") or "").strip()
        record_id = str(msg.get("record_id") or p.stem)
        attachments = msg.get("attachments") or {}
        attached = list(attachments.get("attached_records") or [])

        details.append(f"### {record_id}\n")
        details.append(f"- message: {_short(message) if message else '(empty)'}")
        details.append(f"- attached_records: {len(attached)}\n")

        for rec in attached:
            count_total_attachments += 1
            rid = str(rec.get("record_id") or "")
            ocr_mode = str(rec.get("ocr_mode") or "")
            schema = str(rec.get("roi_schema_version") or "")
            json_rel = str(rec.get("json_path") or "")
            img_rel = rec.get("image_path")
            client_debug = rec.get("client_debug") or {}

            line = f"- {rid} ({ocr_mode}/{schema})"
            if client_debug.get("capture_pipeline"):
                line += f" pipeline={client_debug.get('capture_pipeline')}"
            if client_debug.get("video_w") and client_debug.get("video_h"):
                line += f" video={client_debug.get('video_w')}x{client_debug.get('video_h')}"
            details.append(line)

            # quick ROI size hints (for multicrop)
            roi_rects = list(client_debug.get("roi_rects") or [])
            if roi_rects:
                want = {x.get("label"): x for x in roi_rects if isinstance(x, dict)}
                for k in ("possible", "cost", "count"):
                    r = want.get(k) or {}
                    if r:
                        details.append(f"  - roi[{k}] {r.get('w')}x{r.get('h')} bytes={r.get('bytes')}")

            # load analyze json for parsed state
            if json_rel:
                jpath = Path("app") / json_rel
                rec_json = _read_json(jpath) or {}
                ocr = rec_json.get("ocr_result") or {}
                ui = rec_json.get("ui_state") or {}
                details.append(
                    f"  - parsed: count={ocr.get('count')} possible={ocr.get('possible')} cost={ocr.get('cost')} "
                    f"attempts_left={ui.get('attempts_left')} rerolls={ui.get('rerolls')} options={len(ui.get('options') or [])}"
                )

            # homography check if fullframe image exists
            if isinstance(img_rel, str) and img_rel.strip():
                count_fullframe_images += 1
                img_path = Path("app") / img_rel
                hres = _homography_on_image(rt, img_path)
                if hres:
                    details.append(
                        f"  - homography: ok={hres.ok} reason={hres.reason} inliers={hres.inliers} good_matches={hres.good_matches}"
                    )
                    if not hres.ok and hres.reason == "low_inliers":
                        low_inliers_hits += 1
            details.append("")

    rows.append("## Findings\n")
    rows.append(f"- total attached records: {count_total_attachments}")
    rows.append(f"- fullframe images in attachments: {count_fullframe_images}")
    rows.append(f"- homography low_inliers hits: {low_inliers_hits}\n")
    rows.extend(details)

    out_path.write_text("\n".join(rows).rstrip() + "\n", encoding="utf-8")
    print(f"Wrote: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
