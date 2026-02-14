"""
Generate a small ROI placement example image for the "화면 조정" modal.

- Input: a representative 16:9 Lost Ark gem-processing screenshot (2560x1440)
- Output: JPEG under frontend/static and app/static so both Vercel + local server can serve it

We avoid Pillow to keep dependencies minimal (OpenCV is already in the gem conda env).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np


@dataclass(frozen=True)
class RoiBox:
    label: str
    x: float  # center x in base coords
    y: float  # center y in base coords
    w: float
    h: float


ROI_BASE_W = 2560
ROI_BASE_H = 1440

# Keep these in sync with ROI_BOXES in frontend/index.html (screen_v2_multicrop).
ROI_BOXES: List[RoiBox] = [
    RoiBox("option1", 1043.72, 793.61, 150.71, 112.23),
    RoiBox("option2", 1197.63, 795.22, 147.50, 105.81),
    RoiBox("option3", 1360.36, 796.02, 158.72, 113.83),
    RoiBox("option4", 1512.67, 795.22, 152.31, 109.02),
    RoiBox("possible", 1669.79, 794.41, 126.66, 59.32),
    RoiBox("cost", 1579.21, 887.40, 150.00, 70.00),
    RoiBox("count", 1476.60, 1022.08, 120.00, 80.00),
]


def _center_to_rect(box: RoiBox) -> Tuple[int, int, int, int]:
    x1 = int(round(box.x - box.w / 2))
    y1 = int(round(box.y - box.h / 2))
    x2 = int(round(box.x + box.w / 2))
    y2 = int(round(box.y + box.h / 2))
    return x1, y1, x2, y2


def _clamp_rect(x1: int, y1: int, x2: int, y2: int, w: int, h: int) -> Tuple[int, int, int, int]:
    x1 = max(0, min(w - 1, x1))
    y1 = max(0, min(h - 1, y1))
    x2 = max(1, min(w, x2))
    y2 = max(1, min(h, y2))
    if x2 <= x1:
        x2 = min(w, x1 + 1)
    if y2 <= y1:
        y2 = min(h, y1 + 1)
    return x1, y1, x2, y2


def _draw_alpha_rect(img: np.ndarray, rect: Tuple[int, int, int, int], bgr: Tuple[int, int, int], alpha: float) -> None:
    x1, y1, x2, y2 = rect
    if alpha <= 0:
        return
    overlay = img.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), bgr, thickness=-1)
    cv2.addWeighted(overlay, alpha, img, 1.0 - alpha, 0, dst=img)


def main() -> None:
    repo = Path(__file__).resolve().parents[1]
    default_in = repo / "test_data/images/2026-02-12_18-08-29_b1e932_001.png"
    src_path = default_in
    if not src_path.exists():
        raise SystemExit(f"Input image not found: {src_path}")

    img = cv2.imread(str(src_path), cv2.IMREAD_COLOR)
    if img is None:
        raise SystemExit(f"Failed to read image: {src_path}")

    h, w = img.shape[:2]
    if (w, h) != (ROI_BASE_W, ROI_BASE_H):
        # We expect a 2560x1440 reference; still handle other sizes by scaling.
        sx = w / float(ROI_BASE_W)
        sy = h / float(ROI_BASE_H)
    else:
        sx = 1.0
        sy = 1.0

    rects: Dict[str, Tuple[int, int, int, int]] = {}
    for box in ROI_BOXES:
        x1, y1, x2, y2 = _center_to_rect(box)
        x1 = int(round(x1 * sx))
        x2 = int(round(x2 * sx))
        y1 = int(round(y1 * sy))
        y2 = int(round(y2 * sy))
        rects[box.label] = _clamp_rect(x1, y1, x2, y2, w=w, h=h)

    # Crop to the relevant ROI cluster with padding so the example is readable.
    labels = ["option1", "option2", "option3", "option4", "possible", "cost", "count"]
    xs1 = [rects[l][0] for l in labels]
    ys1 = [rects[l][1] for l in labels]
    xs2 = [rects[l][2] for l in labels]
    ys2 = [rects[l][3] for l in labels]
    pad_x = int(round(220 * sx))
    pad_y = int(round(220 * sy))
    cx1 = max(0, min(xs1) - pad_x)
    cy1 = max(0, min(ys1) - pad_y)
    cx2 = min(w, max(xs2) + pad_x)
    cy2 = min(h, max(ys2) + pad_y)
    crop = img[cy1:cy2, cx1:cx2].copy()
    ch, cw = crop.shape[:2]

    # Scale down for web delivery.
    out_w = 900
    if cw <= 0 or ch <= 0:
        raise SystemExit("Invalid crop size")
    scale = out_w / float(cw)
    out_h = max(1, int(round(ch * scale)))
    crop_small = cv2.resize(crop, (out_w, out_h), interpolation=cv2.INTER_AREA)

    def scale_rect(rect: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
        x1, y1, x2, y2 = rect
        x1 -= cx1
        x2 -= cx1
        y1 -= cy1
        y2 -= cy1
        x1s = int(round(x1 * scale))
        x2s = int(round(x2 * scale))
        y1s = int(round(y1 * scale))
        y2s = int(round(y2 * scale))
        return _clamp_rect(x1s, y1s, x2s, y2s, w=out_w, h=out_h)

    # Colors (BGR) roughly matching the UI legend.
    COLOR_OPTIONS = (120, 115, 255)  # rose-ish
    COLOR_REROLL = (255, 255, 110)   # cyan-ish
    COLOR_COUNT = (80, 205, 255)     # amber-ish
    COLOR_COST = (140, 255, 140)     # emerald-ish

    overlay = crop_small.copy()

    # Options union + separators
    opt_rects = [scale_rect(rects[l]) for l in ["option1", "option2", "option3", "option4"]]
    xmin = min(r[0] for r in opt_rects)
    ymin = min(r[1] for r in opt_rects)
    xmax = max(r[2] for r in opt_rects)
    ymax = max(r[3] for r in opt_rects)
    union = _clamp_rect(xmin, ymin, xmax, ymax, w=out_w, h=out_h)

    _draw_alpha_rect(overlay, union, COLOR_OPTIONS, alpha=0.12)

    # Boundaries between option boxes.
    opt_rects_sorted = sorted(opt_rects, key=lambda r: (r[0] + r[2]) / 2)
    boundaries = []
    for i in range(3):
        left = opt_rects_sorted[i]
        right = opt_rects_sorted[i + 1]
        bx = int(round(((left[2] + right[0]) / 2)))
        boundaries.append(bx)
    for bx in boundaries:
        cv2.line(overlay, (bx, union[1]), (bx, union[3]), COLOR_OPTIONS, thickness=2, lineType=cv2.LINE_AA)

    # Single boxes
    rr = scale_rect(rects["possible"])
    cc = scale_rect(rects["count"])
    co = scale_rect(rects["cost"])
    _draw_alpha_rect(overlay, rr, COLOR_REROLL, alpha=0.12)
    _draw_alpha_rect(overlay, cc, COLOR_COUNT, alpha=0.12)
    _draw_alpha_rect(overlay, co, COLOR_COST, alpha=0.12)

    # Borders last
    cv2.rectangle(overlay, (union[0], union[1]), (union[2], union[3]), COLOR_OPTIONS, thickness=3, lineType=cv2.LINE_AA)
    cv2.rectangle(overlay, (rr[0], rr[1]), (rr[2], rr[3]), COLOR_REROLL, thickness=3, lineType=cv2.LINE_AA)
    cv2.rectangle(overlay, (cc[0], cc[1]), (cc[2], cc[3]), COLOR_COUNT, thickness=3, lineType=cv2.LINE_AA)
    cv2.rectangle(overlay, (co[0], co[1]), (co[2], co[3]), COLOR_COST, thickness=3, lineType=cv2.LINE_AA)

    # Output
    out_frontend = repo / "frontend/static/roi_example.jpg"
    out_app = repo / "app/static/roi_example.jpg"
    out_frontend.parent.mkdir(parents=True, exist_ok=True)
    out_app.parent.mkdir(parents=True, exist_ok=True)

    params = [int(cv2.IMWRITE_JPEG_QUALITY), 86]
    ok1 = cv2.imwrite(str(out_frontend), overlay, params)
    ok2 = cv2.imwrite(str(out_app), overlay, params)
    if not ok1 or not ok2:
        raise SystemExit("Failed to write output images")

    print(f"Wrote: {out_frontend}")
    print(f"Wrote: {out_app}")


if __name__ == "__main__":
    main()

