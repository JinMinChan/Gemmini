from __future__ import annotations

import argparse
import csv
import glob
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

# Keep these in sync with frontend ROI schema (screen_v2_multicrop base geometry).
ROI_BASE_WIDTH = 2560
ROI_BASE_HEIGHT = 1440
ROI_BOXES: List[Dict[str, Any]] = [
    {"label": "option1", "x": 1043.72, "y": 793.61, "w": 150.71, "h": 112.23},
    {"label": "option2", "x": 1197.63, "y": 795.22, "w": 147.50, "h": 105.81},
    {"label": "option3", "x": 1360.36, "y": 796.02, "w": 158.72, "h": 113.83},
    {"label": "option4", "x": 1512.67, "y": 795.22, "w": 152.31, "h": 109.02},
    {"label": "possible", "x": 1669.79, "y": 794.41, "w": 126.66, "h": 59.32},
    {"label": "cost", "x": 1579.21, "y": 887.40, "w": 96.20, "h": 52.91},
    {"label": "count", "x": 1476.60, "y": 1022.08, "w": 64.13, "h": 49.70},
]

LABEL_COLORS_BGR: Dict[str, Tuple[int, int, int]] = {
    "option1": (60, 60, 255),
    "option2": (60, 220, 60),
    "option3": (255, 160, 60),
    "option4": (255, 60, 200),
    "possible": (255, 220, 60),
    "cost": (60, 220, 255),
    "count": (200, 200, 200),
}


def _roi_corners(box: Dict[str, Any]) -> np.ndarray:
    cx = float(box["x"])
    cy = float(box["y"])
    bw = float(box["w"])
    bh = float(box["h"])
    x1 = cx - bw / 2.0
    y1 = cy - bh / 2.0
    x2 = cx + bw / 2.0
    y2 = cy + bh / 2.0
    pts = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)
    return pts.reshape(-1, 1, 2)


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _load_bgr(path: str) -> Optional[np.ndarray]:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    return img


def _find_homography_orb(
    *,
    ref_kp: List[cv2.KeyPoint],
    ref_desc: np.ndarray,
    tgt_gray: np.ndarray,
    orb: cv2.ORB,
    ratio: float,
    top_n: int,
    ransac_thresh: float,
) -> Tuple[Optional[np.ndarray], int, int, int]:
    kp2, des2 = orb.detectAndCompute(tgt_gray, None)
    if des2 is None or len(kp2) < 20:
        return None, len(kp2), 0, 0

    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    knn = bf.knnMatch(ref_desc, des2, k=2)
    good = []
    for m, n in knn:
        if m.distance < ratio * n.distance:
            good.append(m)
    if len(good) < 30:
        return None, len(kp2), len(good), 0

    good = sorted(good, key=lambda m: m.distance)[: max(30, int(top_n))]
    src_pts = np.float32([ref_kp[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, float(ransac_thresh))
    if H is None or mask is None:
        return None, len(kp2), len(good), 0
    inliers = int(mask.ravel().sum())
    return H, len(kp2), len(good), inliers


def _draw_overlay(
    *,
    img_bgr: np.ndarray,
    H: np.ndarray,
    ref_shape_hw: Tuple[int, int],
    draw_ref_frame: bool,
    title: str,
    matches: int,
    inliers: int,
) -> np.ndarray:
    out = img_bgr.copy()

    if draw_ref_frame:
        ref_h, ref_w = ref_shape_hw
        corners = np.array([[0, 0], [ref_w, 0], [ref_w, ref_h], [0, ref_h]], dtype=np.float32).reshape(-1, 1, 2)
        proj = cv2.perspectiveTransform(corners, H)
        pts = np.clip(proj.reshape(-1, 2), -10000, 100000).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(out, [pts], True, (255, 255, 255), 2, cv2.LINE_AA)

    for box in ROI_BOXES:
        label = str(box["label"])
        corners = _roi_corners(box)
        proj = cv2.perspectiveTransform(corners, H)
        pts = np.clip(proj.reshape(-1, 2), -10000, 100000).astype(np.int32).reshape(-1, 1, 2)
        color = LABEL_COLORS_BGR.get(label, (255, 255, 255))
        cv2.polylines(out, [pts], True, color, 2, cv2.LINE_AA)

        # Label text near the top-left corner of the projected ROI.
        x0, y0 = int(pts[0, 0, 0]), int(pts[0, 0, 1])
        x0 = max(0, min(out.shape[1] - 1, x0))
        y0 = max(0, min(out.shape[0] - 1, y0))
        cv2.putText(
            out,
            label,
            (x0 + 3, max(0, y0 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
            cv2.LINE_AA,
        )

    # Status header
    header = f"{title} | matches={matches} inliers={inliers}"
    cv2.rectangle(out, (10, 10), (10 + min(1200, 14 * len(header)), 48), (0, 0, 0), -1)
    cv2.putText(out, header, (16, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (230, 230, 230), 2, cv2.LINE_AA)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Overlay Gemmini ROI boxes on record images using ORB+homography.")
    ap.add_argument(
        "--ref-image",
        default="test_data/images/2026-02-12_18-08-29_b1e932_001.png",
        help="Reference 16:9 screenshot used as the ROI schema coordinate space.",
    )
    ap.add_argument(
        "--input-glob",
        default="app/records/**/images/*",
        help="Glob for input images (recursive patterns supported).",
    )
    ap.add_argument("--out-dir", default="codex/roi_overlay_records", help="Output directory.")
    ap.add_argument("--limit", type=int, default=0, help="Max number of images to process (0 = all).")
    ap.add_argument("--min-inliers", type=int, default=40, help="Minimum RANSAC inliers to accept a homography.")
    ap.add_argument("--ratio", type=float, default=0.75, help="KNN ratio test threshold.")
    ap.add_argument("--top-n", type=int, default=200, help="Max number of matches used for homography estimation.")
    ap.add_argument("--ransac-thresh", type=float, default=5.0, help="RANSAC reprojection threshold.")
    ap.add_argument("--draw-ref-frame", action="store_true", help="Draw the projected reference frame border.")
    args = ap.parse_args()

    ref_path = str(args.ref_image)
    ref_bgr = _load_bgr(ref_path)
    if ref_bgr is None:
        raise SystemExit(f"Failed to read --ref-image: {ref_path}")
    ref_gray = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2GRAY)

    # Precompute reference features.
    orb = cv2.ORB_create(nfeatures=5000)
    ref_kp, ref_desc = orb.detectAndCompute(ref_gray, None)
    if ref_desc is None or len(ref_kp) < 50:
        raise SystemExit("Reference image has too few features; pick a different --ref-image.")

    in_paths = sorted(glob.glob(str(args.input_glob), recursive=True))
    if args.limit and args.limit > 0:
        in_paths = in_paths[: int(args.limit)]

    out_dir = Path(args.out_dir)
    ok_dir = out_dir / "ok"
    fail_dir = out_dir / "fail"
    _ensure_dir(ok_dir)
    _ensure_dir(fail_dir)

    rows: List[Dict[str, Any]] = []
    ok = 0
    for idx, path in enumerate(in_paths, start=1):
        img_bgr = _load_bgr(path)
        if img_bgr is None:
            rows.append({"path": path, "ok": 0, "reason": "read_fail"})
            continue

        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        H, n_kp, n_matches, inliers = _find_homography_orb(
            ref_kp=ref_kp,
            ref_desc=ref_desc,
            tgt_gray=gray,
            orb=orb,
            ratio=float(args.ratio),
            top_n=int(args.top_n),
            ransac_thresh=float(args.ransac_thresh),
        )

        base_name = Path(path).name
        stem = Path(path).stem
        title = f"{idx:03d}/{len(in_paths):03d} {base_name}"
        if H is None or int(inliers) < int(args.min_inliers):
            out = img_bgr.copy()
            cv2.rectangle(out, (10, 10), (1200, 48), (0, 0, 0), -1)
            msg = f"FAIL {title} | kp={n_kp} matches={n_matches} inliers={inliers}"
            cv2.putText(out, msg, (16, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (60, 60, 255), 2, cv2.LINE_AA)
            out_path = fail_dir / f"{stem}__fail.jpg"
            cv2.imwrite(str(out_path), out, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
            rows.append(
                {
                    "path": path,
                    "ok": 0,
                    "reason": "low_inliers" if H is not None else "no_homography",
                    "kp": n_kp,
                    "matches": n_matches,
                    "inliers": inliers,
                    "out_path": str(out_path),
                }
            )
            continue

        overlay = _draw_overlay(
            img_bgr=img_bgr,
            H=H,
            ref_shape_hw=(ref_gray.shape[0], ref_gray.shape[1]),
            draw_ref_frame=bool(args.draw_ref_frame),
            title=title,
            matches=n_matches,
            inliers=inliers,
        )
        out_path = ok_dir / f"{stem}__overlay.jpg"
        cv2.imwrite(str(out_path), overlay, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        ok += 1
        rows.append(
            {
                "path": path,
                "ok": 1,
                "reason": "ok",
                "kp": n_kp,
                "matches": n_matches,
                "inliers": inliers,
                "out_path": str(out_path),
            }
        )

    # Write CSV summary.
    csv_path = out_dir / "results.csv"
    fieldnames = ["path", "ok", "reason", "kp", "matches", "inliers", "out_path"]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})

    print(f"Processed {len(in_paths)} images. OK={ok} FAIL={len(in_paths) - ok}")
    print(f"Output: {out_dir}")
    print(f"CSV: {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

