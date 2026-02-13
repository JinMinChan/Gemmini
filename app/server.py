from __future__ import annotations

import hashlib
import json
import os
import re
import secrets
import shutil
import sys
import threading
import time
from collections import defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent
STATIC_DIR = BASE_DIR / "static"
FRONTEND_INDEX = ROOT_DIR / "frontend" / "index.html"
UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
RECORDS_DIR = BASE_DIR / "records"
RECORDS_DIR.mkdir(parents=True, exist_ok=True)

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

DEFAULT_MODEL_PATH_STEM = (
    ROOT_DIR / "models" / "production" / "gemmini_v9" / "best_model"
)

GOAL_SUCCESS_MODEL_PATH = ROOT_DIR / "goal_success" / "runs" / "dist625" / "best.pt"
GOAL_SUCCESS_MODEL_PATH_FALLBACK = ROOT_DIR / "goal_success" / "runs" / "dist625_test" / "best.pt"

GOAL_SUCCESS_MC_ROLLOUTS = int(os.getenv("GEMMINI_GOAL_MC_ROLLOUTS", "256"))
PROXY_SHARED_SECRET = str(os.getenv("GEMMINI_PROXY_SHARED_SECRET", "") or "").strip()
PROXY_SHARED_SECRET_HEADER = "x-gemmini-key"
PROXY_CLIENT_IP_HEADER = "x-gemmini-client-ip"
DEFAULT_ROI_SCHEMA_VERSION = "screen_v1"
MULTICROP_ROI_SCHEMA_VERSION = "screen_v2_multicrop"
SUPPORTED_ROI_SCHEMA_VERSIONS = {DEFAULT_ROI_SCHEMA_VERSION, MULTICROP_ROI_SCHEMA_VERSION}
ROI_REQUIRED_LABELS = ("option1", "option2", "option3", "option4", "possible", "cost", "count")

MAX_UPLOAD_BYTES = 12 * 1024 * 1024  # 12MB
ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}
ALLOWED_MIME_TYPES = {"image/png", "image/jpeg", "image/webp"}

RATE_LIMIT_WINDOW_SECONDS = int(os.getenv("GEMMINI_RATE_LIMIT_WINDOW_SECONDS", "60"))
RATE_LIMIT_MAX_REQUESTS = int(os.getenv("GEMMINI_RATE_LIMIT_MAX_REQUESTS", "20"))

REPORT_CACHE_TTL_SECONDS = int(os.getenv("GEMMINI_REPORT_CACHE_TTL_SECONDS", "1800"))
REPORT_CACHE_MAX_ITEMS_PER_CLIENT = int(os.getenv("GEMMINI_REPORT_CACHE_MAX_ITEMS_PER_CLIENT", "6"))
REPORT_CACHE_MAX_TOTAL_MB = int(os.getenv("GEMMINI_REPORT_CACHE_MAX_TOTAL_MB", "256"))
REPORT_CACHE_MAX_ENTRY_BYTES = int(os.getenv("GEMMINI_REPORT_CACHE_MAX_ENTRY_BYTES", str(8 * 1024 * 1024)))
REPORT_ATTACH_MAX_ITEMS = int(os.getenv("GEMMINI_REPORT_ATTACH_MAX_ITEMS", "3"))

HOMOGRAPHY_TEMPLATE_PATH = Path(
    str(os.getenv("GEMMINI_HOMOGRAPHY_TEMPLATE_PATH", "") or "").strip()
    or (ROOT_DIR / "test_data" / "images" / "2026-02-12_18-08-29_b1e932_001.png")
)
HOMOGRAPHY_ENABLED = str(os.getenv("GEMMINI_HOMOGRAPHY_ENABLED", "1") or "").strip().lower() not in (
    "0",
    "false",
    "no",
    "off",
)
HOMOGRAPHY_NFEATURES = int(os.getenv("GEMMINI_HOMOGRAPHY_NFEATURES", "5000"))
HOMOGRAPHY_RATIO = float(os.getenv("GEMMINI_HOMOGRAPHY_RATIO", "0.75"))
HOMOGRAPHY_TOP_N = int(os.getenv("GEMMINI_HOMOGRAPHY_TOP_N", "200"))
HOMOGRAPHY_MIN_GOOD_MATCHES = int(os.getenv("GEMMINI_HOMOGRAPHY_MIN_GOOD_MATCHES", "30"))
HOMOGRAPHY_MIN_INLIERS = int(os.getenv("GEMMINI_HOMOGRAPHY_MIN_INLIERS", "40"))
HOMOGRAPHY_RANSAC_THRESH = float(os.getenv("GEMMINI_HOMOGRAPHY_RANSAC_THRESH", "5.0"))
HOMOGRAPHY_ROI_PAD_PX = int(os.getenv("GEMMINI_HOMOGRAPHY_ROI_PAD_PX", "6"))

ACTION_NAME = {
    0: "process",
    1: "reroll",
    2: "stop",
}

_requests_by_key: Dict[str, Deque[float]] = defaultdict(deque)
_parser_instance = None
_rl_runtime = None
_goal_success_runtime = None
_runtime_lock = threading.Lock()
_report_cache_lock = threading.Lock()
_report_cache_total_bytes = 0
_report_cache: Dict[str, Deque[dict]] = defaultdict(deque)
_homography_lock = threading.Lock()
_homography_runtime = None
_homography_runtime_error: Optional[str] = None

app = FastAPI(title="Gemmini Web API", version="0.3.0")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.on_event("startup")
async def _startup_warmup() -> None:
    """
    Warm up heavy runtimes so the first "젬미나이" click doesn't pay model init costs.
    Set GEMMINI_WARMUP=0 to disable.
    """
    raw = os.getenv("GEMMINI_WARMUP", "1").strip().lower()
    if raw in ("0", "false", "no", "off"):
        return

    t0 = time.perf_counter()
    try:
        _get_parser()
        _get_rl_runtime()
    except Exception as e:
        # Warmup failure shouldn't prevent the server from starting.
        print(f"[warmup] failed: {type(e).__name__}: {e}")
    else:
        dt_ms = (time.perf_counter() - t0) * 1000.0
        print(f"[warmup] ready in {dt_ms:.0f}ms")

cors_origins = os.getenv("CORS_ORIGINS", "*")
allowed_origins = ["*"] if cors_origins == "*" else [x.strip() for x in cors_origins.split(",") if x.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


def _to_builtin(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {k: _to_builtin(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_builtin(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_to_builtin(v) for v in value)
    return value


def _client_ip(request: Request) -> str:
    # When deployed behind a trusted proxy (e.g., Vercel), the proxy should pass
    # the original client IP via a dedicated header. We only trust this when the
    # shared-secret mode is enabled, otherwise anyone could spoof it.
    if PROXY_SHARED_SECRET:
        raw = str(request.headers.get(PROXY_CLIENT_IP_HEADER, "") or "").strip()
        if raw:
            return raw.split(",")[0].strip()
    xff = request.headers.get("x-forwarded-for", "").strip()
    if xff:
        return xff.split(",")[0].strip()
    if request.client:
        return request.client.host
    return "unknown"


def _record_dirs_for_time_bucket(dt: datetime, *, create: bool = True) -> tuple[str, str, Path, Path, Path]:
    """
    Records are bucketed by local date/hour for easier debugging when proxied IPs
    are ambiguous (e.g. Vercel).

    Layout:
      app/records/YYYY-MM-DD/HH/{images,json,message}/...
    """
    dt_local = dt.astimezone()
    date_dir = dt_local.strftime("%Y-%m-%d")
    hour_dir = dt_local.strftime("%H")
    base_dir = RECORDS_DIR / date_dir / hour_dir
    images_dir = base_dir / "images"
    json_dir = base_dir / "json"
    message_dir = base_dir / "message"
    if create:
        images_dir.mkdir(parents=True, exist_ok=True)
        json_dir.mkdir(parents=True, exist_ok=True)
        message_dir.mkdir(parents=True, exist_ok=True)
    return date_dir, hour_dir, images_dir, json_dir, message_dir


def _sanitize_rate_key(value: str) -> str:
    raw = str(value or "").strip()
    if not raw:
        return ""
    sanitized = re.sub(r"[^0-9A-Za-z._:-]+", "_", raw).strip("._:-")
    return sanitized[:80]


def _rate_limit_key(*, client_ip: str, client_id: Optional[str]) -> str:
    cid = _sanitize_rate_key(client_id) if client_id else ""
    if cid:
        return f"cid:{cid}"
    ip = _sanitize_rate_key(client_ip) or "unknown"
    return f"ip:{ip}"


def _check_rate_limit(key: str) -> None:
    now = time.time()
    bucket = _requests_by_key[key]
    while bucket and (now - bucket[0] > RATE_LIMIT_WINDOW_SECONDS):
        bucket.popleft()
    if len(bucket) >= RATE_LIMIT_MAX_REQUESTS:
        raise HTTPException(status_code=429, detail="Too many requests. Try again in a minute.")
    bucket.append(now)


def _verify_proxy_secret(request: Request) -> None:
    """
    Optional shared-secret guard for reverse-proxy deployments (e.g. Vercel -> AI server).
    If GEMMINI_PROXY_SHARED_SECRET is unset, this check is disabled.
    """
    if not PROXY_SHARED_SECRET:
        return
    supplied = str(request.headers.get(PROXY_SHARED_SECRET_HEADER, "") or "").strip()
    if not supplied or supplied != PROXY_SHARED_SECRET:
        raise HTTPException(status_code=401, detail="Unauthorized proxy key")

def _ext_from_kind(kind: str) -> str:
    k = str(kind or "").strip().lower()
    if k == "jpeg":
        return "jpg"
    if k in ("png", "webp", "jpg"):
        return k
    return "bin"


def _prune_report_cache_locked(now_ts: float) -> None:
    global _report_cache_total_bytes
    expire_before = float(now_ts) - float(REPORT_CACHE_TTL_SECONDS)
    # Drop expired + enforce per-client caps.
    for cid in list(_report_cache.keys()):
        bucket = _report_cache.get(cid)
        if not bucket:
            _report_cache.pop(cid, None)
            continue
        while bucket and float(bucket[0].get("ts", 0.0)) < expire_before:
            popped = bucket.popleft()
            _report_cache_total_bytes -= int(popped.get("bytes", 0) or 0)
        while len(bucket) > int(REPORT_CACHE_MAX_ITEMS_PER_CLIENT):
            popped = bucket.popleft()
            _report_cache_total_bytes -= int(popped.get("bytes", 0) or 0)
        if not bucket:
            _report_cache.pop(cid, None)

    # Drop oldest globally if over total budget.
    max_total = int(REPORT_CACHE_MAX_TOTAL_MB) * 1024 * 1024
    while _report_cache_total_bytes > max_total and _report_cache:
        oldest_cid: Optional[str] = None
        oldest_ts: Optional[float] = None
        for cid, bucket in _report_cache.items():
            if not bucket:
                continue
            ts = float(bucket[0].get("ts", 0.0))
            if oldest_ts is None or ts < oldest_ts:
                oldest_ts = ts
                oldest_cid = cid
        if oldest_cid is None:
            break
        bucket = _report_cache.get(oldest_cid)
        if not bucket:
            _report_cache.pop(oldest_cid, None)
            continue
        popped = bucket.popleft()
        _report_cache_total_bytes -= int(popped.get("bytes", 0) or 0)
        if not bucket:
            _report_cache.pop(oldest_cid, None)


def _report_cache_add(
    *,
    client_id: Optional[str],
    entry: dict,
) -> None:
    """
    Keep a small in-memory cache of recent analyze inputs keyed by client_id.
    This enables "attach evidence to bug report" without writing ROI images to
    disk on every analyze call.
    """
    cid = _sanitize_rate_key(client_id) if client_id else ""
    if not cid:
        return
    now_ts = time.time()
    item_bytes = int(entry.get("bytes", 0) or 0)
    if item_bytes < 0:
        item_bytes = 0
    if item_bytes > int(REPORT_CACHE_MAX_ENTRY_BYTES):
        # Avoid caching suspiciously large payloads.
        entry = dict(entry)
        entry["rois"] = {}
        entry["bytes"] = 0
        item_bytes = 0
    entry["ts"] = float(entry.get("ts") or now_ts)
    entry["cached_at"] = float(now_ts)

    global _report_cache_total_bytes
    with _report_cache_lock:
        _prune_report_cache_locked(now_ts)
        bucket = _report_cache[cid]
        bucket.append(entry)
        _report_cache_total_bytes += item_bytes
        _prune_report_cache_locked(now_ts)


def _report_cache_pick(
    *,
    client_id: Optional[str],
    record_id: Optional[str],
) -> List[dict]:
    cid = _sanitize_rate_key(client_id) if client_id else ""
    if not cid:
        return []
    now_ts = time.time()
    with _report_cache_lock:
        _prune_report_cache_locked(now_ts)
        items = list(_report_cache.get(cid) or [])

    if not items:
        return []

    max_items = max(1, int(REPORT_ATTACH_MAX_ITEMS))
    rid = str(record_id or "").strip()
    if rid:
        hit_idx = None
        for idx, item in enumerate(items):
            if str(item.get("record_id") or "") == rid:
                hit_idx = idx
                break
        if hit_idx is not None:
            start = max(0, hit_idx - max_items + 1)
            return items[start : hit_idx + 1]
    return items[-max_items:]


def _persist_report_attachments(
    *,
    report_abs_path: Path,
    report_rel_path: str,
    client_id: Optional[str],
    record_id: Optional[str],
    client_context: Any,
) -> dict:
    """
    Persist a "bundle" directory next to the bug report json containing:
    - ROI images (when available)
    - copies of referenced analyze json files (small; easier to inspect)
    Returns an attachments payload suitable for embedding in the report json.
    """
    report_rel = Path(str(report_rel_path))
    bundle_name = report_rel.stem
    bundle_abs_dir = report_abs_path.parent / bundle_name
    roi_abs_dir = bundle_abs_dir / "roi"
    analyze_abs_dir = bundle_abs_dir / "analyze_json"
    roi_abs_dir.mkdir(parents=True, exist_ok=True)
    analyze_abs_dir.mkdir(parents=True, exist_ok=True)

    bundle_rel_dir = report_rel.parent / bundle_name
    selected = _report_cache_pick(client_id=client_id, record_id=record_id)

    saved_items: List[dict] = []
    for item in selected:
        rid = str(item.get("record_id") or "").strip()
        bucket = item.get("record_bucket") or {}
        date_dir = str(bucket.get("date") or "").strip()
        hour_dir = str(bucket.get("hour") or "").strip()
        json_rel_path = str(item.get("json_path") or "").strip() or None
        image_rel_path = str(item.get("image_path") or "").strip() or None

        roi_files: List[dict] = []
        rois = item.get("rois") if isinstance(item.get("rois"), dict) else {}
        for label, roi in (rois or {}).items():
            if not isinstance(roi, dict):
                continue
            kind = roi.get("kind") or "png"
            ext = _ext_from_kind(str(kind))
            data = roi.get("data")
            if not isinstance(data, (bytes, bytearray)):
                continue
            out_name = f"{rid}__{label}.{ext}"
            out_abs = roi_abs_dir / out_name
            out_abs.write_bytes(bytes(data))
            roi_files.append(
                {
                    "label": str(label),
                    "kind": str(kind),
                    "bytes": int(len(data)),
                    "path": str((bundle_rel_dir / "roi" / out_name).as_posix()),
                }
            )

        # Copy analyze json for convenience (small).
        copied_analyze = None
        if date_dir and hour_dir and rid:
            src_abs = RECORDS_DIR / date_dir / hour_dir / "json" / f"{rid}.json"
            if src_abs.exists():
                dst_abs = analyze_abs_dir / f"{rid}.json"
                try:
                    shutil.copy2(src_abs, dst_abs)
                    copied_analyze = str((bundle_rel_dir / "analyze_json" / f"{rid}.json").as_posix())
                except Exception:
                    copied_analyze = None

        saved_items.append(
            {
                "record_id": rid or None,
                "record_bucket": {"date": date_dir or None, "hour": hour_dir or None},
                "ocr_mode": item.get("ocr_mode"),
                "roi_schema_version": item.get("roi_schema_version"),
                "json_path": json_rel_path,
                "image_path": image_rel_path,
                "roi_files": roi_files,
                "analyze_json_copy": copied_analyze,
                "client_debug": item.get("client_debug"),
            }
        )

    meta = {
        "bundle": str(bundle_rel_dir.as_posix()),
        "attached_records": saved_items,
        "attached_count": int(len(saved_items)),
        "client_id": _sanitize_rate_key(client_id) or None,
        "record_id": str(record_id or "").strip() or None,
        "client_context": _to_builtin(client_context) if client_context is not None else None,
        "cached_limits": {
            "ttl_seconds": int(REPORT_CACHE_TTL_SECONDS),
            "max_items_per_client": int(REPORT_CACHE_MAX_ITEMS_PER_CLIENT),
            "max_total_mb": int(REPORT_CACHE_MAX_TOTAL_MB),
            "attach_max_items": int(REPORT_ATTACH_MAX_ITEMS),
        },
    }
    (bundle_abs_dir / "meta.json").write_text(
        json.dumps(_to_builtin(meta), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return meta


def _sniff_image_kind(data: bytes) -> str | None:
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "png"
    if len(data) >= 3 and data[:3] == b"\xff\xd8\xff":
        return "jpeg"
    if len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "webp"
    return None


def _validate_stat(name: str, value: Optional[int], default: int = 1) -> int:
    if value is None:
        return default
    v = int(value)
    if v < 1 or v > 5:
        raise HTTPException(status_code=400, detail=f"{name} must be in [1,5]")
    return v


def _validate_choice(name: str, value: Optional[str], *, allowed: set[str], default: str) -> str:
    if value is None:
        return str(default)
    v = str(value).strip().lower()
    if not v:
        return str(default)
    if v not in allowed:
        raise HTTPException(status_code=400, detail=f"{name} must be one of {sorted(allowed)}")
    return v


def _normalize_roi_schema_version(value: Optional[str]) -> tuple[str, bool]:
    """
    Soft validation for client ROI schema.
    Unknown schema is allowed (for forward compatibility) and simply marked unsupported.
    """
    if value is None:
        return DEFAULT_ROI_SCHEMA_VERSION, True
    v = re.sub(r"[^a-zA-Z0-9._-]+", "", str(value).strip().lower())
    if not v:
        return DEFAULT_ROI_SCHEMA_VERSION, True
    return v, bool(v in SUPPORTED_ROI_SCHEMA_VERSIONS)


def _resolve_rl_model_path() -> Path:
    """
    Resolve RL model path with priority:
    1) GEMMINI_RL_MODEL_PATH env override
    2) production default model (gemmini_v9)
    """
    env_path = str(os.getenv("GEMMINI_RL_MODEL_PATH", "") or "").strip()
    candidates: List[Path] = []
    if env_path:
        candidates.append(Path(env_path))
    candidates.append(DEFAULT_MODEL_PATH_STEM)

    for cand in candidates:
        if cand.exists():
            return cand
        z = cand.with_suffix(".zip")
        if z.exists():
            return z

    tried = [str(p) for p in candidates]
    raise FileNotFoundError(f"RL model not found. tried={tried}")


async def _read_valid_image(image: UploadFile) -> tuple[str, bytes, str]:
    filename = image.filename or "upload"
    ext = Path(filename).suffix.lower()
    if ext and ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"Unsupported extension: {ext}")
    if image.content_type not in ALLOWED_MIME_TYPES:
        raise HTTPException(status_code=400, detail=f"Unsupported content type: {image.content_type}")

    data = await image.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty file")
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail=f"File too large. Max {MAX_UPLOAD_BYTES // (1024 * 1024)}MB")

    detected = _sniff_image_kind(data)
    if detected is None:
        raise HTTPException(status_code=400, detail="Invalid image signature")

    return filename, data, detected


def _build_record_name(detected: str, *, now: Optional[datetime] = None) -> tuple[str, str, datetime]:
    """
    Build a unique record id. We include microseconds + a short random suffix to
    avoid directory scans for sequence numbers (which become expensive when
    bucketing by hour across many users).
    """
    dt = (now or datetime.now(timezone.utc)).astimezone()
    date_str = dt.strftime("%Y-%m-%d_%H-%M-%S_%f")
    suffix = secrets.token_hex(2)  # 4 hex chars
    ext = "jpg" if detected == "jpeg" else detected
    base = f"{date_str}_{suffix}"
    return base, ext, dt


def _persist_image_bytes(base: str, ext: str, data: bytes, *, record_dt: datetime) -> tuple[str, str]:
    stored_name = f"{base}.{ext}"
    date_dir, hour_dir, images_dir, _, _ = _record_dirs_for_time_bucket(record_dt, create=True)
    (images_dir / stored_name).write_bytes(data)
    return stored_name, f"records/{date_dir}/{hour_dir}/images/{stored_name}"


def _persist_analyze_record(
    *,
    base: str,
    record_dt: datetime,
    created_at: str,
    image_rel_path: Optional[str],
    source_filename: str,
    client_ip: str,
    client_id: Optional[str],
    elapsed_ms: float,
    manual_stats: dict,
    goal_success: Optional[dict],
    ocr_result: dict,
    ui_state: dict,
    rl_result: dict,
) -> tuple[str, str]:
    date_dir, hour_dir, _, json_dir, _ = _record_dirs_for_time_bucket(record_dt, create=True)
    json_name = f"{base}.json"
    payload = _to_builtin(
        {
            "record_id": base,
            "record_bucket": {"date": date_dir, "hour": hour_dir},
            "created_at": created_at,
            "source_filename": source_filename,
            "client_ip": client_ip,
            "client_id": client_id,
            "elapsed_ms": round(float(elapsed_ms), 2),
            "stored_image": image_rel_path,
            "manual_stats": manual_stats,
            "goal_success": goal_success or {},
            "ocr_result": ocr_result,
            "ui_state": ui_state,
            "rl": rl_result,
        }
    )
    (json_dir / json_name).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return json_name, f"records/{date_dir}/{hour_dir}/json/{json_name}"


def _persist_bug_report(
    *,
    client_ip: str,
    client_id: Optional[str],
    message: str,
    record_id: Optional[str],
    user_agent: Optional[str],
) -> str:
    dt = datetime.now(timezone.utc).astimezone()
    date_dir, hour_dir, _, _, message_dir = _record_dirs_for_time_bucket(dt, create=True)
    created_at = dt.isoformat()
    stamp = dt.strftime("%Y-%m-%d_%H-%M-%S_%f")
    short = hashlib.sha256(f"{client_ip}|{created_at}|{message}".encode("utf-8", errors="ignore")).hexdigest()[:8]
    json_name = f"{stamp}_{short}.json"
    payload = {
        "created_at": created_at,
        "record_id": str(record_id or "").strip() or None,
        "client_ip": str(client_ip),
        "client_id": str(client_id) if client_id else None,
        "user_agent": str(user_agent or ""),
        "message": str(message),
    }
    (message_dir / json_name).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return f"records/{date_dir}/{hour_dir}/message/{json_name}"


def _parse_count(count_text: Optional[str]) -> tuple[Optional[int], Optional[int]]:
    if not count_text:
        return None, None
    m = re.search(r"(\d+)\s*/\s*(\d+)", str(count_text))
    if not m:
        return None, None
    return int(m.group(1)), int(m.group(2))


def _extract_signed_int(text: str) -> Optional[int]:
    s = str(text or "")
    m = re.search(r"[+\-]\s*\d+", s)
    if m:
        return int(m.group().replace(" ", ""))
    m = re.search(r"\d+", s)
    if m:
        return int(m.group())
    return None


def _get_parser():
    global _parser_instance
    if _parser_instance is None:
        from gemmini_vision.parser import GameStateParser

        _parser_instance = GameStateParser()
    return _parser_instance


class HomographyRoiRuntime:
    def __init__(self, *, template_path: Path) -> None:
        from gemmini_vision.detect import ANNOTATION

        self.annotation = ANNOTATION
        self.template_path = Path(template_path)
        if not self.template_path.exists():
            raise FileNotFoundError(f"homography template not found: {self.template_path}")

        ref_bgr = cv2.imread(str(self.template_path), cv2.IMREAD_COLOR)
        if ref_bgr is None:
            raise FileNotFoundError(f"homography template unreadable: {self.template_path}")

        self.base_w = int(self.annotation.get("width") or 0)
        self.base_h = int(self.annotation.get("height") or 0)
        if self.base_w <= 0 or self.base_h <= 0:
            raise ValueError("invalid template base size")

        ref_gray = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2GRAY)
        self.orb = cv2.ORB_create(nfeatures=int(HOMOGRAPHY_NFEATURES))
        kp, desc = self.orb.detectAndCompute(ref_gray, None)
        if desc is None or kp is None or len(kp) < 50:
            raise RuntimeError("homography template has too few features")

        self.ref_kp = kp
        self.ref_desc = desc
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING)

    @staticmethod
    def _roi_corners(box: dict) -> np.ndarray:
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

    def extract_rois(self, frame_bgr: np.ndarray) -> tuple[Optional[Dict[str, np.ndarray]], dict]:
        debug: dict = {
            "ok": False,
            "template_path": str(self.template_path),
            "nfeatures": int(HOMOGRAPHY_NFEATURES),
        }
        if not isinstance(frame_bgr, np.ndarray) or frame_bgr.size == 0:
            debug["reason"] = "empty_frame"
            return None, debug

        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        kp2, des2 = self.orb.detectAndCompute(gray, None)
        debug["kp_target"] = int(len(kp2) if kp2 is not None else 0)
        if des2 is None or kp2 is None or len(kp2) < 20:
            debug["reason"] = "no_target_features"
            return None, debug

        knn = self.bf.knnMatch(self.ref_desc, des2, k=2)
        good = []
        ratio = float(HOMOGRAPHY_RATIO)
        for m, n in knn:
            if m.distance < ratio * n.distance:
                good.append(m)
        debug["good_matches"] = int(len(good))
        if len(good) < int(HOMOGRAPHY_MIN_GOOD_MATCHES):
            debug["reason"] = "few_good_matches"
            return None, debug

        good = sorted(good, key=lambda m: m.distance)[: int(HOMOGRAPHY_TOP_N)]
        debug["used_matches"] = int(len(good))
        src_pts = np.float32([self.ref_kp[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, float(HOMOGRAPHY_RANSAC_THRESH))
        if H is None or mask is None:
            debug["reason"] = "homography_failed"
            return None, debug

        inliers = int(mask.ravel().sum())
        debug["inliers"] = int(inliers)
        if inliers < int(HOMOGRAPHY_MIN_INLIERS):
            debug["reason"] = "low_inliers"
            return None, debug

        # Project the reference frame corners to estimate a content rect in the target frame.
        try:
            corners = np.array(
                [[0, 0], [self.base_w, 0], [self.base_w, self.base_h], [0, self.base_h]],
                dtype=np.float32,
            ).reshape(-1, 1, 2)
            proj = cv2.perspectiveTransform(corners, H)
            xs = proj[:, 0, 0]
            ys = proj[:, 0, 1]
            debug["content_rect"] = {
                "x1": float(xs.min()),
                "y1": float(ys.min()),
                "x2": float(xs.max()),
                "y2": float(ys.max()),
            }
        except Exception:
            debug["content_rect"] = None

        # Extract each ROI by projecting the reference ROI boxes onto the target frame.
        rois: Dict[str, np.ndarray] = {}
        h, w = frame_bgr.shape[:2]
        pad = int(HOMOGRAPHY_ROI_PAD_PX)
        for box in list(self.annotation.get("boxes") or []):
            label = str(box.get("label") or "").strip()
            if not label:
                continue
            pts = self._roi_corners(box)
            proj = cv2.perspectiveTransform(pts, H)
            xs = proj[:, 0, 0]
            ys = proj[:, 0, 1]
            x1 = int(np.floor(xs.min())) - pad
            y1 = int(np.floor(ys.min())) - pad
            x2 = int(np.ceil(xs.max())) + pad
            y2 = int(np.ceil(ys.max())) + pad
            x1 = max(0, min(w - 1, x1))
            y1 = max(0, min(h - 1, y1))
            x2 = max(0, min(w, x2))
            y2 = max(0, min(h, y2))
            if x2 - x1 < 8 or y2 - y1 < 8:
                continue
            rois[label] = frame_bgr[y1:y2, x1:x2].copy()

        missing = [x for x in ROI_REQUIRED_LABELS if x not in rois]
        debug["roi_labels"] = sorted(list(rois.keys()))
        debug["roi_missing"] = missing
        if missing:
            debug["reason"] = "missing_rois"
            return None, debug

        debug["ok"] = True
        return rois, debug


def _get_homography_runtime() -> tuple[Optional[HomographyRoiRuntime], Optional[str]]:
    global _homography_runtime, _homography_runtime_error
    if not HOMOGRAPHY_ENABLED:
        return None, "disabled"
    if _homography_runtime is not None:
        return _homography_runtime, None
    if _homography_runtime_error is not None:
        return None, _homography_runtime_error

    with _homography_lock:
        if _homography_runtime is not None:
            return _homography_runtime, None
        if _homography_runtime_error is not None:
            return None, _homography_runtime_error
        try:
            _homography_runtime = HomographyRoiRuntime(template_path=HOMOGRAPHY_TEMPLATE_PATH)
            return _homography_runtime, None
        except Exception as e:
            _homography_runtime_error = f"{type(e).__name__}: {e}"
            return None, _homography_runtime_error


class RLRuntime:
    def __init__(self) -> None:
        from sb3_contrib import MaskablePPO
        from gem_core.role_env import (
            ArcGridGemRoleEnv,
            GEM_ALLOWED_SUBOPTS,
            GEM_TYPE_TO_ID,
            OPTION_TYPE_TO_ID,
            ProcessingOption,
            ROLE_IMPORTANCE,
            ROLE_TO_ID,
            SUBOPT_TO_ID,
        )
        import torch

        self.torch = torch
        self.ProcessingOption = ProcessingOption
        self.OPTION_TYPE_TO_ID = OPTION_TYPE_TO_ID
        self.ROLE_TO_ID = ROLE_TO_ID
        self.GEM_TYPE_TO_ID = GEM_TYPE_TO_ID
        self.SUBOPT_TO_ID = SUBOPT_TO_ID
        self.GEM_ALLOWED_SUBOPTS = GEM_ALLOWED_SUBOPTS
        self.ROLE_IMPORTANCE = ROLE_IMPORTANCE
        self.ArcGridGemRoleEnv = ArcGridGemRoleEnv
        self.env = ArcGridGemRoleEnv(shaping=False)
        # Lazily allocated pool for batched Monte Carlo rollouts.
        self._mc_envs: List[Any] = []

        model_path = _resolve_rl_model_path()
        self.model_path = model_path
        self.model = MaskablePPO.load(str(model_path), env=self.env)
        try:
            self.model_device = str(self.model.device)
        except Exception:
            self.model_device = "unknown"

    def _noop(self, tag: str = "noop_keep"):
        return self.ProcessingOption(weight=1.0, effect={}, exclude_condition=lambda _s: False, tag=tag)

    @staticmethod
    def _has_any(text: str, tokens: List[str]) -> bool:
        t = text.lower()
        return any(x in t for x in tokens)

    @staticmethod
    def _safe_int(value: Any, default: int = 0) -> int:
        try:
            return int(value)
        except Exception:
            return default

    def _parse_role_id(self, ui_state: dict) -> int:
        role = str((ui_state or {}).get("role") or "dealer").strip().lower()
        if role not in self.ROLE_TO_ID:
            role = "dealer"
        return int(self.ROLE_TO_ID[role])

    def _parse_gem_type_id(self, ui_state: dict) -> int:
        gem_type = str((ui_state or {}).get("gem_type") or "stable").strip().lower()
        if gem_type not in self.GEM_TYPE_TO_ID:
            gem_type = "stable"
        return int(self.GEM_TYPE_TO_ID[gem_type])

    @staticmethod
    def _norm_text(value: Any) -> str:
        return str(value or "").strip().lower().replace(" ", "")

    def _infer_subopt_id_from_item(self, item: Any) -> Optional[int]:
        if not isinstance(item, dict):
            blob = self._norm_text(item)
        else:
            blob = self._norm_text(
                " ".join(
                    [
                        str(item.get("raw_option") or ""),
                        str(item.get("option") or ""),
                        str(item.get("text") or ""),
                        str(item.get("formatted") or ""),
                    ]
                )
            )

        if not blob:
            return None

        # Keep longer/specific patterns first.
        token_to_name = (
            ("아군공격력강화", "ally_attack_boost"),
            ("아군공격강화", "ally_attack_boost"),
            ("ally_attack_boost", "ally_attack_boost"),
            ("아군피해강화", "ally_damage_boost"),
            ("ally_damage_boost", "ally_damage_boost"),
            ("추가피해", "additional_damage"),
            ("additional_damage", "additional_damage"),
            ("보스피해", "boss_damage"),
            ("boss_damage", "boss_damage"),
            ("낙인력", "stigma"),
            ("stigma", "stigma"),
            ("공격력", "attack"),
            ("attack", "attack"),
        )
        for token, name in token_to_name:
            if token in blob and name in self.SUBOPT_TO_ID:
                return int(self.SUBOPT_TO_ID[name])
        return None

    def _fallback_kind_pair(self, *, role_id: int, gem_type_id: int) -> tuple[int, int]:
        allowed = list(self.GEM_ALLOWED_SUBOPTS[int(gem_type_id)])
        if len(allowed) < 2:
            return 0, 1
        weights = self.ROLE_IMPORTANCE.get(int(role_id), {})
        ranked = sorted(allowed, key=lambda x: (-float(weights.get(int(x), 0.0)), int(x)))
        return int(ranked[0]), int(ranked[1])

    def _infer_effect_kinds(self, *, option_items: List[Any], role_id: int, gem_type_id: int) -> tuple[int, int]:
        allowed = set(int(x) for x in self.GEM_ALLOWED_SUBOPTS[int(gem_type_id)])
        e1_kind: Optional[int] = None
        e2_kind: Optional[int] = None

        for raw in option_items:
            if not isinstance(raw, dict):
                continue
            category = str(raw.get("category") or "").strip().lower()
            position = str(raw.get("position") or "").strip().lower()
            text_blob = self._norm_text(" ".join([raw.get("text") or "", raw.get("formatted") or ""]))
            if category == "effect1" and ("부옵션2" in text_blob or position == "right"):
                category = "effect2"
            elif category == "effect2" and ("부옵션1" in text_blob or position == "left"):
                category = "effect1"
            if category not in ("effect1", "effect2"):
                continue
            sid = self._infer_subopt_id_from_item(raw)
            if sid is None or sid not in allowed:
                continue
            if category == "effect1":
                e1_kind = int(sid)
            elif category == "effect2":
                e2_kind = int(sid)

        default1, default2 = self._fallback_kind_pair(role_id=role_id, gem_type_id=gem_type_id)
        if e1_kind is None:
            e1_kind = default1
        if e2_kind is None:
            e2_kind = default2 if default2 != e1_kind else default1

        if e1_kind == e2_kind:
            for sid in sorted(allowed):
                if sid != e1_kind:
                    e2_kind = int(sid)
                    break

        if e1_kind not in allowed:
            e1_kind = default1
        if e2_kind not in allowed or e2_kind == e1_kind:
            e2_kind = default2 if default2 != e1_kind else e2_kind

        return int(e1_kind), int(e2_kind)

    def _option_from_ui_item(self, item: Any):
        if not isinstance(item, dict):
            text = str(item or "")
            value = _extract_signed_int(text)
            item = {"category": "", "value": value if value is not None else 0, "text": text}

        category = str(item.get("category") or "").strip().lower()
        value = self._safe_int(item.get("value"), 0)
        text = str(item.get("text") or item.get("raw_option") or item.get("option") or "").strip()
        blob = " ".join(
            [
                text,
                str(item.get("raw_option") or ""),
                str(item.get("option") or ""),
                str(item.get("formatted") or ""),
            ]
        ).lower()

        # Stat options: value is always the direct delta.
        if category == "willpower":
            return self.ProcessingOption(1.0, {"willpower": int(value)}, lambda _s: False, "willpower")
        if category == "points":
            return self.ProcessingOption(1.0, {"points": int(value)}, lambda _s: False, "points")
        if category == "effect1":
            if value == 0 or self._has_any(blob, ["change", "변경"]):
                return self._noop("change_effect1")
            return self.ProcessingOption(1.0, {"effect1_level": int(value)}, lambda _s: False, "effect1_level")
        if category == "effect2":
            if value == 0 or self._has_any(blob, ["change", "변경"]):
                return self._noop("change_effect2")
            return self.ProcessingOption(1.0, {"effect2_level": int(value)}, lambda _s: False, "effect2_level")

        # Special options.
        if category == "special":
            # Prefer value-based decoding for robustness against OCR text encoding noise.
            if abs(int(value)) in (1, 2):
                rerolls = max(1, min(2, abs(int(value))))
                return self.ProcessingOption(1.0, {"rerolls": rerolls}, lambda _s: False, "rerolls")
            if self._has_any(blob, ["reroll", "다른 항목", "항목 보기"]):
                rerolls = abs(int(value)) if int(value) != 0 else (_extract_signed_int(text) or 1)
                rerolls = max(1, min(2, int(rerolls)))
                return self.ProcessingOption(1.0, {"rerolls": rerolls}, lambda _s: False, "rerolls")
            if self._has_any(blob, ["cost", "비용", "가공 비용"]) or abs(int(value)) == 100:
                delta = 1 if int(value) >= 0 else -1
                if value == 0:
                    if self._has_any(blob, ["-", "감소", "down"]):
                        delta = -1
                    else:
                        delta = 1
                return self.ProcessingOption(1.0, {"gold_state": delta}, lambda _s: False, "gold_state")
            return self._noop("noop_keep")

        # Last-resort text parsing for unknown category.
        if self._has_any(blob, ["willpower", "의지력"]):
            return self.ProcessingOption(1.0, {"willpower": int(value)}, lambda _s: False, "willpower")
        if self._has_any(blob, ["points", "질서", "혼돈", "포인트"]):
            return self.ProcessingOption(1.0, {"points": int(value)}, lambda _s: False, "points")
        if self._has_any(blob, ["effect1", "부옵션1"]):
            if value == 0:
                return self._noop("change_effect1")
            return self.ProcessingOption(1.0, {"effect1_level": int(value)}, lambda _s: False, "effect1_level")
        if self._has_any(blob, ["effect2", "부옵션2"]):
            if value == 0:
                return self._noop("change_effect2")
            return self.ProcessingOption(1.0, {"effect2_level": int(value)}, lambda _s: False, "effect2_level")
        if self._has_any(blob, ["reroll", "다른 항목", "항목 보기"]):
            rerolls = abs(int(value)) if int(value) != 0 else (_extract_signed_int(text) or 1)
            rerolls = max(1, min(2, int(rerolls)))
            return self.ProcessingOption(1.0, {"rerolls": rerolls}, lambda _s: False, "rerolls")
        if self._has_any(blob, ["cost", "비용", "가공 비용"]):
            delta = 1 if int(value) >= 0 else -1
            return self.ProcessingOption(1.0, {"gold_state": delta}, lambda _s: False, "gold_state")

        return self._noop("noop_keep")

    def recommend(self, ocr_result: dict, ui_state: dict, manual_stats: dict) -> dict:
        count_left, count_right = _parse_count(ocr_result.get("count"))

        attempts_left = count_left if count_left is not None else self._safe_int(ui_state.get("attempts_left"), 0)
        attempts_left = max(0, attempts_left)

        if count_right in (7, 9):
            max_attempts = int(count_right)
        else:
            max_attempts = 9 if attempts_left > 7 else 7
        if attempts_left > max_attempts:
            max_attempts = attempts_left

        initial_rerolls = 1 if max_attempts <= 7 else 2
        rerolls = self._safe_int(ui_state.get("rerolls"), initial_rerolls)
        rerolls = max(0, rerolls)

        # UI uses -1/0/1, env uses 0/1/2.
        cost_state_ui = self._safe_int(ui_state.get("cost_state"), 0)
        env_gold_state = max(0, min(2, cost_state_ui + 1))

        role_id = self._parse_role_id(ui_state)
        gem_type_id = self._parse_gem_type_id(ui_state)

        option_items = ui_state.get("options") if isinstance(ui_state, dict) else []
        option_items = option_items if isinstance(option_items, list) else []
        effect1_kind, effect2_kind = self._infer_effect_kinds(
            option_items=option_items,
            role_id=role_id,
            gem_type_id=gem_type_id,
        )

        state = {
            "willpower": int(manual_stats["willpower"]),
            "points": int(manual_stats["points"]),
            "effect1_level": int(manual_stats["effect1_level"]),
            "effect2_level": int(manual_stats["effect2_level"]),
            "attemptsLeft": int(attempts_left),
            "rerolls": int(rerolls),
            "gold_state": int(env_gold_state),
            "role_id": int(role_id),
            "gem_type_id": int(gem_type_id),
            "effect1_kind": int(effect1_kind),
            "effect2_kind": int(effect2_kind),
        }

        options = [self._option_from_ui_item(item) for item in option_items[:4]]
        while len(options) < 4:
            options.append(self._noop("noop_keep"))

        # IMPORTANT: ArcGridGemEnv._generate_options() sorts sampled options by type id before
        # encoding them into observations (to stabilize ordering). The RL policy and the
        # goal_success model were trained on that canonical order; keep inference consistent.
        def _opt_type_id(opt) -> int:
            try:
                effect = getattr(opt, "effect", None) or {}
                if effect:
                    k = next(iter(effect.keys()))
                    return int(self.OPTION_TYPE_TO_ID.get(k, 999))
                tag = str(getattr(opt, "tag", "") or "")
                return int(self.OPTION_TYPE_TO_ID.get(tag, 999))
            except Exception:
                return 999

        options.sort(key=_opt_type_id)

        self.env.max_attempts = int(max_attempts)
        self.env.initial_rerolls = int(initial_rerolls)
        self.env.state = state
        self.env.current_options = options
        self.env.terminated = False
        self.env.truncated = False
        self.env.first_process_done = state["attemptsLeft"] < self.env.max_attempts

        obs = self.env._get_obs()
        masks = self.env.action_masks()

        if not any(masks):
            return {
                "action": None,
                "action_name": None,
                "confidence": 0.0,
                "action_mask": masks,
                "action_probs": [],
                "state_used": state,
            }

        action, _ = self.model.predict(obs, action_masks=masks, deterministic=True)
        action = int(action)

        obs_t = self.torch.as_tensor(obs, dtype=self.torch.float32).unsqueeze(0).to(self.model.device)
        mask_t = self.torch.as_tensor(masks, dtype=self.torch.bool).unsqueeze(0).to(self.model.device)
        with self.torch.no_grad():
            dist = self.model.policy.get_distribution(obs_t, action_masks=mask_t)
            probs = dist.distribution.probs[0].detach().cpu().numpy()

        action_probs = []
        for idx, prob in enumerate(probs.tolist()):
            if masks[idx]:
                action_probs.append({"action": idx, "action_name": ACTION_NAME.get(idx, str(idx)), "prob": float(prob)})
        action_probs.sort(key=lambda x: x["prob"], reverse=True)

        confidence = float(probs[action]) if 0 <= action < len(probs) else 0.0
        return {
            "action": action,
            "action_name": ACTION_NAME.get(action, str(action)),
            "confidence": confidence,
            "action_mask": masks,
            "action_probs": action_probs,
            "state_used": state,
        }

    def _derive_mc_seed(
        self,
        *,
        goal: dict,
        base_state: dict,
        base_opts: list,
        max_attempts: int,
        initial_rerolls: int,
        forced_first_action: Optional[int],
    ) -> int:
        h = hashlib.sha256()
        try:
            h.update(np.asarray(self.env._get_obs(), dtype=np.float32).tobytes())
        except Exception:
            h.update(repr(base_state).encode("utf-8", errors="ignore"))
            h.update(repr(base_opts).encode("utf-8", errors="ignore"))
        h.update(json.dumps(goal, sort_keys=True).encode("utf-8", errors="ignore"))
        h.update(str(max_attempts).encode("ascii"))
        h.update(str(initial_rerolls).encode("ascii"))
        if forced_first_action is not None:
            h.update(f"forced_first_action:{int(forced_first_action)}".encode("ascii"))
        return int(h.hexdigest()[:8], 16)

    def _estimate_goal_success_mc_internal(
        self,
        goal: dict,
        *,
        n_rollouts: int,
        seed: Optional[int],
        forced_first_action: Optional[int],
    ) -> dict:
        n = int(n_rollouts)
        if n < 1:
            raise ValueError("n_rollouts must be >= 1")

        base_state = dict(self.env.state or {})
        base_opts = list(self.env.current_options or [])
        max_attempts = int(getattr(self.env, "max_attempts", 9))
        initial_rerolls = int(getattr(self.env, "initial_rerolls", 2))

        if seed is None:
            seed = self._derive_mc_seed(
                goal=goal,
                base_state=base_state,
                base_opts=base_opts,
                max_attempts=max_attempts,
                initial_rerolls=initial_rerolls,
                forced_first_action=forced_first_action,
            )
        seed = int(seed)

        if len(self._mc_envs) < n:
            self._mc_envs.extend([self.ArcGridGemRoleEnv(seed=0, shaping=False) for _ in range(n - len(self._mc_envs))])
        envs = self._mc_envs[:n]

        obs_cache: List[np.ndarray] = [np.zeros((64,), dtype=np.float32) for _ in range(n)]

        # Reset all envs to the same snapshot, with different RNG seeds.
        for i, env in enumerate(envs):
            env.shaping = False
            env.max_attempts = max_attempts
            env.initial_rerolls = initial_rerolls
            env.state = base_state.copy()
            env.current_options = list(base_opts)
            env.terminated = False
            env.truncated = False
            # IMPORTANT: env instances are reused across requests; reset step counters
            # to avoid hitting max_decision_steps and truncating immediately (false 0%).
            try:
                env.decision_steps = 0
            except Exception:
                pass
            env.first_process_done = env.state.get("attemptsLeft", env.max_attempts) < env.max_attempts
            env.rng = np.random.default_rng(seed + i)
            obs_cache[i] = np.asarray(env._get_obs(), dtype=np.float32)

        # Optional first action probe: force the first action, then follow policy.
        if forced_first_action is not None:
            fa = int(forced_first_action)
            for i, env in enumerate(envs):
                if env.terminated or env.truncated:
                    continue
                masks = env.action_masks()
                if not (0 <= fa < len(masks)) or not masks[fa]:
                    env.terminated = True
                    continue
                obs_next, _r, _term, _trunc, _info = env.step(fa)
                obs_cache[i] = np.asarray(obs_next, dtype=np.float32)

        # Rollout all envs in lock-step with batched forward.
        while True:
            active: List[int] = []
            obs_batch: List[np.ndarray] = []
            mask_batch: List[List[bool]] = []

            for idx, env in enumerate(envs):
                if env.terminated or env.truncated:
                    continue
                masks = env.action_masks()
                if not any(masks):
                    env.terminated = True
                    continue
                active.append(idx)
                obs_batch.append(obs_cache[idx])
                mask_batch.append(masks)

            if not active:
                break

            obs_b = np.asarray(obs_batch, dtype=np.float32)
            mask_b = np.asarray(mask_batch, dtype=bool)
            actions, _ = self.model.predict(obs_b, action_masks=mask_b, deterministic=True)
            actions = np.asarray(actions).reshape(-1)

            for j, env_idx in enumerate(active):
                obs_next, _r, _term, _trunc, _info = envs[env_idx].step(int(actions[j]))
                obs_cache[env_idx] = np.asarray(obs_next, dtype=np.float32)

        gw = int(goal["willpower"])
        gp = int(goal["points"])
        ge1 = int(goal["effect1_level"])
        ge2 = int(goal["effect2_level"])

        ok = 0
        for env in envs:
            s = env.state
            if (
                int(s.get("willpower", 0)) >= gw
                and int(s.get("points", 0)) >= gp
                and int(s.get("effect1_level", 0)) >= ge1
                and int(s.get("effect2_level", 0)) >= ge2
            ):
                ok += 1

        p = ok / float(n)
        return {
            "success_prob": float(p),
            "successes": int(ok),
            "n_rollouts": int(n),
            "seed": int(seed),
            "forced_first_action": (int(forced_first_action) if forced_first_action is not None else None),
        }

    def estimate_goal_success_mc(
        self,
        goal: dict,
        *,
        n_rollouts: int = 256,
        seed: Optional[int] = None,
    ) -> dict:
        """
        Policy-following MC success probability from current snapshot.
        IMPORTANT: call this after recommend(), so env.state/env.current_options are set.
        """
        return self._estimate_goal_success_mc_internal(
            goal,
            n_rollouts=int(n_rollouts),
            seed=seed,
            forced_first_action=None,
        )

    def estimate_goal_success_mc_by_first_action(
        self,
        goal: dict,
        *,
        n_rollouts: int = 128,
        seed: Optional[int] = None,
    ) -> dict:
        """
        Probe success probability by forcing each valid first action once, then following policy.
        Used as a guardrail for obvious false-zero cases.
        """
        masks = list(self.env.action_masks())
        stats: List[dict] = []

        for action in range(3):
            if not (0 <= action < len(masks)) or not masks[action]:
                continue
            per_seed = (None if seed is None else int(seed) + (action * 1_000_003))
            r = self._estimate_goal_success_mc_internal(
                goal,
                n_rollouts=int(n_rollouts),
                seed=per_seed,
                forced_first_action=int(action),
            )
            stats.append(
                {
                    "action": int(action),
                    "action_name": ACTION_NAME.get(int(action), str(action)),
                    "success_prob": float(r["success_prob"]),
                    "successes": int(r["successes"]),
                    "n_rollouts": int(r["n_rollouts"]),
                    "seed": int(r["seed"]),
                }
            )

        stats.sort(key=lambda x: float(x["success_prob"]), reverse=True)
        best = stats[0] if stats else None
        return {
            "action_stats": stats,
            "best_action": (int(best["action"]) if best is not None else None),
            "best_action_name": (str(best["action_name"]) if best is not None else None),
            "best_success_prob": (float(best["success_prob"]) if best is not None else 0.0),
            "n_rollouts": int(n_rollouts),
        }

    def estimate_goal_success_mc_with_strategy(
        self,
        goal: dict,
        *,
        n_rollouts: int = 128,
        strategy: str = "always_process",
        seed: Optional[int] = None,
    ) -> dict:
        """
        Non-policy fallback MC for false-zero guardrails.
        strategy:
          - always_process: prefer process > reroll > stop
          - random_valid: sample uniformly from valid actions
        """
        n = int(n_rollouts)
        if n < 1:
            raise ValueError("n_rollouts must be >= 1")

        strategy_name = str(strategy or "always_process").strip().lower()
        if strategy_name not in {"always_process", "random_valid"}:
            raise ValueError(f"unknown strategy: {strategy_name}")

        base_state = dict(self.env.state or {})
        base_opts = list(self.env.current_options or [])
        max_attempts = int(getattr(self.env, "max_attempts", 9))
        initial_rerolls = int(getattr(self.env, "initial_rerolls", 2))

        if seed is None:
            base_seed = self._derive_mc_seed(
                goal=goal,
                base_state=base_state,
                base_opts=base_opts,
                max_attempts=max_attempts,
                initial_rerolls=initial_rerolls,
                forced_first_action=None,
            )
            h = hashlib.sha256()
            h.update(str(base_seed).encode("ascii"))
            h.update(strategy_name.encode("ascii"))
            seed = int(h.hexdigest()[:8], 16)
        seed = int(seed)

        if len(self._mc_envs) < n:
            self._mc_envs.extend([self.ArcGridGemRoleEnv(seed=0, shaping=False) for _ in range(n - len(self._mc_envs))])
        envs = self._mc_envs[:n]

        for i, env in enumerate(envs):
            env.shaping = False
            env.max_attempts = max_attempts
            env.initial_rerolls = initial_rerolls
            env.state = base_state.copy()
            env.current_options = list(base_opts)
            env.terminated = False
            env.truncated = False
            try:
                env.decision_steps = 0
            except Exception:
                pass
            env.first_process_done = env.state.get("attemptsLeft", env.max_attempts) < env.max_attempts
            env.rng = np.random.default_rng(seed + i)

        while True:
            active = False
            for env in envs:
                if env.terminated or env.truncated:
                    continue
                masks = list(env.action_masks())
                valid = [idx for idx, ok in enumerate(masks) if ok]
                if not valid:
                    env.terminated = True
                    continue

                active = True
                if strategy_name == "always_process":
                    if masks[0]:
                        action = 0
                    elif masks[1]:
                        action = 1
                    else:
                        action = 2
                else:
                    action = int(env.rng.choice(valid))
                env.step(int(action))

            if not active:
                break

        gw = int(goal["willpower"])
        gp = int(goal["points"])
        ge1 = int(goal["effect1_level"])
        ge2 = int(goal["effect2_level"])

        ok = 0
        for env in envs:
            s = env.state
            if (
                int(s.get("willpower", 0)) >= gw
                and int(s.get("points", 0)) >= gp
                and int(s.get("effect1_level", 0)) >= ge1
                and int(s.get("effect2_level", 0)) >= ge2
            ):
                ok += 1

        p = ok / float(n)
        return {
            "strategy": strategy_name,
            "success_prob": float(p),
            "successes": int(ok),
            "n_rollouts": int(n),
            "seed": int(seed),
        }


class GoalSuccessRuntime:
    """
    Predict goal success probability using a supervised model (goal_success/runs/dist625/best.pt).

    The model predicts a 625-bin distribution over final (willpower, points, effect1_level, effect2_level),
    each in [1..5], by following the current RL policy from a given snapshot.
    """

    def __init__(self) -> None:
        import torch
        import torch.nn as nn

        self.torch = torch
        self.nn = nn
        self.device = "cpu"

        model_path = GOAL_SUCCESS_MODEL_PATH if GOAL_SUCCESS_MODEL_PATH.exists() else GOAL_SUCCESS_MODEL_PATH_FALLBACK
        if not model_path.exists():
            raise FileNotFoundError(
                f"Goal success model not found: {GOAL_SUCCESS_MODEL_PATH} (or fallback {GOAL_SUCCESS_MODEL_PATH_FALLBACK})"
            )

        self.model_path = model_path
        ckpt = torch.load(str(model_path), map_location="cpu")
        run_meta = ckpt.get("run_meta") or {}
        hidden = int(run_meta.get("hidden", 256))
        depth = int(run_meta.get("depth", 2))
        dropout = float(run_meta.get("dropout", 0.0))

        self.model = self._build_mlp(in_dim=47, hidden=hidden, depth=depth, dropout=dropout).to(self.device)
        self.model.load_state_dict(ckpt["model_state"])
        self.model.eval()

    def _build_mlp(self, in_dim: int, hidden: int, depth: int, dropout: float):
        layers = []
        dim = int(in_dim)
        for _ in range(max(1, int(depth))):
            layers.append(self.nn.Linear(dim, int(hidden)))
            layers.append(self.nn.ReLU())
            if float(dropout) > 0:
                layers.append(self.nn.Dropout(float(dropout)))
            dim = int(hidden)
        layers.append(self.nn.Linear(dim, 625))
        return self.nn.Sequential(*layers)

    @staticmethod
    def _final_state_to_index(w: int, p: int, e1: int, e2: int) -> int:
        # each in [1..5]
        return ((w - 1) * 125) + ((p - 1) * 25) + ((e1 - 1) * 5) + (e2 - 1)

    @staticmethod
    def _normalize_obs(obs: np.ndarray) -> np.ndarray:
        """
        Mirror goal_success/train_dist625.py::_normalize_obs

        obs shape (47,) from ArcGridGemEnv._get_obs():
          base = [w,p,e1,e2,attemptsLeft,rerolls,gold_state] + 4*(onehot9 + value1)
        """
        # IMPORTANT: avoid mutating caller-owned arrays (obs is often reused by debuggers/benchmarks).
        x = np.asarray(obs, dtype=np.float32).reshape(-1).copy()
        if x.shape[0] != 47:
            raise ValueError(f"expected obs shape (47,), got {x.shape}")

        # stats 1..5 -> 0..1
        x[0:4] = (x[0:4] - 1.0) / 4.0
        # attemptsLeft up to 9 -> 0..1
        x[4] = x[4] / 9.0
        # rerolls up to 2 -> 0..1
        x[5] = x[5] / 2.0
        # gold_state 0..2 -> 0..1
        x[6] = x[6] / 2.0

        # option values: last element of each 10-d block; typical magnitude <= 4 (or -1)
        off = 7
        for _ in range(4):
            x[off + 9] = x[off + 9] / 4.0
            off += 10

        return x

    def predict_success_prob(self, obs: np.ndarray, goal: dict) -> float:
        gw = int(goal["willpower"])
        gp = int(goal["points"])
        ge1 = int(goal["effect1_level"])
        ge2 = int(goal["effect2_level"])

        x = self._normalize_obs(obs).reshape(1, 47)
        xt = self.torch.from_numpy(x).to(device=self.device, dtype=self.torch.float32)

        with self.torch.no_grad():
            logits = self.model(xt)
            probs = self.torch.softmax(logits, dim=-1)[0].detach().cpu().numpy()

        p_success = 0.0
        for w in range(gw, 6):
            for p in range(gp, 6):
                for e1 in range(ge1, 6):
                    for e2 in range(ge2, 6):
                        p_success += float(probs[self._final_state_to_index(w, p, e1, e2)])
        return float(max(0.0, min(1.0, p_success)))


def _get_rl_runtime():
    global _rl_runtime
    if _rl_runtime is None:
        _rl_runtime = RLRuntime()
    return _rl_runtime


def _get_goal_success_runtime():
    global _goal_success_runtime
    if _goal_success_runtime is None:
        _goal_success_runtime = GoalSuccessRuntime()
    return _goal_success_runtime


@app.get("/")
async def index() -> FileResponse:
    # Local/dev convenience: serve the Vercel frontend if present so we can
    # verify UI changes without redeploying.
    if FRONTEND_INDEX.exists():
        return FileResponse(FRONTEND_INDEX)
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/health")
async def health() -> dict:
    return {"ok": True}


@app.post("/api/report")
async def submit_report(
    request: Request,
    message: str = Form(...),
    record_id: Optional[str] = Form(None),
    client_id: Optional[str] = Form(None),
    client_context: Optional[str] = Form(None),
) -> dict:
    _verify_proxy_secret(request)
    ip = _client_ip(request)
    rate_key = _rate_limit_key(client_ip=ip, client_id=client_id)
    _check_rate_limit(rate_key)

    text = str(message or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="message is required")
    if len(text) > 4000:
        raise HTTPException(status_code=400, detail="message too long (max 4000 chars)")

    parsed_context: Any = None
    if client_context:
        try:
            parsed_context = json.loads(client_context)
        except Exception:
            parsed_context = {"raw": str(client_context)}

    rel_path = _persist_bug_report(
        client_ip=ip,
        client_id=_sanitize_rate_key(client_id) or None,
        message=text,
        record_id=record_id,
        user_agent=request.headers.get("user-agent"),
    )

    report_abs_path = (BASE_DIR / rel_path).resolve()
    attachments: Optional[dict] = None
    attach_error: Optional[str] = None
    try:
        attachments = _persist_report_attachments(
            report_abs_path=report_abs_path,
            report_rel_path=rel_path,
            client_id=client_id,
            record_id=record_id,
            client_context=parsed_context,
        )
    except Exception as e:
        attach_error = f"{type(e).__name__}: {e}"

    try:
        existing = {}
        if report_abs_path.exists():
            existing = json.loads(report_abs_path.read_text(encoding="utf-8"))
        if attachments is not None:
            existing["attachments"] = _to_builtin(attachments)
        if attach_error:
            existing["attachments_error"] = str(attach_error)
        if parsed_context is not None:
            existing["client_context"] = _to_builtin(parsed_context)
        report_abs_path.write_text(json.dumps(_to_builtin(existing), ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        # Best-effort; the bug report itself is already persisted.
        pass

    resp = {"ok": True, "message_path": rel_path}
    if attachments is not None:
        resp["attached_count"] = int(attachments.get("attached_count") or 0)
        resp["bundle"] = str(attachments.get("bundle") or "")
    if attach_error:
        resp["attachments_error"] = attach_error
    return resp


@app.post("/api/upload")
async def upload_image(
    request: Request,
    image: UploadFile = File(...),
    client_id: Optional[str] = Form(None),
) -> dict:
    _verify_proxy_secret(request)
    ip = _client_ip(request)
    rate_key = _rate_limit_key(client_ip=ip, client_id=client_id)
    _check_rate_limit(rate_key)

    filename, data, detected = await _read_valid_image(image)
    base, ext, record_dt = _build_record_name(detected)
    stored_name, stored_rel_path = _persist_image_bytes(base, ext, data, record_dt=record_dt)
    digest = hashlib.sha256(data).hexdigest()[:16]

    return {
        "ok": True,
        "filename": filename,
        "stored_as": stored_name,
        "bytes": len(data),
        "sha256_16": digest,
        "client_ip": ip,
        "record_image_path": stored_rel_path,
    }


@app.post("/api/analyze")
async def analyze_image(
    request: Request,
    image: Optional[UploadFile] = File(None),
    roi_option1: Optional[UploadFile] = File(None),
    roi_option2: Optional[UploadFile] = File(None),
    roi_option3: Optional[UploadFile] = File(None),
    roi_option4: Optional[UploadFile] = File(None),
    roi_possible: Optional[UploadFile] = File(None),
    roi_cost: Optional[UploadFile] = File(None),
    roi_count: Optional[UploadFile] = File(None),
    willpower: Optional[int] = Form(None),
    points: Optional[int] = Form(None),
    effect1_level: Optional[int] = Form(None),
    effect2_level: Optional[int] = Form(None),
    target_willpower: Optional[int] = Form(None),
    target_points: Optional[int] = Form(None),
    target_effect1_level: Optional[int] = Form(None),
    target_effect2_level: Optional[int] = Form(None),
    role: Optional[str] = Form(None),
    gem_type: Optional[str] = Form(None),
    roi_schema_version: Optional[str] = Form(None),
    client_id: Optional[str] = Form(None),
    client_debug: Optional[str] = Form(None),
) -> dict:
    _verify_proxy_secret(request)
    ip = _client_ip(request)
    rate_key = _rate_limit_key(client_ip=ip, client_id=client_id)
    _check_rate_limit(rate_key)

    t_total0 = time.perf_counter()
    timings_ms: dict[str, float] = {}

    manual_stats = {
        "willpower": _validate_stat("willpower", willpower),
        "points": _validate_stat("points", points),
        "effect1_level": _validate_stat("effect1_level", effect1_level),
        "effect2_level": _validate_stat("effect2_level", effect2_level),
    }
    target_goal = {
        "willpower": _validate_stat("target_willpower", target_willpower, default=4),
        "points": _validate_stat("target_points", target_points, default=4),
        "effect1_level": _validate_stat("target_effect1_level", target_effect1_level, default=4),
        "effect2_level": _validate_stat("target_effect2_level", target_effect2_level, default=4),
    }
    selected_role = _validate_choice("role", role, allowed={"dealer", "support"}, default="dealer")
    selected_gem_type = _validate_choice(
        "gem_type",
        gem_type,
        allowed={"stable", "solid", "immutable"},
        default="stable",
    )
    selected_roi_schema, roi_schema_supported = _normalize_roi_schema_version(roi_schema_version)
    roi_schema_warning: Optional[str] = None
    if not roi_schema_supported:
        roi_schema_warning = f"unsupported_roi_schema:{selected_roi_schema}"

    client_debug_payload: Any = None
    if client_debug:
        try:
            client_debug_payload = json.loads(client_debug)
        except Exception:
            client_debug_payload = {"raw": str(client_debug)}

    roi_uploads: Dict[str, Optional[UploadFile]] = {
        "option1": roi_option1,
        "option2": roi_option2,
        "option3": roi_option3,
        "option4": roi_option4,
        "possible": roi_possible,
        "cost": roi_cost,
        "count": roi_count,
    }
    roi_images: Dict[str, np.ndarray] = {}
    roi_meta: Dict[str, dict] = {}
    roi_raw: Dict[str, dict] = {}
    t_roi0 = time.perf_counter()
    for label, upload in roi_uploads.items():
        if upload is None:
            continue
        roi_filename, roi_data, roi_detected = await _read_valid_image(upload)
        roi_raw[label] = {
            "filename": roi_filename,
            "kind": roi_detected,
            "data": roi_data,
        }
        roi_arr = np.frombuffer(roi_data, dtype=np.uint8)
        roi_img = cv2.imdecode(roi_arr, cv2.IMREAD_COLOR)
        if roi_img is None:
            raise HTTPException(status_code=400, detail=f"Failed to decode ROI image: {label}")
        roi_images[label] = roi_img
        roi_meta[label] = {
            "filename": roi_filename,
            "detected_kind": roi_detected,
            "bytes": int(len(roi_data)),
            "shape": [int(x) for x in roi_img.shape],
        }
    timings_ms["read_decode_rois"] = (time.perf_counter() - t_roi0) * 1000.0

    roi_labels = set(roi_images.keys())
    required_roi_labels = set(ROI_REQUIRED_LABELS)
    has_complete_roi_set = required_roi_labels.issubset(roi_labels)

    filename = "roi_bundle"
    detected = "png"
    data: bytes = b""
    img: Optional[np.ndarray] = None
    stored_image_rel_path: Optional[str] = None

    if not has_complete_roi_set:
        if image is None:
            missing = sorted(required_roi_labels - roi_labels)
            raise HTTPException(
                status_code=400,
                detail=f"Missing image input. provide full image or complete ROI set: missing={missing}",
            )
        t_read0 = time.perf_counter()
        filename, data, detected = await _read_valid_image(image)
        timings_ms["read_image"] = (time.perf_counter() - t_read0) * 1000.0

    base, ext, record_dt = _build_record_name(detected)

    if data:
        t_persist0 = time.perf_counter()
        _stored_image_name, stored_image_rel_path = _persist_image_bytes(base, ext, data, record_dt=record_dt)
        timings_ms["persist_image"] = (time.perf_counter() - t_persist0) * 1000.0

        t_decode0 = time.perf_counter()
        arr = np.frombuffer(data, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        timings_ms["decode_image"] = (time.perf_counter() - t_decode0) * 1000.0
        if img is None:
            raise HTTPException(status_code=400, detail="Failed to decode image")

    t_parser0 = time.perf_counter()
    parser = _get_parser()
    timings_ms["get_parser"] = (time.perf_counter() - t_parser0) * 1000.0

    ocr_mode = "fullframe"
    homography_debug: Optional[dict] = None
    t_ocr0 = time.perf_counter()
    if has_complete_roi_set:
        ocr_mode = "multi_crop"
        ocr_result = parser.parse_ocr_rois(roi_images)
    else:
        ocr_result = {}
        if isinstance(img, np.ndarray):
            hrt, hrt_err = _get_homography_runtime()
            if hrt is not None:
                try:
                    rois2, hdbg = hrt.extract_rois(img)
                    homography_debug = _to_builtin(hdbg)
                    if rois2 is not None:
                        ocr_mode = "fullframe_homography_rois"
                        ocr_result = parser.parse_ocr_rois(rois2)
                    else:
                        ocr_result = parser.parse_ocr_result(img)
                except Exception as e:
                    homography_debug = {"ok": False, "reason": "exception", "error": f"{type(e).__name__}: {e}"}
                    ocr_result = parser.parse_ocr_result(img)
            else:
                if hrt_err:
                    homography_debug = {"ok": False, "reason": "runtime_unavailable", "error": str(hrt_err)}
                ocr_result = parser.parse_ocr_result(img)
    if not isinstance(ocr_result, dict):
        ocr_result = {}
    timings_ms["ocr_parse"] = (time.perf_counter() - t_ocr0) * 1000.0

    t_ui0 = time.perf_counter()
    try:
        ui_state = (
            parser.convert_to_ui_state(ocr_result)
            if ocr_result
            else {"options": [], "rerolls": 0, "attempts_left": 0, "cost_state": 0}
        )
    except Exception:
        ui_state = {"options": [], "rerolls": 0, "attempts_left": 0, "cost_state": 0}
    ui_state["role"] = selected_role
    ui_state["gem_type"] = selected_gem_type
    ui_state["roi_schema_version"] = selected_roi_schema
    timings_ms["ui_state_convert"] = (time.perf_counter() - t_ui0) * 1000.0

    goal_success_prob: Optional[float] = None
    goal_success_reason: str = "unknown"
    goal_success_error: Optional[str] = None
    goal_success_mc: Optional[dict] = None
    goal_success_mc_first_action: Optional[dict] = None
    goal_success_mc_strategy: Dict[str, dict] = {}

    rl_result: dict = {}
    runtime: Any = None
    goal_runtime: Any = None

    t_lock0 = time.perf_counter()
    _runtime_lock.acquire()
    timings_ms["runtime_lock_wait"] = (time.perf_counter() - t_lock0) * 1000.0
    try:
        t_rl_rt0 = time.perf_counter()
        runtime = _get_rl_runtime()
        timings_ms["get_rl_runtime"] = (time.perf_counter() - t_rl_rt0) * 1000.0

        t_rl0 = time.perf_counter()
        rl_result = runtime.recommend(ocr_result, ui_state, manual_stats)
        timings_ms["rl_recommend"] = (time.perf_counter() - t_rl0) * 1000.0

        state_used = rl_result.get("state_used") if isinstance(rl_result, dict) else None
        state_used = state_used if isinstance(state_used, dict) else {}
        attempts_left_used = int(state_used.get("attemptsLeft") or 0)
        action_name_used = str(rl_result.get("action_name") or "")

        already_success = all(
            int(state_used.get(k) or 0) >= int(target_goal[k])
            for k in ("willpower", "points", "effect1_level", "effect2_level")
        )

        # If the policy stops (or there's nothing to do), the final state is deterministic.
        if attempts_left_used <= 0 or action_name_used == "stop":
            goal_success_prob = 1.0 if already_success else 0.0
            goal_success_reason = "deterministic_final"
            timings_ms["goal_success_predict"] = 0.0
        else:
            t_goal0 = time.perf_counter()
            try:
                n_mc = int(GOAL_SUCCESS_MC_ROLLOUTS)
                goal_success_mc = runtime.estimate_goal_success_mc(target_goal, n_rollouts=n_mc)
                goal_success_prob = float(goal_success_mc["success_prob"])
                goal_success_reason = f"mc_rollout_{n_mc}"

                # Guardrail for false-zero: if policy-following MC says 0%, probe each valid first action once.
                if goal_success_prob <= 0.0 and attempts_left_used > 0:
                    t_probe0 = time.perf_counter()
                    probe_rollouts = max(64, int(n_mc) // 2)
                    goal_success_mc_first_action = runtime.estimate_goal_success_mc_by_first_action(
                        target_goal,
                        n_rollouts=probe_rollouts,
                    )
                    timings_ms["goal_success_probe_first_action"] = (time.perf_counter() - t_probe0) * 1000.0

                    best_prob = float(goal_success_mc_first_action.get("best_success_prob") or 0.0)
                    best_action = goal_success_mc_first_action.get("best_action")
                    if best_prob > 0.0 and best_action is not None:
                        goal_success_prob = best_prob
                        goal_success_reason = f"mc_best_first_action_{probe_rollouts}"
                        if isinstance(rl_result, dict):
                            rl_result["action"] = int(best_action)
                            rl_result["action_name"] = ACTION_NAME.get(int(best_action), str(best_action))
                            rl_result["confidence"] = float(best_prob)
                            stats = list(goal_success_mc_first_action.get("action_stats") or [])
                            rl_result["action_probs"] = [
                                {
                                    "action": int(x.get("action", -1)),
                                    "action_name": str(x.get("action_name") or ACTION_NAME.get(int(x.get("action", -1)), "")),
                                    "prob": float(x.get("success_prob", 0.0)),
                                }
                                for x in stats
                            ]
                            rl_result["action_overridden_by"] = "goal_success_first_action_probe"

                    # If still 0%, try non-policy fallback strategies to reduce false-zero.
                    if float(goal_success_prob or 0.0) <= 0.0:
                        for strategy in ("always_process", "random_valid"):
                            t_s0 = time.perf_counter()
                            r_strategy = runtime.estimate_goal_success_mc_with_strategy(
                                target_goal,
                                n_rollouts=probe_rollouts,
                                strategy=strategy,
                            )
                            timings_ms[f"goal_success_probe_{strategy}"] = (time.perf_counter() - t_s0) * 1000.0
                            goal_success_mc_strategy[strategy] = r_strategy
                            sp = float(r_strategy.get("success_prob") or 0.0)
                            if sp > float(goal_success_prob or 0.0):
                                goal_success_prob = sp
                                goal_success_reason = f"mc_{strategy}_{probe_rollouts}"
            except Exception as e:
                goal_success_prob = None
                goal_success_reason = "goal_success_mc_error"
                goal_success_error = f"{type(e).__name__}: {e}"
            timings_ms["goal_success_predict"] = (time.perf_counter() - t_goal0) * 1000.0

        # UX rule: if success probability is 0%, recommend stop.
        try:
            if goal_success_prob is not None and float(goal_success_prob) <= 0.0 and isinstance(rl_result, dict):
                rl_result["action"] = 2
                rl_result["action_name"] = ACTION_NAME.get(2, "stop")
                rl_result["confidence"] = 1.0
                rl_result["action_mask"] = [False, False, True]
                rl_result["action_probs"] = [{"action": 2, "action_name": ACTION_NAME.get(2, "stop"), "prob": 1.0}]
                rl_result["action_overridden_by"] = "goal_success_zero"
        except Exception:
            pass
    finally:
        _runtime_lock.release()

    elapsed_ms = (time.perf_counter() - t_total0) * 1000.0
    timings_ms["total"] = elapsed_ms

    goal_success_payload = {
        "target": target_goal,
        "success_prob": goal_success_prob,
        "reason": goal_success_reason,
        "error": goal_success_error,
        "mc": goal_success_mc or {},
        "mc_first_action": goal_success_mc_first_action or {},
        "mc_strategy": goal_success_mc_strategy,
    }

    debug = {
        "timings_ms": _to_builtin({k: round(float(v), 2) for k, v in timings_ms.items()}),
        "selection": {
            "role": selected_role,
            "gem_type": selected_gem_type,
            "roi_schema_version": selected_roi_schema,
            "roi_schema_supported": bool(roi_schema_supported),
            "ocr_mode": ocr_mode,
        },
        "image": {
            "detected_kind": detected,
            "bytes": int(len(data)),
            "shape": [int(x) for x in (img.shape if isinstance(img, np.ndarray) else [])],
        },
        "roi": {
            "provided_labels": sorted(list(roi_labels)),
            "required_labels": list(ROI_REQUIRED_LABELS),
            "complete": bool(has_complete_roi_set),
            "items": _to_builtin(roi_meta),
        },
        "env": {
            "python": str(sys.version).split()[0],
            "opencv": str(getattr(cv2, "__version__", "") or ""),
        },
        "models": {
            "ocr_gpu": None,
            "rl_model_path": str(getattr(runtime, "model_path", "") or "") if runtime is not None else "",
            "rl_model_device": str(getattr(runtime, "model_device", "") or "") if runtime is not None else "",
            "goal_success_model_path": (
                str(getattr(goal_runtime, "model_path", "") or "")
                if goal_runtime is not None
                else f"mc_rollout_{GOAL_SUCCESS_MC_ROLLOUTS}"
            ),
        },
        "goal_success": _to_builtin(goal_success_payload),
        "client_debug": _to_builtin(client_debug_payload),
    }
    if homography_debug is not None:
        debug["homography"] = _to_builtin(homography_debug)
    if roi_labels and not has_complete_roi_set and image is not None:
        warnings = list(debug.get("warnings") or [])
        missing = sorted(required_roi_labels - roi_labels)
        warnings.append(f"incomplete_roi_set_fallback_fullframe:missing={','.join(missing)}")
        debug["warnings"] = warnings
    if roi_schema_warning:
        warnings = list(debug.get("warnings") or [])
        warnings.append(roi_schema_warning)
        debug["warnings"] = warnings

    try:
        import torch

        debug["env"]["torch"] = str(getattr(torch, "__version__", "") or "")
        debug["env"]["torch_cuda"] = str(getattr(getattr(torch, "version", None), "cuda", "") or "")
        debug["env"]["cuda_available"] = bool(torch.cuda.is_available())
    except Exception:
        pass

    try:
        import easyocr

        debug["env"]["easyocr"] = str(getattr(easyocr, "__version__", "") or "")
    except Exception:
        pass

    try:
        from gemmini_vision.detect import OCR_GPU_ENABLED as _OCR_GPU_ENABLED

        debug["models"]["ocr_gpu"] = bool(_OCR_GPU_ENABLED)
    except Exception:
        debug["models"]["ocr_gpu"] = None

    created_at = record_dt.isoformat()

    _record_json_name, record_json_rel_path = _persist_analyze_record(
        base=base,
        record_dt=record_dt,
        created_at=created_at,
        image_rel_path=stored_image_rel_path,
        source_filename=filename,
        client_ip=ip,
        client_id=_sanitize_rate_key(client_id) or None,
        elapsed_ms=elapsed_ms,
        manual_stats=manual_stats,
        goal_success=goal_success_payload,
        ocr_result=ocr_result,
        ui_state=ui_state,
        rl_result=rl_result,
    )

    try:
        date_dir, hour_dir, _images_dir, _json_dir, _message_dir = _record_dirs_for_time_bucket(record_dt, create=False)
        roi_bytes = 0
        rois_to_cache: Dict[str, dict] = {}
        for label, blob in (roi_raw or {}).items():
            if not isinstance(blob, dict):
                continue
            data_blob = blob.get("data")
            if not isinstance(data_blob, (bytes, bytearray)):
                continue
            roi_bytes += int(len(data_blob))
            rois_to_cache[str(label)] = {
                "filename": str(blob.get("filename") or ""),
                "kind": str(blob.get("kind") or ""),
                "data": bytes(data_blob),
            }

        _report_cache_add(
            client_id=client_id,
            entry={
                "record_id": base,
                "record_bucket": {"date": date_dir, "hour": hour_dir},
                "json_path": record_json_rel_path,
                "image_path": stored_image_rel_path,
                "ocr_mode": ocr_mode,
                "roi_schema_version": selected_roi_schema,
                "rois": rois_to_cache,
                "bytes": int(roi_bytes),
                "client_debug": _to_builtin(client_debug_payload),
            },
        )
    except Exception:
        # Caching is best-effort; never fail the request because of it.
        pass

    return _to_builtin(
        {
            "ok": True,
            "elapsed_ms": round(float(elapsed_ms), 2),
            "manual_stats": manual_stats,
            "debug": debug,
            "goal_success": goal_success_payload,
            "ocr_result": ocr_result,
            "ui_state": ui_state,
            "rl": rl_result,
            "record": {
                "id": base,
                "image_path": stored_image_rel_path,
                "json_path": record_json_rel_path,
                "created_at": created_at,
            },
        }
    )
