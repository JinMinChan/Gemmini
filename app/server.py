from __future__ import annotations

import hashlib
import json
import os
import re
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

RATE_LIMIT_WINDOW_SECONDS = 60
RATE_LIMIT_MAX_REQUESTS = 20

ACTION_NAME = {
    0: "process",
    1: "reroll",
    2: "stop",
}

_requests_by_ip: Dict[str, Deque[float]] = defaultdict(deque)
_parser_instance = None
_rl_runtime = None
_goal_success_runtime = None
_runtime_lock = threading.Lock()

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


def _sanitize_ip_for_path(client_ip: str) -> str:
    raw = str(client_ip or "unknown").strip()
    if not raw:
        raw = "unknown"
    sanitized = re.sub(r"[^0-9A-Za-z._-]+", "_", raw).strip("._")
    return sanitized or "unknown"


def _record_dirs_for_ip(client_ip: str, *, create: bool = True) -> tuple[str, Path, Path, Path]:
    ip_dir = _sanitize_ip_for_path(client_ip)
    base_dir = RECORDS_DIR / ip_dir
    images_dir = base_dir / "images"
    json_dir = base_dir / "json"
    message_dir = base_dir / "message"
    if create:
        images_dir.mkdir(parents=True, exist_ok=True)
        json_dir.mkdir(parents=True, exist_ok=True)
        message_dir.mkdir(parents=True, exist_ok=True)
    return ip_dir, images_dir, json_dir, message_dir


def _check_rate_limit(ip: str) -> None:
    now = time.time()
    bucket = _requests_by_ip[ip]
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


def _build_record_name(detected: str, client_ip: str) -> tuple[str, str]:
    # Human-readable datetime format: 2026-02-11_14-30-45
    dt = datetime.now(timezone.utc)
    date_str = dt.strftime("%Y-%m-%d_%H-%M-%S")

    _, images_dir, json_dir, _ = _record_dirs_for_ip(client_ip, create=True)

    # Find next sequence number for this datetime within the same IP bucket.
    prefix = f"{date_str}_"
    seq = 1
    for existing_dir in (images_dir, json_dir):
        if existing_dir.exists():
            for existing_file in existing_dir.glob(f"{prefix}*"):
                try:
                    # Extract sequence number from filename
                    name_part = existing_file.stem  # filename without extension
                    seq_part = name_part.split("_")[-1]
                    if seq_part.isdigit():
                        seq = max(seq, int(seq_part) + 1)
                except (IndexError, ValueError):
                    pass

    ext = "jpg" if detected == "jpeg" else detected
    base = f"{date_str}_{seq:03d}"
    return base, ext


def _persist_image_bytes(base: str, ext: str, data: bytes, client_ip: str) -> tuple[str, str]:
    stored_name = f"{base}.{ext}"
    ip_dir, images_dir, _, _ = _record_dirs_for_ip(client_ip, create=True)
    # Store once under per-IP records tree (avoid duplicate growth).
    (images_dir / stored_name).write_bytes(data)
    return stored_name, f"records/{ip_dir}/images/{stored_name}"


def _persist_analyze_record(
    *,
    base: str,
    created_at: str,
    image_rel_path: Optional[str],
    source_filename: str,
    client_ip: str,
    elapsed_ms: float,
    manual_stats: dict,
    goal_success: Optional[dict],
    ocr_result: dict,
    ui_state: dict,
    rl_result: dict,
) -> tuple[str, str]:
    ip_dir, _, json_dir, _ = _record_dirs_for_ip(client_ip, create=True)
    json_name = f"{base}.json"
    payload = _to_builtin(
        {
            "record_id": base,
            "created_at": created_at,
            "source_filename": source_filename,
            "client_ip": client_ip,
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
    return json_name, f"records/{ip_dir}/json/{json_name}"


def _persist_bug_report(
    *,
    client_ip: str,
    message: str,
    record_id: Optional[str],
    user_agent: Optional[str],
) -> str:
    ip_dir, _, _, message_dir = _record_dirs_for_ip(client_ip, create=True)
    created_at = datetime.now(timezone.utc).isoformat()
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%S_%f")
    short = hashlib.sha256(f"{client_ip}|{created_at}|{message}".encode("utf-8", errors="ignore")).hexdigest()[:8]
    json_name = f"{stamp}_{short}.json"
    payload = {
        "created_at": created_at,
        "record_id": str(record_id or "").strip() or None,
        "client_ip": str(client_ip),
        "user_agent": str(user_agent or ""),
        "message": str(message),
    }
    (message_dir / json_name).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return f"records/{ip_dir}/message/{json_name}"


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
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/health")
async def health() -> dict:
    return {"ok": True}


@app.post("/api/report")
async def submit_report(
    request: Request,
    message: str = Form(...),
    record_id: Optional[str] = Form(None),
) -> dict:
    _verify_proxy_secret(request)
    ip = _client_ip(request)
    _check_rate_limit(ip)

    text = str(message or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="message is required")
    if len(text) > 4000:
        raise HTTPException(status_code=400, detail="message too long (max 4000 chars)")

    rel_path = _persist_bug_report(
        client_ip=ip,
        message=text,
        record_id=record_id,
        user_agent=request.headers.get("user-agent"),
    )
    return {"ok": True, "message_path": rel_path}


@app.post("/api/upload")
async def upload_image(request: Request, image: UploadFile = File(...)) -> dict:
    _verify_proxy_secret(request)
    ip = _client_ip(request)
    _check_rate_limit(ip)

    filename, data, detected = await _read_valid_image(image)
    base, ext = _build_record_name(detected, ip)
    stored_name, stored_rel_path = _persist_image_bytes(base, ext, data, ip)
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
    client_debug: Optional[str] = Form(None),
) -> dict:
    _verify_proxy_secret(request)
    ip = _client_ip(request)
    _check_rate_limit(ip)

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
    t_roi0 = time.perf_counter()
    for label, upload in roi_uploads.items():
        if upload is None:
            continue
        roi_filename, roi_data, roi_detected = await _read_valid_image(upload)
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

    base, ext = _build_record_name(detected, ip)

    if data:
        t_persist0 = time.perf_counter()
        _stored_image_name, stored_image_rel_path = _persist_image_bytes(base, ext, data, ip)
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
    t_ocr0 = time.perf_counter()
    if has_complete_roi_set:
        ocr_mode = "multi_crop"
        ocr_result = parser.parse_ocr_rois(roi_images)
    else:
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

    created_at = datetime.now(timezone.utc).isoformat()

    _record_json_name, record_json_rel_path = _persist_analyze_record(
        base=base,
        created_at=created_at,
        image_rel_path=stored_image_rel_path,
        source_filename=filename,
        client_ip=ip,
        elapsed_ms=elapsed_ms,
        manual_stats=manual_stats,
        goal_success=goal_success_payload,
        ocr_result=ocr_result,
        ui_state=ui_state,
        rl_result=rl_result,
    )

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
