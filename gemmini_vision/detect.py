import os

# Avoid OpenMP duplicate runtime crashes when mixing packages that bundle MKL/OpenMP.
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import cv2
import easyocr
import re
import numpy as np
from difflib import SequenceMatcher

# =========================
# OCR 엔진
# =========================
def _env_bool(name: str) -> bool | None:
    raw = os.getenv(name)
    if raw is None:
        return None
    v = raw.strip().lower()
    if v in ("1", "true", "yes", "y", "on"):
        return True
    if v in ("0", "false", "no", "n", "off"):
        return False
    return None


def _cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


_CUDA_AVAILABLE = _cuda_available()
_OCR_GPU_REQUESTED = _env_bool("GEMMINI_OCR_GPU")
OCR_GPU_ENABLED = bool(_CUDA_AVAILABLE if _OCR_GPU_REQUESTED is None else (_OCR_GPU_REQUESTED and _CUDA_AVAILABLE))

reader_ko = easyocr.Reader(["ko"], gpu=OCR_GPU_ENABLED, verbose=False)
reader_num = easyocr.Reader(["en"], gpu=OCR_GPU_ENABLED, verbose=False)

# =========================
# annotation (center 기준)
# =========================
ANNOTATION = {
    "width": 2560,
    "height": 1440,
    "boxes": [
        {"label": "option1", "x": 1043.72, "y": 793.61, "w": 150.71, "h": 112.23},
        {"label": "option2", "x": 1197.63, "y": 795.22, "w": 147.50, "h": 105.81},
        {"label": "option3", "x": 1360.36, "y": 796.02, "w": 158.72, "h": 113.83},
        {"label": "option4", "x": 1512.67, "y": 795.22, "w": 152.31, "h": 109.02},
        {"label": "possible", "x": 1669.79, "y": 794.41, "w": 126.66, "h": 59.32},
        # NOTE: cost/count boxes are slightly enlarged to avoid clipping digits
        # under different UI scales (e.g. "1,800" losing the leading '1').
        {"label": "cost",     "x": 1579.21, "y": 887.40, "w": 150.00,  "h": 70.00},
        {"label": "count",    "x": 1476.60, "y": 1022.08,"w": 120.00,  "h": 80.00},
    ]
}

# =========================
# 전처리
# =========================
def preprocess(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    _, th = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
    return th


def preprocess_numeric_variants(img):
    """
    숫자 OCR 안정화를 위한 다중 전처리.
    회색 배경(contrast 낮음)에서도 인식되도록 binary / inverse / adaptive를 함께 시도한다.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)

    variants = []

    _, th150 = cv2.threshold(blur, 150, 255, cv2.THRESH_BINARY)
    variants.append(th150)

    _, th150_inv = cv2.threshold(blur, 150, 255, cv2.THRESH_BINARY_INV)
    variants.append(th150_inv)

    adaptive = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 5
    )
    variants.append(adaptive)
    variants.append(cv2.bitwise_not(adaptive))

    _, otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    variants.append(otsu)
    variants.append(cv2.bitwise_not(otsu))

    return variants


def read_numeric_text_with_fallback(roi, allowlist, expected_len=None):
    """
    숫자 필드 전용 OCR:
    - 여러 전처리를 시도하고
    - confidence/길이 기준으로 가장 그럴듯한 문자열을 고른다.
    """
    best_text = ""
    best_score = -1.0
    clean_allowlist = ''.join(dict.fromkeys(allowlist))

    for idx, proc in enumerate(preprocess_numeric_variants(roi)):
        try:
            results = reader_num.readtext(proc, detail=1, allowlist=clean_allowlist)
        except Exception:
            results = []

        for item in results:
            if not isinstance(item, (list, tuple)) or len(item) < 2:
                continue

            txt = str(item[1])
            conf = float(item[2]) if len(item) >= 3 and isinstance(item[2], (int, float)) else 0.0

            if '/' in clean_allowlist:
                filtered = ''.join(ch for ch in txt if ch.isdigit() or ch == '/')
            else:
                filtered = ''.join(ch for ch in txt if ch.isdigit())

            if not filtered:
                continue

            # 앞쪽 variant를 조금 우대, confidence/길이도 반영
            score = conf + 0.03 * len(filtered) - 0.01 * idx
            if '/' in clean_allowlist:
                # For count fields we strongly prefer candidates that include '/'.
                # EasyOCR sometimes returns a high-confidence "19" from "(7/9)",
                # which would collapse into "1/9" after normalization.
                score += 0.60 if '/' in filtered else -0.60
            if expected_len is not None:
                score -= 0.08 * abs(len(filtered) - expected_len)
            if score > best_score:
                best_score = score
                best_text = filtered

    if best_text:
        return best_text

    # 최종 fallback: 기존 단일 전처리
    proc = preprocess(roi)
    raw = reader_num.readtext(proc, detail=0, allowlist=clean_allowlist)
    return raw[0] if raw else ""


def _safe_crop_frac(roi, *, x1f: float, x2f: float, y1f: float, y2f: float):
    h, w = roi.shape[:2]
    x1 = int(round(max(0.0, min(1.0, x1f)) * w))
    x2 = int(round(max(0.0, min(1.0, x2f)) * w))
    y1 = int(round(max(0.0, min(1.0, y1f)) * h))
    y2 = int(round(max(0.0, min(1.0, y2f)) * h))
    x1 = max(0, min(w - 1, x1))
    x2 = max(0, min(w, x2))
    y1 = max(0, min(h - 1, y1))
    y2 = max(0, min(h, y2))
    if x2 - x1 < 8 or y2 - y1 < 8:
        return None
    return roi[y1:y2, x1:x2]


def _tight_numeric_bbox_crop(roi):
    """
    Try to isolate the bright numeric glyphs inside a small ROI (e.g. "(7/9)").
    This is intentionally cheap and tolerant because the ROI is tiny.
    """
    try:
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (3, 3), 0)
        _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # If background becomes white, invert so glyphs are white.
        if float(th.mean()) > 127.0:
            th = cv2.bitwise_not(th)

        kernel = np.ones((2, 2), np.uint8)
        th = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel, iterations=1)

        ys, xs = np.where(th > 0)
        if xs.size < 16 or ys.size < 16:
            return None

        h, w = roi.shape[:2]
        x1 = int(xs.min())
        x2 = int(xs.max())
        y1 = int(ys.min())
        y2 = int(ys.max())

        pad = max(2, int(round(min(h, w) * 0.08)))
        x1 = max(0, x1 - pad)
        y1 = max(0, y1 - pad)
        x2 = min(w, x2 + pad)
        y2 = min(h, y2 + pad)

        if x2 - x1 < 8 or y2 - y1 < 8:
            return None

        return roi[y1:y2, x1:x2]
    except Exception:
        return None


def _parse_count_norm(norm: str):
    try:
        left_s, right_s = str(norm).split("/", 1)
        return int(left_s), int(right_s)
    except Exception:
        return None


def _count_raw_has_valid_den(raw: str) -> bool:
    # We only support count denominators 7 or 9 (rare/hero). Exclude other modes (e.g. /5).
    s = str(raw or "")
    digits = re.findall(r"\d", s)
    if not digits:
        return False
    if "7" not in digits and "9" not in digits:
        return False
    if "/" in s:
        den_part = s.split("/")[-1]
        den_digits = re.findall(r"\d", den_part)
        return bool(den_digits) and any(d in ("7", "9") for d in den_digits)
    return True


def read_possible_text_with_multicrop(roi):
    """
    Robust OCR for the "n회 가능" field.

    The left refresh icon is frequently mistaken as a digit (mostly 3/5),
    so we sample a few numeric-focused sub-crops and select by vote.
    """
    if roi is None or not isinstance(roi, np.ndarray) or roi.size == 0:
        return "", 0

    candidates = []
    specs = [
        # Legacy crop kept for compatibility with older UI captures.
        ("legacy", 0.15, 0.48, 0.15, 0.90, 1.0),
        # Lift top boundary to keep the full glyph body (fixes 1 -> 3 cases).
        ("legacy_top", 0.15, 0.48, 0.10, 0.90, 1.0),
        # Shift right to reduce icon contamination (fixes 0 -> 5 cases).
        ("shifted", 0.20, 0.50, 0.10, 0.90, 2.0),
        # Slightly tighter center crop for noisy captures.
        ("center", 0.24, 0.52, 0.15, 0.90, 2.0),
    ]

    for name, x1f, x2f, y1f, y2f, weight in specs:
        sub = _safe_crop_frac(roi, x1f=x1f, x2f=x2f, y1f=y1f, y2f=y2f)
        if sub is not None:
            candidates.append((name, sub, float(weight)))

    if not candidates:
        return "", 0

    votes = []
    scores = {}
    raws_by_value = {}
    for _name, sub, weight in candidates:
        raw = read_numeric_text_with_fallback(sub, allowlist="012345", expected_len=1)
        if not raw:
            continue
        value = normalize_possible(raw)
        votes.append(int(value))
        scores[int(value)] = float(scores.get(int(value), 0.0)) + float(weight)
        raws_by_value.setdefault(int(value), []).append((float(weight), str(raw)))

    if not votes:
        return "", 0

    if not scores:
        return "", 0

    best_score = max(scores.values())
    tied = [v for v, s in scores.items() if s == best_score]

    if len(tied) == 1:
        picked = tied[0]
    else:
        # Tie-break toward the median cluster, then the smaller value.
        sv = sorted(votes)
        n = len(sv)
        med = float(sv[n // 2]) if (n % 2 == 1) else ((sv[n // 2 - 1] + sv[n // 2]) / 2.0)
        tied.sort(key=lambda v: (abs(float(v) - med), v))
        picked = tied[0]

    picked_raw = ""
    for _w, raw in sorted(raws_by_value.get(picked, []), key=lambda x: (-x[0], len(x[1]))):
        picked_raw = raw
        break
    return picked_raw, int(picked)


def read_count_text_with_multicrop(roi):
    """
    Robust count OCR for the \"(n/7)\" or \"(n/9)\" field.

    EasyOCR is often distracted by nearby Korean glyphs and parentheses, returning
    strings like \"19\" or \"1284/52\". Instead of hardcoding a single crop,
    try multiple cheap sub-crops and choose the most plausible ratio.
    """
    if roi is None or not isinstance(roi, np.ndarray) or roi.size == 0:
        return "", None

    candidates = [("full", roi)]

    # Numeric-heavy crops (keep slash context; avoid cutting too aggressively).
    c1 = _safe_crop_frac(roi, x1f=0.10, x2f=0.99, y1f=0.00, y2f=1.00)
    if c1 is not None:
        candidates.append(("wide", c1))
    c2 = _safe_crop_frac(roi, x1f=0.20, x2f=0.99, y1f=0.05, y2f=0.95)
    if c2 is not None:
        candidates.append(("right", c2))
    c3 = _safe_crop_frac(roi, x1f=0.05, x2f=0.95, y1f=0.10, y2f=0.90)
    if c3 is not None:
        candidates.append(("center", c3))

    tight = _tight_numeric_bbox_crop(roi)
    if tight is not None:
        candidates.append(("tight", tight))

    best_raw = ""
    best_norm = None
    best_score = -1e9

    for _name, sub in candidates:
        raw = read_numeric_text_with_fallback(sub, allowlist="0123456789/", expected_len=3)
        norm = normalize_count(raw)
        if norm is None:
            continue
        parsed = _parse_count_norm(norm)
        if not parsed:
            continue
        left, right = parsed
        if right not in (7, 9):
            continue
        if left < 0 or left > right:
            continue
        if not _count_raw_has_valid_den(raw):
            continue

        raw_s = str(raw or "")
        score = 0.0
        score += 1.0 if "/" in raw_s else -0.5
        score -= 0.25 * max(0, raw_s.count("/") - 1)
        # Penalize very long digit streams (often noise like \"1284/52\").
        score -= 0.06 * max(0, len(raw_s) - 5)
        # Prefer clean denominators that explicitly include 7/9.
        den_part = raw_s.split("/")[-1] if "/" in raw_s else raw_s
        score += 0.8 if ("7" in den_part or "9" in den_part) else 0.0

        # Prefer keeping context (full ROI) when equally plausible.
        if _name == "full":
            score += 0.10

        if score > best_score:
            best_score = score
            best_raw = raw
            best_norm = norm

    return best_raw, best_norm

# =========================
# center ROI crop
# =========================
def crop_center(img, box, debug=False):
    img_h, img_w = img.shape[:2]

    # 실제 이미지 크기에 맞춰 좌표 스케일링
    scale_x = img_w / ANNOTATION["width"]
    scale_y = img_h / ANNOTATION["height"]
    
    cx = box["x"] * scale_x
    cy = box["y"] * scale_y
    w  = box["w"] * scale_x
    h  = box["h"] * scale_y

    x1 = int(cx - w / 2)
    y1 = int(cy - h / 2)
    x2 = int(cx + w / 2)
    y2 = int(cy + h / 2)
    
    # 경계 검사
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(img_w, x2)
    y2 = min(img_h, y2)
    
    if debug:
        print(f"  스케일: {scale_x:.3f}x{scale_y:.3f}, 좌표: ({x1},{y1})-({x2},{y2})")

    return img[y1:y2, x1:x2]

# =========================
# 옵션 화이트리스트
# =========================
OPTION_WHITELIST = {
    'effect1': ['공격력', '추가피해', '아군피해강화', '아군 피해 강화', '낙인력', '아군공격강화', '아군공격력강화', '보스피해'],
    'effect2': ['공격력', '추가피해', '아군피해강화', '아군 피해 강화', '낙인력', '아군공격강화', '아군공격력강화', '보스피해'],
    'willpower': ['의지력 효율', '의지력효율'],
    'points': ['질서 포인트', '혼돈 포인트', '질서포인트', '혼돈포인트'],
    'special': ['상태유지', '상태 유지', '가공 상태 유지', '새로고침', '비용', '가공 비용']
}

# OCR 텍스트 정규화
def normalize_ocr_text(text):
    """
    OCR 텍스트에서 노이즈 제거 및 정규화
    """
    # 공백 제거
    text = text.replace(' ', '')
    # 일반적인 OCR 오류 패턴 제거
    noise_patterns = ['-거', '-히', '나거', '-', '.']
    for pattern in noise_patterns:
        text = text.replace(pattern, '')
    return text

# =========================
# 색상 기반 카테고리 분류
# =========================
def detect_option_category_by_color(roi):
    """
    다이아몬드 영역의 대표 색상으로 카테고리 구분
    Returns: 'effect1'(초록/좌), 'effect2'(파랑/우), 'willpower'(빨강/상), 'points'(주황/하), 'special'(회색)
    """
    # 중심부 샘플링 (다이아몬드 중심)
    h, w = roi.shape[:2]
    center_region = roi[h//3:2*h//3, w//3:2*w//3]
    
    # BGR을 HSV로 변환
    hsv = cv2.cvtColor(center_region, cv2.COLOR_BGR2HSV)
    mean_hsv = cv2.mean(hsv)[:3]
    
    h_val = mean_hsv[0]
    s_val = mean_hsv[1]
    v_val = mean_hsv[2]
    
    # 채도가 낮으면 회색 (특수 옵션)
    if s_val < 50:
        return 'special'  # 상태유지, 새로고침, 비용변경
    
    # Hue 값으로 색상 구분 (범위 확장)
    if 30 <= h_val <= 90:  # 초록 (범위 확장)
        return 'effect1'  # 부옵션1 (좌측)
    elif 95 <= h_val <= 135:  # 파랑 (범위 확장)
        return 'effect2'  # 부옵션2 (우측)
    elif h_val < 10 or h_val > 170:  # 빨강
        return 'willpower'  # 의지력 (상단)
    elif 10 <= h_val <= 30:  # 주황/노랑
        return 'points'  # 질서혼돈 (하단)
    else:
        # unknown이지만 채도가 있으면 부옵션으로 추정
        print(f"      경고: 색상 인식 실패 (H:{h_val:.1f}, S:{s_val:.1f}, V:{v_val:.1f})")
        if s_val > 80:  # 채도가 높으면
            # 90-95 사이는 effect2로 추정 (파란색 경계)
            if 90 <= h_val <= 95:
                return 'effect2'
            return 'effect1'  # 기본값
        return 'unknown'

# =========================
# 옵션 텍스트 매칭
# =========================
def fuzzy_match_option(ocr_text, category):
    """
    OCR 텍스트를 화이트리스트와 매칭
    """
    whitelist = OPTION_WHITELIST.get(category, [])
    best_match = None
    best_score = 0
    
    # OCR 텍스트 정규화 (공백, 노이즈 제거)
    normalized_ocr = normalize_ocr_text(ocr_text)
    
    # 특수 옵션: "가공" prefix 제거
    if category == 'special' and '가공' in normalized_ocr:
        normalized_ocr = normalized_ocr.replace('가공', '')
        print(f"    [special] '가공' prefix 제거: '{ocr_text}' -> '{normalized_ocr}'")
    
    # 키워드 기반 우선 매칭 (명확한 구분을 위해)
    # "아군공격강화"가 단순 "공격력"으로 잘못 매칭되는 것을 먼저 차단한다.
    if ('아군' in normalized_ocr) and ('공격' in normalized_ocr) and ('강화' in normalized_ocr):
        for cand in ('아군공격력강화', '아군공격강화'):
            if cand in whitelist:
                return cand

    # 일반 "공격"은 마지막 fallback으로 사용 (너무 이르게 적용하면 오매칭이 많음).
    
    if '피해' in normalized_ocr:
        # "보스피해"와 "아군피해강화" 구분
        if '보스' in normalized_ocr:
            if '보스피해' in whitelist:
                return '보스피해'
        elif '아군' in normalized_ocr or '강화' in normalized_ocr:
            if '아군피해강화' in whitelist:
                return '아군피해강화'
        # 그냥 "추가피해"
        elif '추가' in normalized_ocr:
            if '추가피해' in whitelist:
                return '추가피해'
    
    if '낙인' in normalized_ocr:
        if '낙인력' in whitelist:
            return '낙인력'

    if '공격' in normalized_ocr:
        for option in whitelist:
            if '공격' in option:
                return option
    
    # 특수 옵션 매칭
    if category == 'special':
        if '상태' in normalized_ocr or '유지' in normalized_ocr:
            return '상태유지'
        if '새로고침' in normalized_ocr or '세로고침' in normalized_ocr:
            return '새로고침'
        if '비용' in normalized_ocr:
            return '비용'
    
    # 옵션명이 OCR 텍스트에 포함되어 있는지 (예: "보스"가 "보스피해"에 포함)
    for option in whitelist:
        if len(option) >= 2 and option[:2] in normalized_ocr:
            # 유사도로 추가 검증
            score = SequenceMatcher(None, normalized_ocr, option).ratio()
            if score > 0.5:
                return option
    
    # 유사도 매칭 (OCR 오류 보정)
    for option in whitelist:
        score = SequenceMatcher(None, normalized_ocr, option).ratio()
        if score > best_score:
            best_score = score
            best_match = option
    
    return best_match if best_score > 0.45 else None

# =========================
# 숫자 추출
# =========================
def detect_minus_sign_by_edge(roi_img, option_name=""):
    """
    이미지 엣지 분석으로 음수 기호(-) 감지
    음수: 좌측에 - 기호가 있으므로 좌측 엣지가 우측 엣지의 1.5배 이상
    
    반환값: True (음수), False (양수), None (판단 불가)
    """
    if roi_img is None or roi_img.size == 0:
        return None
    
    try:
        gray = cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        h, w = edges.shape
        
        # 좌측/우측 엣지 비교 (좌측 좁음: w//5)
        left_edge = np.sum(edges[:, :w//5])
        right_edge = np.sum(edges[:, 4*w//5:])
        
        # 비율: 좌측 엣지가 우측의 1.5배 이상이면 음수 가능성
        if right_edge > 0:
            ratio = left_edge / right_edge
            print(f"      [엣지분석] {option_name}: 좌측={left_edge:.0f}, 우측={right_edge:.0f}, 비율={ratio:.2f}")
            if ratio > 1.5:  # 기준을 1.8에서 1.5로 완화
                print(f"      → 음수 감지됨!")
                return True
            else:
                print(f"      → 양수 (비율 < 1.5)")
                return False
        
        return False
    except Exception as e:
        print(f"      엣지 분석 오류: {e}")
        return None


def _score_sign_tokens(text):
    """
    증가/감소 키워드 기반 점수.
    OCR 오인식을 고려해 완전일치 + 부분문자 힌트를 함께 사용한다.
    """
    t = (text or "").replace(" ", "")
    if not t:
        return 0, 0

    plus = 0
    minus = 0

    if '+' in t:
        plus += 3
    if '-' in t:
        minus += 3

    if '증가' in t:
        plus += 4
    if '감소' in t:
        minus += 4

    # 부분 힌트(오인식 대응)
    if '증' in t:
        plus += 1
    if '감' in t:
        minus += 1
    if ('증' in t) and ('가' in t):
        plus += 1
    if ('감' in t) and ('소' in t):
        minus += 1

    return plus, minus


def infer_sign_from_roi_keyword(roi_img):
    """
    옵션 ROI에서 '증가/감소' 키워드를 재탐색해 부호를 추정한다.
    - 전체/하단/우측/우하단 영역을 각각 OCR
    - 여러 이진화 변형으로 재시도
    """
    if roi_img is None or roi_img.size == 0:
        return None

    h, w = roi_img.shape[:2]
    regions = [
        roi_img,
        roi_img[int(h * 0.45):, :],                 # 하단
        roi_img[:, int(w * 0.42):],                 # 우측
        roi_img[int(h * 0.45):, int(w * 0.32):],    # 우하단
    ]

    plus_score = 0
    minus_score = 0

    for region in regions:
        if region.size == 0 or region.shape[0] < 8 or region.shape[1] < 8:
            continue

        for proc in preprocess_numeric_variants(region):
            try:
                texts = reader_ko.readtext(proc, detail=0)
            except Exception:
                texts = []
            merged = ''.join(texts) if texts else ''
            p, m = _score_sign_tokens(merged)
            if p > plus_score:
                plus_score = p
            if m > minus_score:
                minus_score = m

    # 충분히 확실한 경우만 반환
    if minus_score >= 3 and minus_score > plus_score:
        return -1
    if plus_score >= 3 and plus_score > minus_score:
        return 1
    return None


def infer_sign_from_value_color(roi_img, category=None):
    """
    텍스트 OCR이 실패했을 때, 값 표시 영역(우하단)의 글자색으로 부호를 보조 추정.
    - 빨강 우세: 음수(-)
    - 초록 우세: 양수(+)
    """
    if roi_img is None or roi_img.size == 0:
        return None

    h, w = roi_img.shape[:2]
    value_region = roi_img[int(h * 0.42):, int(w * 0.33):]
    if value_region.size == 0:
        return None

    hsv = cv2.cvtColor(value_region, cv2.COLOR_BGR2HSV)
    hch = hsv[:, :, 0]
    sch = hsv[:, :, 1]
    vch = hsv[:, :, 2]

    sat_mask = (sch >= 60) & (vch >= 60)
    red_mask = (((hch <= 10) | (hch >= 170)) & sat_mask)
    green_mask = (((hch >= 35) & (hch <= 95)) & sat_mask)

    red = int(np.count_nonzero(red_mask))
    green = int(np.count_nonzero(green_mask))

    # 충분한 픽셀이 있고 우세가 분명할 때만 사용
    if red >= max(60, int(green * 1.35)):
        return -1
    if green >= max(60, int(red * 1.35)):
        return 1
    return None


def infer_sign_from_text(ocr_text):
    """
    텍스트에서 부호 힌트 추출.
    음수 오검출을 줄이기 위해 명확한 힌트가 있을 때만 -를 반환한다.
    """
    text = (ocr_text or "").replace(" ", "")

    has_minus = ('-' in text) or ('감소' in text)
    has_plus = ('+' in text) or ('증가' in text)

    if has_minus and not has_plus:
        return -1
    if has_plus and not has_minus:
        return 1
    return None


def extract_effect_level(ocr_text):
    """
    effect 계열 숫자 추출.
    'Lv.3 증가' 형태를 우선 인식하고, 실패하면 단일 숫자 후보에서 선택한다.
    """
    text = ocr_text or ""
    lv_norm = (
        text.replace('|', 'L')
        .replace('Ⅰ', 'I')
        .replace('Ｌ', 'L')
        .replace('ｌ', 'l')
    )

    # Lv / Lw / Iv 오인식을 모두 약하게 허용
    m = re.search(r'[LlI][VvWw]?\s*[\.:]?\s*(\d)', lv_norm)
    if m:
        v = int(m.group(1))
        if v >= 5:
            # Lv 숫자 오인식(예: 3 -> 7) 완화
            return 3
        return max(1, min(4, v))

    digits = [int(d) for d in re.findall(r'\d', text)]
    if not digits:
        return 1

    # 허용 숫자 우선 (1~4)
    valid = [d for d in digits if 1 <= d <= 4]
    if valid:
        return valid[-1]

    # Lv 표기가 있는데 숫자가 깨진 경우(예: 7) 완화 보정
    if re.search(r'[LlI][VvWw]?', lv_norm) and digits[-1] >= 5:
        return 3

    return max(1, min(4, digits[-1]))


def clamp_option_value(category, option_name, value):
    """
    게임 규칙 기반 값 범위 강제.
    """
    option_name = option_name or ""

    if category in ('effect1', 'effect2'):
        if '변경' in option_name:
            return 0
        if value < 0:
            return -1
        return max(1, min(4, value))

    if category in ('willpower', 'points'):
        if value < 0:
            return -1
        return max(1, min(4, value))

    if category == 'special':
        if '상태' in option_name:
            return 0
        if '새로고침' in option_name or '세로고침' in option_name:
            if value <= 1:
                return 1
            return 2
        if '비용' in option_name:
            return 100 if value >= 0 else -100

    return value


def extract_number_value(ocr_text, option_name, roi_img=None, category=None):
    """
    OCR 텍스트에서 옵션 값을 추출.
    - effect 계열은 'Lv.x' 우선 파싱
    - 부호는 텍스트 힌트(증가/감소, +/-)를 우선
    - 모호할 때만 엣지 분석으로 음수 여부 보조 판단
    """
    text = ocr_text or ""
    option_name = option_name or ""
    sign_hint = infer_sign_from_text(text)
    if sign_hint is None and roi_img is not None:
        sign_hint = infer_sign_from_roi_keyword(roi_img)
    if sign_hint is None and roi_img is not None:
        sign_hint = infer_sign_from_value_color(roi_img, category)

    # 특수 옵션
    if '상태' in option_name:
        return 0
    if '비용' in option_name:
        if sign_hint == -1:
            return -100
        return 100
    if '세로고침' in option_name or '새로고침' in option_name:
        digits = [int(d) for d in re.findall(r'\d', text)]
        if digits:
            return 1 if digits[-1] <= 1 else 2
        return 1

    # effect 옵션 변경(숫자 없음)
    if category in ('effect1', 'effect2') and '변경' in text.replace(" ", ""):
        return 0

    # 카테고리별 숫자 추출
    if category in ('effect1', 'effect2'):
        mag = extract_effect_level(text)
    else:
        digits = [int(d) for d in re.findall(r'\d', text)]
        if digits:
            valid = [d for d in digits if 1 <= d <= 4]
            if valid:
                mag = valid[-1]
            else:
                # OCR often misreads '+1' as '7' (especially for 질서/혼돈/의지력).
                # When we only see out-of-range digits, prefer the conservative '+1'
                # instead of clamping to 4 (which creates obvious UI errors like '+4').
                if category in ("willpower", "points") and any(d in (5, 7) for d in digits):
                    mag = 1
                else:
                    mag = max(1, min(4, digits[-1]))
        else:
            mag = 1

    # 부호 결정: 텍스트 우선, 모호하면 엣지 보조
    if sign_hint == -1:
        value = -1
    elif sign_hint == 1:
        value = mag
    else:
        is_minus = detect_minus_sign_by_edge(roi_img, option_name) if roi_img is not None else None
        value = -1 if is_minus is True else mag

    return clamp_option_value(category, option_name, value)

# =========================
# 통합 옵션 감지
# =========================
def detect_option(roi_img):
    """
    하나의 다이아몬드 영역을 분석하여 옵션 정보 반환
    """
    try:
        # 1. 색상으로 카테고리 파악
        category = detect_option_category_by_color(roi_img)
        
        # 2. OCR로 전체 텍스트 읽기
        proc = preprocess(roi_img)
        ocr_result = reader_ko.readtext(proc, detail=0)
        ocr_text = ' '.join(ocr_result) if ocr_result else ""
        
        print(f"    OCR 텍스트: '{ocr_text}', 카테고리: {category}")
        
        # 3. 화이트리스트 매칭
        option_name = fuzzy_match_option(ocr_text, category)
        compact_ocr = ocr_text.replace(" ", "")

        # 텍스트 기반 보정: 색상 분류가 틀릴 때 willpower/points/special 복구
        if option_name is None:
            if '의지' in compact_ocr:
                category = 'willpower'
                option_name = '의지력 효율'
            elif ('질서' in compact_ocr) or ('혼돈' in compact_ocr) or ('포인트' in compact_ocr):
                category = 'points'
                option_name = '질서 포인트' if '질서' in compact_ocr else '혼돈 포인트'
            elif (
                ('항목' in compact_ocr and '보기' in compact_ocr)
                or ('다른항목보기' in compact_ocr)
                or ('다른' in compact_ocr and '보기' in compact_ocr)
            ):
                category = 'special'
                option_name = '새로고침'
            elif '상태' in compact_ocr:
                category = 'special'
                option_name = '상태유지'
            elif '비용' in compact_ocr:
                category = 'special'
                option_name = '비용'

        # effect 계열의 "효과 변경"은 숫자 옵션이 아니라 변환 옵션으로 처리
        raw_no_space = compact_ocr
        if category in ('effect1', 'effect2') and '변경' in raw_no_space:
            if option_name:
                option_name = f"{option_name} 변경"
            else:
                option_name = "효과 변경"
        
        # 4. 숫자 추출 (ROI 이미지도 함께 전달해서 이미지 기반 음수 감지)
        value = extract_number_value(ocr_text, option_name, roi_img, category)
        
        # 5. 부옵션 위치 판단 (ROI 자체 위치는 사용하지 않고 카테고리로만)
        # 나중에 UI에서 position 정보로 교차 검증 가능
        if category == 'effect1':
            position = 'left'
        elif category == 'effect2':
            position = 'right'
        else:
            position = 'other'
        
        result = {
            'category': category,
            'option': option_name,
            'value': value,
            'raw_ocr': ocr_text,
            'position': position,
            'formatted': f"{option_name} {value:+d}" if option_name and value != 0 else option_name or ocr_text
        }
        
        print(f"    매칭 결과: {option_name} {value:+d}" if option_name else f"    매칭 실패: {ocr_text}")
        
        return result
    
    except Exception as e:
        print(f"    detect_option 오류: {e}")
        import traceback
        traceback.print_exc()
        return {
            'category': 'unknown',
            'option': None,
            'value': 0,
            'raw_ocr': '',
            'position': 'other',
            'formatted': '인식 실패'
        }

# =========================
# 보정 로직
# =========================
def normalize_possible(text):
    digits = re.findall(r'\d', str(text))
    if not digits:
        return 0
    vals = [int(d) for d in digits]
    # OCR이 여러 숫자를 붙여 읽는 경우가 많아 최빈값 우선 사용
    counts = {}
    for v in vals:
        if 0 <= v <= 5:
            counts[v] = counts.get(v, 0) + 1

    if counts:
        best_count = max(counts.values())
        # 동률이면 앞에서 먼저 나온 숫자를 채택
        for v in vals:
            if counts.get(v, 0) == best_count:
                return v

    v = vals[-1]
    return max(0, min(5, v))

VALID_COSTS = [0, 900, 1800]

def normalize_cost(text):
    digits = ''.join(c for c in text if c.isdigit())
    if digits == "":
        return 0
    v = int(digits)
    return min(VALID_COSTS, key=lambda x: abs(x - v))

def normalize_count(text):
    if text is None:
        return None

    raw = str(text).strip()
    if raw == "":
        return None

    raw = (
        raw.replace('\\', '/')
        .replace('|', '/')
        .replace('I', '1')
        .replace('l', '1')
    )

    # EasyOCR sometimes returns multiple slashes like "11/7/9".
    # In that case the regex below would lock onto the first "11/7" and lose the true denominator.
    # Fall back to digit-stream reconstruction which reliably picks 7/9.
    if raw.count('/') >= 2:
        digits = [int(d) for d in re.findall(r'\d', raw)]
        if len(digits) >= 2:
            if 9 in digits:
                right = 9
            elif 7 in digits:
                right = 7
            else:
                right = digits[-1]
            cand = [d for d in digits if d != right]
            left = max(cand) if cand else digits[0]
            if right not in (7, 9):
                right = 7 if abs(right - 7) <= abs(right - 9) else 9
            if left > right:
                left = right
            return f"{left}/{right}"

    # 1) 정상/유사 패턴: x/y
    m = re.search(r'(\d{1,2})\s*/\s*(\d{1,2})', raw)
    if m:
        left_str = m.group(1)
        right_str = m.group(2)

        # Right side is almost always 7 or 9, but OCR often appends an extra
        # digit from the closing parenthesis, e.g. "9)" -> "91".
        if '9' in right_str:
            right = 9
        elif '7' in right_str:
            right = 7
        else:
            right = int(right_str)

        # Left side is a single digit, but OCR may prepend a stray digit from
        # the opening parenthesis. Prefer the largest digit found to avoid
        # collapsing into 0 when we see "70/91" etc.
        left_digits = [int(d) for d in re.findall(r'\d', left_str)]
        if len(left_digits) == 1:
            left = int(left_digits[0])
        else:
            # Common noise pattern for "(5/9)" is "75/9" where a stray leading
            # 7 appears before the true numerator. Fix this specific case
            # without changing broader behavior.
            if (
                right == 9
                and len(left_digits) >= 2
                and left_digits[0] == 7
                and 0 <= left_digits[1] <= 6
            ):
                left = int(left_digits[1])
            else:
                left = max(left_digits) if left_digits else 0

        # 총 횟수는 7 또는 9로 수렴 (OCR 오인식 보정)
        if right not in (7, 9):
            right = 7 if abs(right - 7) <= abs(right - 9) else 9

        left = max(0, min(9, left))
        right = max(0, min(9, right))
        if left > right:
            left = right
        return f"{left}/{right}"

    # 2) 슬래시가 깨졌을 때: 숫자 2개를 x/y로 복원
    digits = [int(d) for d in re.findall(r'\d', raw)]
    if len(digits) >= 2:
        # Denominator is almost always present as 7 or 9 in the OCR stream.
        if 9 in digits:
            right = 9
        elif 7 in digits:
            right = 7
        else:
            right = digits[-1]

        # Pick a plausible numerator without collapsing to 0 from parentheses.
        cand = [d for d in digits if d != right]
        left = max(cand) if cand else digits[0]
        if right not in (7, 9):
            right = 7 if abs(right - 7) <= abs(right - 9) else 9
        if left > right:
            left = right
        return f"{left}/{right}"

    return None

# =========================
# 메인
# =========================
def main():
    img = cv2.imread("./arkgrid.png")
    if img is None:
        print("이미지를 불러올 수 없습니다.")
        return

    results = {}

    for box in ANNOTATION["boxes"]:
        roi = crop_center(img, box)
        label = box["label"]

        # ---------- 옵션 감지 (새로운 방식) ----------
        if label.startswith("option"):
            option_info = detect_option(roi)
            results.setdefault("options", []).append(option_info)

        elif label == "possible":
            _raw, norm = read_possible_text_with_multicrop(roi)
            results["possible"] = norm

        elif label == "cost":
            raw = read_numeric_text_with_fallback(roi, allowlist="0189")
            results["cost"] = normalize_cost(raw)

        elif label == "count":
            raw = read_numeric_text_with_fallback(roi, allowlist="0123456789/", expected_len=3)
            results["count"] = normalize_count(raw)

    # =========================
    # 출력
    # =========================
    print("\n===== OCR RESULT =====")
    for k, v in results.items():
        print(f"{k}: {v}")

# =========================
if __name__ == "__main__":
    main()
