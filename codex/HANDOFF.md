# Gemmini Handoff (2026-02-15)

새 세션(새 Codex/LLM)에서 **빠르게 프로젝트 맥락을 복구**할 수 있도록, 현재 저장소 구조/런타임 아키텍처/핵심 규칙/운영 팁을 요약한 문서입니다.

## 1) 한 줄 요약

브라우저에서 로스트아크 젬 가공 화면을 **화면공유** → 프론트가 **ROI 7개를 멀티-크롭(multi-crop)** 해서 `/api/analyze`로 전송 → AI 서버가 **OCR → 상태 파싱 → RL 추천 + MC 성공확률** 계산 → 결과를 UI에 표시.

## 2) 아키텍처(배포 형태)

### 기본 흐름

1. **Browser (frontend/index.html)**  
   - 화면 공유
   - ROI(옵션 4개 + 리롤/가공횟수/골드)를 crop하여 요청
   - 필요시 fallback으로 fullframe JPEG 업로드

2. **Vercel Edge Proxy (frontend/api/*.js)**  
   - `/api/analyze`, `/api/report`, `/api/notice`, `/api/health`
   - 환경변수 `AI_BACKEND_URL`로 AI 서버에 프록시
   - 환경변수 `API_SHARED_SECRET`가 있으면 `x-gemmini-key`로 인증 헤더 추가
   - 가능한 경우 실제 클라이언트 IP를 `x-gemmini-client-ip`로 전달

3. **AI Server (FastAPI: app/server.py)**  
   - OCR 파싱(`gemmini_vision/*`)
   - RL 추천(`models/production/...` + `gem_core/*`)
   - 목표 성공확률 MC 추정
   - 기록 저장(`app/records/YYYY-MM-DD/...`)

### 환경변수(핵심)

Vercel:
- `AI_BACKEND_URL`: AI 서버 베이스 URL (예: `https://xxxx.ngrok-free.app`)
- `API_SHARED_SECRET`: 프록시 인증용 시크릿

AI 서버:
- `GEMMINI_PROXY_SHARED_SECRET`: Vercel `API_SHARED_SECRET`와 동일 (설정 시 analyze/report/upload는 인증 필요)
- `GEMMINI_RATE_LIMIT_WINDOW_SECONDS` / `GEMMINI_RATE_LIMIT_MAX_REQUESTS`: 429 rate limit
- `GEMMINI_GOAL_MC_ROLLOUTS`: MC rollouts (기본 256)
- `GEMMINI_OCR_GPU`: OCR에 GPU 사용 여부(가능하면 사용)
- `GEMMINI_HOMOGRAPHY_*`: fullframe fallback에서 content-rect/homography 추출 튜닝

## 3) 저장소 구조(폴더별 역할)

### 루트
- `run_server.py`: `uvicorn.run("app.server:app", host="0.0.0.0", port=8000)`
- `requirements.txt`: AI 서버 런타임 의존성
- `start_ngrok.sh`: ngrok로 AI 서버 외부 노출
- `start_cloudflare.sh`: cloudflared Quick Tunnel(망에서 7844 막히면 실패 가능)

### `frontend/` (Vercel Root Directory)
- `frontend/index.html`: 실제 서비스 UI (화면공유 + 상태입력 + ROI 조정 + 제보)
- `frontend/api/analyze.js`: Edge 프록시 → AI 서버 `/api/analyze`
- `frontend/api/report.js`: Edge 프록시 → AI 서버 `/api/report`
- `frontend/api/notice.js`: Edge 프록시 → AI 서버 `/api/notice`
- `frontend/api/health.js`: Edge 헬스체크(backend까지 ping)
- `frontend/vercel.json`: 라우팅/캐시(no-store) 설정
- `frontend/static/roi_example.jpg`: ROI 조정 모달에서 보여주는 예시 이미지

### `app/` (AI 서버)
- `app/server.py`: FastAPI 서버 본체
- `app/static/index.html`: 로컬 테스트용 프론트(백엔드에서 `/`로 서빙)
- `app/static/roi_example.jpg`: 로컬 프론트 예시 이미지
- `app/notice.json`: 공지사항(파일 기반, **Vercel redeploy 없이 즉시 반영**)
- `app/records/`: 운영 중 생성되는 기록(날짜 버킷)
- `app/uploads/`: 업로드 임시(자리표시자)

### `gemmini_vision/` (OCR)
- `gemmini_vision/detect.py`: EasyOCR + 전처리 + 숫자/옵션 정규화
- `gemmini_vision/parser.py`: ROI별 OCR 수행 → 서비스 `ui_state`로 변환

### `gem_core/` (RL 환경/추론 보조)
- `gem_core/role_env.py`: 환경/관측/행동 마스크 등

### `models/`
- `models/production/gemmini_v9/best_model.zip`: 운영 모델(서버가 사용)

### `codex/` (운영/디버그 도구)
- `codex/overlay_rois_on_records.py`: records 이미지 전체에 ROI 오버레이(성공/실패 시각화)
- `codex/analyze_bug_reports_by_date.py`: message/제보 분석
- `codex/*`: 기타 원인분석/마이그레이션 스크립트

## 4) `/api/analyze` 데이터 흐름(중요)

1. 프론트는 기본적으로 **multi-crop**으로 ROI 7개를 업로드:
   - `option1..4`, `possible`, `cost`, `count` (PNG)
   - `roi_schema_version=screen_v2_multicrop`
2. ROI 업로드가 실패하면 fallback으로 **fullframe JPEG**를 업로드:
   - `image` 필드에 1장 업로드
   - 서버는 가능하면 **content-rect/homography로 ROI를 추출**해서 OCR
3. 서버는 OCR 결과를 `ui_state`로 변환한 뒤:
   - RL로 추천 행동(`process/reroll/stop`) 출력
   - MC rollouts로 목표 성공확률(`goal_success.success_prob`) 출력
4. 사용자가 아래 값을 수동 수정(+/-)하면 override가 서버로 전달됨:
   - `override_rerolls`, `override_attempts_left`, `override_cost_state`
   - 서버는 `ui_state.manual_override`에 기록

## 5) OCR/ROI 규칙(핵심)

### ROI 스키마(기본)
- `screen_v2_multicrop` 기준좌표: 2560x1440 (16:9)
- 프론트는 21:9 등 비율이 다르면 **중앙 16:9 content 영역으로 매핑** 후 crop
- 사용자는 UI의 “화면 조정”에서 ROI를 드래그로 미세 조정 가능(로컬스토리지 저장)

### 숫자 필드 정규화

#### possible (리롤 n회 가능)
- 한글/아이콘 때문에 오인식이 잦아, ROI 내부에서 숫자 영역만 sub-crop 후 OCR
- `normalize_possible()`로 0..5 정수로 정규화

#### cost (골드)
- `normalize_cost()`로 {0, 900, 1800} 중 가장 가까운 값으로 스냅
- UI에서는 `cost_state`로 저장:
  - `-1 => 0G`, `0 => 900G`, `1 => 1800G`

#### count (가공횟수: (n/7) 또는 (n/9))
- **중요 규칙:** 운영 타겟은 희귀/영웅만이라 **분모는 7/9만 지원**.  
  고급젬 같은 `/5` 케이스는 의도적으로 **미지원(=count None 처리)**.
- EasyOCR이 `(7/9)`를 `19`, `1284/52` 같은 쓰레기로 뱉는 케이스가 많아서:
  - `gemmini_vision.detect.read_count_text_with_multicrop()`가 ROI 내부에서
    `full/wide/right/center/tight` 여러 crop 후보로 OCR → **7/9가 포함된 후보만 채택**

### 옵션(category/value) 추출
- `detect_option()`이 색상 기반으로 카테고리를 추정 + 한글 OCR로 옵션명을 fuzzy match + 숫자(+/-) 추출
- 알려진 리스크:
  - 옵션 ROI 중앙에는 초록 화살표/텍스트가 섞여 **색상 기반 카테고리 오판**이 발생할 수 있음
  - 개선 방향(메모): “중앙 평균색” 대신 테두리/배경 ring 샘플링이 더 안정적

## 6) 게임 상태 변환 규칙(`ui_state`)

`gemmini_vision/parser.py:convert_to_ui_state()`

- `attempts_left`: `count`의 **분자(n)** 를 그대로 사용 (9/9 시작 → 8/9 → … 감소)
- `rerolls`(리롤 횟수):
  - 시작 턴(7/7)은 1, 9/9는 2로 고정(초기 상태에서 possible OCR이 흔들리는 문제 방지)
  - 진행 중에는 `possible` OCR 값을 우선 사용
- `options`: 4개 옵션을 `[text, category, raw_option, value]` 형태로 UI에 전달

## 7) RL 추천 + 성공확률(Goal Success)

### RL 추천
- `app/server.py:RLRuntime`가 `models/production/gemmini_v9/best_model.zip` 로딩
- `recommend()`가 `process/reroll/stop` 중 하나를 선택(액션 마스크 적용)

### 성공확률(목표 달성 확률)
- 현재는 **MC rollouts 기반**:
  - 기본: `GEMMINI_GOAL_MC_ROLLOUTS` (default 256)
  - false-zero 방지:
    - 0%면 `first_action_probe`(각 첫 행동 강제) + (always_process/random_valid) 전략 probe로 보조
- UX 규칙:
  - `success_prob == 0`이면 추천 행동을 강제로 `stop`으로 override (`action_overridden_by="goal_success_zero"`)

## 8) 기록/제보 저장 규칙

### 기록(Analyze)
- 버킷: `app/records/YYYY-MM-DD/`
- 저장:
  - `json/`: 분석 결과(항상 저장)
  - `images/`: fullframe 업로드일 때만 저장(multi-crop 기본 경로에서는 저장 안 됨)
  - 기본값으로 **auto-retry fullframe(`capture_retry_*`)는 이미지 저장을 생략**해서 디스크 사용량을 줄임 (`GEMMINI_PERSIST_AUTORETRY_FULLFRAME_IMAGE=1`이면 저장)

### 버그 제보(`/api/report`)
- `app/records/YYYY-MM-DD/message/*.json` 저장
- 중요한 설계:
  - analyze 호출마다 ROI를 디스크에 저장하지 않고,
  - **최근 analyze 입력을 client_id 기준 in-memory cache에 보관**했다가,
  - report 시점에만 ROI/분석 JSON을 `message/<bundle>/...`로 “증거 번들”로 저장

## 9) 보안/운영 규칙(중요)

- `GEMMINI_PROXY_SHARED_SECRET` 설정 시:
  - 프록시에서 `x-gemmini-key` 없으면 401
  - 이때만 `x-gemmini-client-ip`를 신뢰(그 외에는 스푸핑 가능)
- Rate limit은 `client_id`가 있으면 `cid:<id>`로, 없으면 `ip:<ip>`로 묶임
- `/api/report`는 제보 전송 안정성을 위해 analyze와 분리된 key(`...:report_post`)를 사용
- 토큰/시크릿(ngrok token, GitHub PAT 등)은 **절대 저장소/문서에 커밋하지 않기**

## 10) 운영/디버깅 팁

### “content-rect/homography 방향성” 검증 도구
- `conda run -n gem python codex/overlay_rois_on_records.py --input-glob 'app/records/**/images/*' --out-dir codex/roi_overlay_records_all_latest`
- 결과:
  - `ok/`에는 패널+ROI가 잘 잡힌 오버레이
  - `fail/`은 대부분 “젬 패널 없는 화면(공유 선택/설정창/다른 창)”이라 실패가 정상인 경우가 많음

### UI 변경 확인(재배포 없이)
- AI 서버의 `/`는 `frontend/index.html`이 존재하면 그걸 우선 서빙함.
- 즉, 로컬에서 프론트 수정 후 AI 서버만 띄워도 브라우저로 바로 확인 가능.

---

### 새 세션에서 가장 먼저 확인할 것(체크리스트)
1. Vercel env: `AI_BACKEND_URL`, `API_SHARED_SECRET`
2. AI 서버 env: `GEMMINI_PROXY_SHARED_SECRET` (Vercel과 동일)
3. `app/notice.json` 갱신이 즉시 반영되는지(`/api/notice`)
4. analyze에서 `debug.selection.ocr_mode == "multi_crop"`인지
5. records 저장 레이아웃이 `app/records/YYYY-MM-DD/...`로 유지되는지

## 11) 문서 유지 규칙 (중요)

이 문서(`codex/HANDOFF.md`)는 “자동으로 최신”이 되지 않습니다.  
**코드/규칙/경로/환경변수/모델이 변경되면 반드시 이 문서도 같이 업데이트**해야 다음 세션에서 삽질이 줄어듭니다.

권장 운영 규칙:
1. 기능 변경/버그 수정/배포 방식 변경 시, PR/커밋에 `codex/HANDOFF.md` 업데이트를 포함한다.
2. 변경 항목이 아래에 해당하면 반드시 반영한다.
   - API 인터페이스(`/api/*` 입력/출력/필드명)
   - ROI 스키마/좌표/letterbox 매핑 방식
   - OCR 후처리/정규화 규칙(count/possible/cost)
   - RL 모델 경로/모델 교체/평가 규칙
   - records 저장 레이아웃/버킷 방식
   - 보안(시크릿/헤더/인증) 및 rate limit 규칙
3. 새 세션 시작 시에는 항상:
   - `README.md`, `codex/HANDOFF.md`를 읽고
   - `git log -n 20`, `git diff`, `rg "TODO|FIXME|HACK"`로 최근 변화/리스크를 확인한 뒤 작업을 시작한다.
