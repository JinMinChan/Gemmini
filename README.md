# Gemmini 프로젝트 구조 설명서

이 문서는 배포 번들의 **파일별 역할과 책임**을 설명합니다.  
실행 방법보다, 어떤 파일이 어떤 기능을 수행하는지 중심으로 정리했습니다.

## 1. 최상위 엔트리/설정 파일

### `run_server.py`
- Uvicorn 실행 진입점입니다.
- `app.server:app`(FastAPI 앱 객체)을 바인딩해 HTTP 서버를 시작합니다.
- 서비스 프로세스를 띄우는 가장 얇은 부트스트랩 파일입니다.

### `start_ngrok.sh`
- 로컬 `127.0.0.1:8000` 서버를 외부에 노출하기 위한 ngrok 실행 스크립트입니다.
- authtoken 설정 여부를 검사하고, 서버 헬스체크(`GET /api/health`)까지 확인합니다.
- 운영 코드가 아니라 배포/접속 편의용 보조 스크립트입니다.

### `requirements.txt`
- 서버 실행에 필요한 Python 의존성 단일 목록입니다.
- 웹 프레임워크(FastAPI/Uvicorn), OCR(OpenCV/EasyOCR), RL 추론(Torch/SB3) 의존성이 모두 포함되어 있습니다.

## 2. Vercel 배포 루트 분리 (`frontend/`)

Vercel이 저장소 루트의 `requirements.txt`를 보고 Python 의존성(예: `torch`, `easyocr`, `opencv-python`)까지 설치하며
빌드가 느려지는 문제를 피하기 위해, Vercel 배포용 파일을 `frontend/`로 별도 분리했습니다.

- Vercel Project 설정에서 **Root Directory를 `frontend`로 지정**해서 배포합니다.
- `frontend/`에는 정적 `index.html`과 `api/*`(Edge 프록시)만 포함됩니다.
- `frontend/package.json`은 ESM(`type: module`)로 Edge Function 번들링 경고를 줄이기 위한 최소 설정입니다.
- AI 서버(FastAPI)는 기존대로 저장소 루트에서 실행합니다.

### `frontend/index.html`
- Vercel에서 서빙되는 단일 페이지 프론트엔드(UI+JS)입니다.
- `/api/analyze`, `/api/report`로 요청을 보내며, Vercel Edge 프록시가 이를 AI 서버로 전달합니다.

### `frontend/vercel.json`
- Vercel 배포 시 루트(`/`)를 `frontend/index.html`로 연결하는 라우팅/헤더 설정입니다.
- 정적 캐시가 과도하게 남지 않도록 `Cache-Control: no-store` 정책을 포함합니다.

## 3. Vercel 프록시 계층 (`frontend/api/`)

### `frontend/api/analyze.js`
- Vercel Edge Function입니다.
- 브라우저 multipart 요청을 AI 서버의 `/api/analyze`로 전달합니다.
- `API_SHARED_SECRET`가 있으면 `x-gemmini-key` 헤더를 붙여 인증합니다.

### `frontend/api/report.js`
- Vercel Edge Function입니다.
- 사용자 제보를 AI 서버의 `/api/report`로 전달합니다.

### `frontend/api/health.js`
- Vercel 레이어 헬스체크용 경량 엔드포인트입니다.

## 4. API 계층 (`app/`)

### `app/server.py`
- 서비스의 핵심 백엔드입니다.
- 주요 책임:
- 화면 캡처 이미지 입력 검증/디코딩
- OCR 파서 호출 및 UI 상태 변환
- RL 모델 추론(추천 행동 산출)
- 목표 성공 확률 Monte Carlo 추정
- 분석 결과/이미지/버그제보를 IP 단위 폴더로 저장
- 선택적 프록시 인증 검증 (`GEMMINI_PROXY_SHARED_SECRET`)
- FastAPI 라우팅:
- `/` 정적 페이지 반환
- `/api/health` 헬스체크
- `/api/upload` 업로드 저장
- `/api/analyze` OCR+RL 통합 분석
- `/api/report` 버그/개선 제보 저장

### `app/static/index.html`
- 단일 페이지 프론트엔드(UI+JS)입니다.
- 주요 책임:
- 화면 공유 시작/중지
- 기본 경로: ROI 7개(`option1~4`, `possible`, `cost`, `count`)를 crop하여 `/api/analyze` 전송
- fallback 경로: ROI 캡처 실패 시 전체 프레임을 JPEG 압축/리사이즈 후 `/api/analyze` 전송
- OCR 결과/추천 행동/성공 확률 표시
- 수동 스탯 조작(현재/목표 값) 및 옵션 반영
- 버그 제보 입력 후 `/api/report` 전송

### `app/__init__.py`
- `app` 디렉터리를 Python 패키지로 인식시키는 초기화 파일입니다.

### `app/records/.gitkeep`
- 실행 중 생성되는 기록 데이터 루트 디렉터리의 자리표시자입니다.
- 실제 운영 시 `records/<ip>/images|json|message`가 동적으로 생성됩니다.

### `app/uploads/.gitkeep`
- 업로드 임시 디렉터리 자리표시자입니다.

## 5. OCR 계층 (`gemmini_vision/`)

### `gemmini_vision/detect.py`
- OCR 엔진 및 텍스트/숫자 인식 유틸리티를 담당합니다.
- 주요 기능:
- EasyOCR 리더 초기화(`ko`, `en`)
- ROI 전처리/다중 전처리(숫자 인식 안정화)
- 옵션/횟수/비용 텍스트 정규화
- OCR 오인식 보정용 후처리

### `gemmini_vision/parser.py`
- `detect.py` 결과를 게임 상태 구조로 변환하는 어댑터입니다.
- 주요 기능:
- ROI별 OCR 실행 및 결과 통합
- 옵션 리스트를 UI 상태 형식으로 매핑
- `possible/cost/count` 값을 서비스에서 쓰는 상태값으로 변환

### `gemmini_vision/__init__.py`
- OCR 패키지 초기화 파일입니다.

## 6. RL 정책/환경 계층 (`gem_core/`)

### `gem_core/role_env.py`
- 역할(딜러/서폿), 젬 타입을 포함한 강화학습 환경 정의 파일입니다.
- 주요 기능:
- 상태/행동/관측 벡터 정의
- 옵션 생성, 상태 전이, 종료 조건
- 보상(잠재함수/보조신호) 계산
- 서버의 정책 추론과 성공확률 롤아웃 시뮬레이션 기반 환경 제공

### `gem_core/__init__.py`
- RL 코어 패키지 초기화 파일입니다.

## 7. 모델 아티팩트 (`models/`)

### `models/production/gemmini_v9/best_model.zip`
- 현재 서버에서 사용하는 단일 운영 모델입니다.
- `app/server.py`의 기본 모델 경로가 이 파일을 직접 참조합니다.
- 배포 번들에는 이 모델 하나만 포함합니다.

## 8. 배포 아키텍처 (권장)

- 브라우저 -> Vercel(`api/analyze`, `api/report`) -> AI 서버(FastAPI)
- Vercel은 웹페이지/중계만 담당하고, OCR+AI 연산은 AI 서버가 담당합니다.

### Vercel 환경변수
- `AI_BACKEND_URL`: AI 서버 주소 (예: `https://your-ai.example.com`)
- `API_SHARED_SECRET`: Vercel -> AI 서버 인증용 시크릿

### AI 서버 환경변수
- `GEMMINI_PROXY_SHARED_SECRET`: Vercel의 `API_SHARED_SECRET`와 동일 값

## 9. 이미지 압축 정확도 & 동시성 메모

### 이미지 압축 정확도
- 기본값은 OCR 정확도를 해치지 않도록 보수적으로 설정했습니다.
- 긴 변 1920 기준 리사이즈 + JPEG 품질 0.86 시작, 최대 약 1.6MB 목표로 점진 압축합니다.
- 일반적인 1440p 캡처에서는 정확도 손실이 크지 않지만, UI 스케일이 너무 작으면 오인식 가능성은 남습니다.

### 동시 요청
- Vercel은 동시 요청 수용이 강하지만, 실제 병목은 AI 서버 OCR/RL 연산입니다.
- 동시 사용자가 늘면 지연은 증가할 수 있습니다.
- 운영 시 권장:
1. `GEMMINI_GOAL_MC_ROLLOUTS` 조정 (예: 256 -> 128)
2. AI 서버 CPU 확장 또는 인스턴스 분리
3. 요청 큐/레이트리밋 강화

## 10. 현재 정리 원칙

- 폴더명은 역할 기준으로 분리:
- `app`: 웹/API
- `gemmini_vision`: OCR
- `gem_core`: RL 환경/정책 지원
- 불필요 산출물 제거:
- `__pycache__`, 과거 debug 로그, 학습 체크포인트/평가 이미지 미포함
- 운영 모델 단일화:
- 레거시 fallback 모델 제거, 현재 사용 모델만 유지
