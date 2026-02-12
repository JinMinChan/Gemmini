#!/bin/bash
# Ubuntu용 ngrok 실행 스크립트
# 사용법: ./start_ngrok.sh [AUTHTOKEN]

set -e

AUTHTOKEN="${1:-}"

if [ -z "$AUTHTOKEN" ]; then
    # authtoken이 이미 설정되어 있는지 확인
    if ! grep -q "authtoken:" ~/.config/ngrok/ngrok.yml 2>/dev/null; then
        echo "Error: ngrok authtoken이 설정되지 않았습니다."
        echo ""
        echo "사용법:"
        echo "  1. https://dashboard.ngrok.com/get-started/your-authtoken 에서 토큰 복사"
        echo "  2. ./start_ngrok.sh <YOUR_AUTHTOKEN>"
        echo ""
        echo "또는 직접 설정:"
        echo "  ngrok config add-authtoken <YOUR_AUTHTOKEN>"
        exit 1
    fi
else
    # authtoken 설정
    echo "Setting ngrok authtoken..."
    ngrok config add-authtoken "$AUTHTOKEN"
fi

echo "Starting ngrok tunnel to http://127.0.0.1:8000..."
echo "Press Ctrl+C to stop"
echo ""
echo "공개 URL은 아래에 표시됩니다:"
echo "브라우저 경고를 우회하려면 curl에 -H 'ngrok-skip-browser-warning: 1' 헤더 추가"
echo ""

# 서버가 실제로 떠 있는지 간단히 확인 (ngrok은 서버 없이도 켜질 수 있어서 혼동 방지용)
if command -v curl >/dev/null 2>&1; then
    if ! curl -fsS "http://127.0.0.1:8000/api/health" >/dev/null 2>&1; then
        echo "WARNING: 서버가 http://127.0.0.1:8000 에서 응답하지 않습니다."
        echo "  다른 터미널에서 서버를 먼저 켜주세요:"
        echo "    conda run -n gem python run_server.py"
        echo ""
    fi
fi

# ngrok 실행
ngrok http http://127.0.0.1:8000
