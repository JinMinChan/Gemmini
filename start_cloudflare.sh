#!/usr/bin/env bash
#
# Cloudflare Quick Tunnel 실행 스크립트
#
# 목적:
# - 로컬 FastAPI(기본: http://127.0.0.1:8000)를 외부(Vercel)에서 접근 가능하게 노출
# - 인바운드 포트포워딩이 막힌(학교/상위망) 환경에서도 동작
#
# 사용법:
#   ./start_cloudflare.sh                 # 기본: http://127.0.0.1:8000
#   ./start_cloudflare.sh http://127.0.0.1:8000
#
# 출력되는 https://xxxx.trycloudflare.com URL을 Vercel의 AI_BACKEND_URL에 넣으면 됩니다.
# (cloudflared를 껐다 켜면 URL이 바뀝니다. python 서버만 재시작하면 URL은 유지됩니다.)

set -euo pipefail

TARGET_URL="${1:-http://127.0.0.1:8000}"
TARGET_URL="${TARGET_URL%/}"
HEALTH_URL="${TARGET_URL}/api/health"
# Many networks block UDP/QUIC. Default to HTTP/2 (TCP) for stability.
PROTOCOL="${CLOUDFLARED_PROTOCOL:-http2}"

if ! command -v cloudflared >/dev/null 2>&1; then
  cat <<'EOF'
Error: cloudflared가 설치되어 있지 않습니다.

Ubuntu(x86_64) 설치 예시:
  curl -L -o /tmp/cloudflared.deb \
    https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb
  sudo dpkg -i /tmp/cloudflared.deb
  cloudflared --version

sudo가 어렵다면(또는 간단히 로컬 설치로):
  mkdir -p ~/.local/bin
  curl -L -o ~/.local/bin/cloudflared \
    https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64
  chmod +x ~/.local/bin/cloudflared
  ~/.local/bin/cloudflared --version
EOF
  exit 1
fi

echo "Checking local health: ${HEALTH_URL}"
if command -v curl >/dev/null 2>&1; then
  if ! curl -fsS "${HEALTH_URL}" >/dev/null 2>&1; then
    echo "WARNING: 로컬 서버가 ${TARGET_URL} 에서 응답하지 않습니다."
    echo "  다른 터미널에서 서버를 먼저 켜주세요:"
    echo "    conda run -n gem python run_server.py"
    echo ""
  fi
fi

if [ "${GEMMINI_SKIP_CF_PREFLIGHT:-0}" != "1" ] && command -v dig >/dev/null 2>&1 && command -v nc >/dev/null 2>&1; then
  echo "Checking egress to Cloudflare edge (TCP 7844)..."
  # Cloudflare Tunnel requires outbound connectivity to port 7844.
  # Some campus/corporate networks block this port, causing endless retries.
  ok=0
  while IFS= read -r host; do
    host="${host%.}"
    [ -z "$host" ] && continue
    if timeout 4s nc -vz -w 3 "$host" 7844 >/dev/null 2>&1; then
      ok=1
      break
    fi
  done < <(dig +short SRV _v2-origintunneld._tcp.argotunnel.com 2>/dev/null | awk '{print $4}' | head -n 4)

  if [ "$ok" -eq 0 ]; then
    cat <<'EOF'
ERROR: 이 네트워크에서 Cloudflare Tunnel 연결이 막혀 있습니다.
  - cloudflared는 Cloudflare edge의 TCP/UDP 7844 포트로 outbound 연결이 필요합니다.
  - 현재는 TCP 7844 연결이 타임아웃(차단)이라서 아래 같은 로그가 반복됩니다:
    "dial tcp ...:7844: i/o timeout"

해결 방법:
1) 다른 네트워크에서 실행 (가장 빠름): 휴대폰 핫스팟/집 인터넷 등
2) 네트워크 관리자에게 egress TCP/UDP 7844 허용 요청
3) (대안) Oracle/VPS에 reverse-ssh로 중계 서버를 두는 방식

원하면 사전 체크를 끄고 강행할 수 있습니다:
  GEMMINI_SKIP_CF_PREFLIGHT=1 ./start_cloudflare.sh
EOF
    exit 2
  fi
  echo "OK: TCP 7844 reachable."
  echo ""
fi

echo "Starting Cloudflare Quick Tunnel to ${TARGET_URL}"
echo "Transport protocol: ${PROTOCOL} (override with CLOUDFLARED_PROTOCOL=quic)"
echo "Press Ctrl+C to stop"
echo ""

# Print the trycloudflare URL as soon as it appears while keeping the process running.
# NOTE: Ubuntu 기본 awk(mawk)는 match(..., ..., array) 구문을 지원하지 않아서 bash regex로 파싱합니다.
found=0
cfg="$(mktemp /tmp/gemmini_cloudflared_quick.XXXX.yml)"
trap 'rm -f "$cfg"' EXIT
cat >"$cfg" <<YML
url: ${TARGET_URL}
protocol: ${PROTOCOL}
no-autoupdate: true
YML

while IFS= read -r line; do
  printf '%s\n' "$line"
  if [ "$found" -eq 0 ] && [[ "$line" =~ (https://[A-Za-z0-9.-]+\.trycloudflare\.com) ]]; then
    found=1
    url="${BASH_REMATCH[1]}"
    echo ""
    echo "============================================================"
    echo "Public URL: ${url}"
    echo "Vercel AI_BACKEND_URL = ${url}"
    echo "============================================================"
    echo ""
  fi
done < <(cloudflared tunnel --config "$cfg" 2>&1)
