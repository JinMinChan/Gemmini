export const config = { runtime: 'edge' };

function json(data, status = 200) {
  return new Response(JSON.stringify(data), {
    status,
    headers: { 'content-type': 'application/json; charset=utf-8', 'cache-control': 'no-store' },
  });
}

function getBackendBase() {
  const raw = (process.env.AI_BACKEND_URL || '').trim();
  return raw.replace(/\/+$/, '');
}

export default async function handler() {
  const backend = getBackendBase();
  if (!backend) {
    return json({ ok: false, layer: 'vercel-proxy', backend_ok: false, detail: 'AI_BACKEND_URL is not configured' }, 500);
  }

  try {
    const upstream = await fetch(`${backend}/api/health`, {
      method: 'GET',
      headers: {
        'cache-control': 'no-store',
        'ngrok-skip-browser-warning': 'true',
      },
    });
    const body = await upstream.json().catch(() => ({}));
    const backendOk = Boolean(upstream.ok && body && body.ok);
    if (!backendOk) {
      const detail = body?.detail || body?.error || upstream.statusText || 'backend unhealthy';
      return json({ ok: false, layer: 'vercel-proxy', backend_ok: false, backend_status: upstream.status, detail }, 503);
    }
    return json({ ok: true, layer: 'vercel-proxy', backend_ok: true, backend_status: upstream.status }, 200);
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    return json({ ok: false, layer: 'vercel-proxy', backend_ok: false, detail: `backend unreachable: ${msg}` }, 503);
  }
}
