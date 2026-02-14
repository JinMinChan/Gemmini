export const config = { runtime: 'edge' };

function json(data, status = 200) {
  return new Response(JSON.stringify(data), {
    status,
    headers: { 'content-type': 'application/json; charset=utf-8', 'cache-control': 'no-store' },
  });
}

function firstForwardedIp(value) {
  const raw = (value || '').trim();
  if (!raw) return '';
  return raw.split(',')[0].trim();
}

function getClientIp(request) {
  const candidates = [
    'x-vercel-forwarded-for',
    'x-forwarded-for',
    'x-real-ip',
    'cf-connecting-ip',
  ];
  for (const name of candidates) {
    const ip = firstForwardedIp(request.headers.get(name));
    if (ip) return ip;
  }
  return '';
}

function getBackendBase() {
  const raw = (process.env.AI_BACKEND_URL || '').trim();
  return raw.replace(/\/+$/, '');
}

export default async function handler(request) {
  const backend = getBackendBase();
  if (!backend) {
    return json({ ok: false, detail: 'AI_BACKEND_URL is not configured' }, 500);
  }

  const shared = (process.env.API_SHARED_SECRET || '').trim();
  const headers = { 'cache-control': 'no-store' };
  if (shared) headers['x-gemmini-key'] = shared;
  const clientIp = getClientIp(request);
  if (clientIp) headers['x-gemmini-client-ip'] = clientIp;

  try {
    if (request.method === 'GET') {
      const url = new URL(request.url);
      const limit = url.searchParams.get('limit');
      const qs = limit ? `?limit=${encodeURIComponent(limit)}` : '';
      const upstream = await fetch(`${backend}/api/letters${qs}`, { method: 'GET', headers });
      const text = await upstream.text();
      return new Response(text, {
        status: upstream.status,
        headers: {
          'content-type': upstream.headers.get('content-type') || 'application/json; charset=utf-8',
          'cache-control': 'no-store',
        },
      });
    }

    if (request.method === 'POST') {
      const form = await request.formData();
      const upstream = await fetch(`${backend}/api/letters`, {
        method: 'POST',
        headers,
        body: form,
      });
      const text = await upstream.text();
      return new Response(text, {
        status: upstream.status,
        headers: {
          'content-type': upstream.headers.get('content-type') || 'application/json; charset=utf-8',
          'cache-control': 'no-store',
        },
      });
    }

    return json({ ok: false, detail: 'Method Not Allowed' }, 405);
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    return json({ ok: false, detail: `proxy letters failed: ${msg}` }, 502);
  }
}

