export const config = { runtime: 'edge' };

function json(data, status = 200) {
  return new Response(JSON.stringify(data), {
    status,
    headers: { 'content-type': 'application/json; charset=utf-8' },
  });
}

function getBackendBase() {
  const raw = (process.env.AI_BACKEND_URL || '').trim();
  return raw.replace(/\/+$/, '');
}

export default async function handler(request) {
  if (request.method !== 'POST') {
    return json({ ok: false, detail: 'Method Not Allowed' }, 405);
  }

  const backend = getBackendBase();
  if (!backend) {
    return json({ ok: false, detail: 'AI_BACKEND_URL is not configured' }, 500);
  }

  try {
    const form = await request.formData();
    const headers = {};
    const shared = (process.env.API_SHARED_SECRET || '').trim();
    if (shared) {
      headers['x-gemmini-key'] = shared;
    }

    const upstream = await fetch(`${backend}/api/report`, {
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
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    return json({ ok: false, detail: `proxy report failed: ${msg}` }, 502);
  }
}
