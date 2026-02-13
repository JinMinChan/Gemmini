export const config = { runtime: 'edge' };

export default async function handler() {
  return new Response(
    JSON.stringify({ ok: true, layer: 'vercel-proxy' }),
    {
      status: 200,
      headers: { 'content-type': 'application/json; charset=utf-8', 'cache-control': 'no-store' },
    },
  );
}
