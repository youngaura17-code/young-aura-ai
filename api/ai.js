// Young Aura AI proxy — Vercel serverless function.
// The API keys live ONLY in Vercel env vars (NVIDIA_API_KEY, GROQ_API_KEY).
// They are never shipped to the browser. The five Young Aura products call
// this endpoint; it picks a working model, with fallbacks, and returns text.
//
// NVIDIA (primary) — verified working on this key (live generation, 2026-10-31):
//   text: openai/gpt-oss-20b, meta/muse-glimmer-30b, minimaxai/minimax-m3.1-flash
//   vision: meta/llama-3.2-90b-vision-instruct
// Most other listed NVIDIA models 404 ("Function not found for account").
// GROQ (fallback, added 2026-10-31) — verified live, and FAST (0.4–0.6s):
//   text: openai/gpt-oss-120b, openai/gpt-oss-20b, qwen/qwen3.8-27b
//   No vision model on this Groq key, so vision tasks stay NVIDIA-only.

const NVIDIA_MODELS = [
  // Order matters: measured from a warm state, gpt-oss-20b is fastest
  // (~0.75s), then glimmer (~2s), then minimax (~2.9s).
  'openai/gpt-oss-20b',
  'meta/muse-glimmer-30b',
  'minimaxai/minimax-m3.1-flash',
];
const NVIDIA_VISION_MODEL = 'meta/llama-3.2-90b-vision-instruct';
const NVIDIA_URL = 'https://integrate.api.nvidia.com/v1/chat/completions';

const GROQ_MODELS = [
  'openai/gpt-oss-120b',
  'openai/gpt-oss-20b',
  'qwen/qwen3.8-27b',
];
const GROQ_URL = 'https://api.groq.com/openai/v1/chat/completions';

// Vercel Hobby kills functions at ~10s, and a cold NVIDIA model can take
// 8s+ on its own. So: one global 9s deadline for the whole request —
// each model attempt gets only the time left, and we stop trying rather
// than hanging. Groq answers in under a second, so the fallback almost
// always fits even when NVIDIA burns most of the budget.
// BUG FOUND 2026-10-31 (live test): a cold NVIDIA attempt consumed the
// whole 9s deadline and aborted, so Groq NEVER ran and the request 502'd
// even though the fallback was healthy. Fix: NVIDIA gets at most ~4.5s
// of the budget in total (one attempt, capped at 4s) — after that the
// loop skips to Groq, which answers in under a second.

// Tiny per-IP rate limit so one visitor can't drain the free keys.
const hits = new Map();
function rateLimited(ip) {
  const now = Date.now();
  const rec = hits.get(ip) || { count: 0, reset: now + 60000 };
  if (now > rec.reset) { rec.count = 0; rec.reset = now + 60000; }
  rec.count += 1;
  hits.set(ip, rec);
  return rec.count > 30;
}

function cors(res) {
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'POST, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type');
}

async function callChat(url, key, label, model, messages, maxTokens, msLeft) {
  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), Math.max(msLeft, 500));
  try {
    const r = await fetch(url, {
      method: 'POST',
      headers: {
        'Authorization': 'Bearer ' + key,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        model,
        messages,
        max_tokens: maxTokens,
        temperature: 0.7,
      }),
      signal: ctrl.signal,
    });
    if (!r.ok) {
      const t = await r.text().catch(() => '');
      const err = new Error(label + ' ' + r.status + ' ' + t.slice(0, 200));
      err.status = r.status;
      throw err;
    }
    const d = await r.json();
    const text = d && d.choices && d.choices[0] && d.choices[0].message
      ? d.choices[0].message.content || ''
      : '';
    if (!text) throw new Error('empty completion');
    return text;
  } finally {
    clearTimeout(timer);
  }
}

module.exports = async (req, res) => {
  cors(res);
  if (req.method === 'OPTIONS') return res.status(204).end();
  if (req.method !== 'POST') return res.status(405).json({ error: 'POST only' });

  const ip = (req.headers['x-forwarded-for'] || '').split(',')[0].trim() || 'unknown';
  if (rateLimited(ip)) return res.status(429).json({ error: 'Too many AI requests — try again in a minute.' });
  if (!process.env.NVIDIA_API_KEY && !process.env.GROQ_API_KEY) {
    return res.status(500).json({ error: 'AI not configured yet.' });
  }

  let body = req.body;
  if (typeof body === 'string') { try { body = JSON.parse(body); } catch { body = {}; } }
  body = body || {};
  const task = String(body.task || 'text').slice(0, 40);
  const prompt = String(body.prompt || '').slice(0, 6000);
  const image = typeof body.image === 'string' ? body.image.slice(0, 2000000) : null;
  const maxTokens = Math.min(Math.max(parseInt(body.maxTokens, 10) || 700, 300), 1200);
  if (!prompt && !image) return res.status(400).json({ error: 'prompt or image required' });

  const system = 'You are the AI engine inside a Young Aura product for Indian small businesses and creators. '
    + 'Answer in the language the user writes in (English or Hinglish). Be specific and practical, never vague. '
    + 'Return only what was asked for — no preamble, no disclaimers.';

  const isVision = task === 'vision' && image;
  const textMessages = [
    { role: 'system', content: system },
    { role: 'user', content: prompt },
  ];
  const visionMessages = [
    { role: 'system', content: system },
    { role: 'user', content: [
      { type: 'text', text: prompt || 'Describe this screenshot in one line, then list 5 short tags (lowercase, comma-separated) that would help find it later.' },
      { type: 'image_url', image_url: { url: image } },
    ]},
  ];

  // Attempt list, in order: NVIDIA first, Groq as fallback (text tasks only —
  // no vision model exists on this Groq key, so a vision task never goes to Groq).
  const attempts = [];
  if (process.env.NVIDIA_API_KEY) {
    if (isVision) attempts.push({ provider: 'nvidia', url: NVIDIA_URL, key: process.env.NVIDIA_API_KEY, model: NVIDIA_VISION_MODEL, messages: visionMessages });
    for (const model of NVIDIA_MODELS) {
      attempts.push({ provider: 'nvidia', url: NVIDIA_URL, key: process.env.NVIDIA_API_KEY, model, messages: isVision ? visionMessages : textMessages });
    }
  }
  if (!isVision && process.env.GROQ_API_KEY) {
    for (const model of GROQ_MODELS) {
      attempts.push({ provider: 'groq', url: GROQ_URL, key: process.env.GROQ_API_KEY, model, messages: textMessages });
    }
  }

  const deadline = Date.now() + 9000;
  const reqStart = Date.now();
  const debug = [];
  let skipProvider = null; // a 400/401/403 means THIS provider's key or request is broken — skip its other models, but still try the next provider
  for (const a of attempts) {
    if (skipProvider && a.provider === skipProvider) continue;
    // NVIDIA phase budget: NVIDIA gets at most the first ~4.5s of the
    // request. An NVIDIA attempt may only start while enough of that
    // budget remains for it to finish; afterwards the loop hands over to
    // Groq (fast) instead of aborting the request.
    const elapsed = Date.now() - reqStart;
    const msLeft = deadline - Date.now();
    if (msLeft < 800) break;
    let cap = msLeft;
    if (a.provider === 'nvidia') {
      const nvidiaLeft = 4500 - elapsed;
      if (nvidiaLeft < 800) continue;
      cap = Math.min(msLeft, nvidiaLeft);
    }
    const t0 = Date.now();
    try {
      const text = await callChat(a.url, a.key, a.provider, a.model, a.messages, maxTokens, cap);
      return res.status(200).json({ text, model: a.model, provider: a.provider });
    } catch (e) {
      debug.push(a.provider + '/' + a.model + ':' + (Date.now() - t0) + 'ms:' + String(e.message).slice(0, 120));
      // 404 = model not enabled on this key; 429/5xx = busy. Try the next model.
      if (e.status && e.status !== 404 && e.status !== 429 && e.status < 500) skipProvider = a.provider;
    }
  }
  return res.status(502).json({ error: 'AI is busy right now — please try again in a moment.', debug: debug.join(' | ') || 'no error captured' });
};
