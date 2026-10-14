// Thin client for the local backend.

async function throwForStatus(res) {
  let detail = `${res.status} ${res.statusText}`;
  try {
    const body = await res.json();
    if (body && body.detail) detail = body.detail;
  } catch (_) { /* not JSON */ }
  const err = new Error(detail);
  err.status = res.status;
  throw err;
}

export async function health() {
  const res = await fetch('/api/health', { cache: 'no-store' });
  if (!res.ok) await throwForStatus(res);
  return res.json();
}

/** @returns {Promise<Blob>} grayscale PNG where white = selected */
export async function segment(imageBlob, points) {
  const form = new FormData();
  form.append('image', imageBlob, 'image.png');
  form.append('points', JSON.stringify(points));
  const res = await fetch('/api/segment', { method: 'POST', body: form });
  if (!res.ok) await throwForStatus(res);
  return res.blob();
}

/** @returns {Promise<Blob>} RGB PNG of the inpainted image */
export async function inpaint(imageBlob, maskBlob, { quality = 'fast', strength = 0.6 } = {}) {
  const form = new FormData();
  form.append('image', imageBlob, 'image.png');
  form.append('mask', maskBlob, 'mask.png');
  form.append('quality', quality);
  form.append('strength', String(strength));
  const res = await fetch('/api/inpaint', { method: 'POST', body: form });
  if (!res.ok) await throwForStatus(res);
  return res.blob();
}

export async function loadModel(name) {
  const res = await fetch(`/api/models/${encodeURIComponent(name)}/load`, { method: 'POST' });
  if (!res.ok) await throwForStatus(res);
  return res.json();
}

export async function progress() {
  const res = await fetch('/api/progress', { cache: 'no-store' });
  if (!res.ok) await throwForStatus(res);
  return res.json();
}
