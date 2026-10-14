// Pure zoom/pan math for the image viewport. No DOM access, so it is testable in node.

export const MIN_SCALE = 0.05;
export const MAX_SCALE = 32;

/** Transform that fits an image inside a viewport with `padding` CSS px around it. */
export function fitTransform(imgW, imgH, vpW, vpH, padding = 24) {
  const availW = Math.max(1, vpW - padding * 2);
  const availH = Math.max(1, vpH - padding * 2);
  const scale = clamp(Math.min(availW / imgW, availH / imgH), MIN_SCALE, MAX_SCALE);
  return {
    scale,
    tx: (vpW - imgW * scale) / 2,
    ty: (vpH - imgH * scale) / 2,
  };
}

/** Zoom by `factor` keeping the image point under viewport position (px, py) fixed. */
export function zoomAt(t, factor, px, py, min = MIN_SCALE, max = MAX_SCALE) {
  const scale = clamp(t.scale * factor, min, max);
  const ratio = scale / t.scale;
  return {
    scale,
    tx: px - (px - t.tx) * ratio,
    ty: py - (py - t.ty) * ratio,
  };
}

/** Keep the image from being pushed out of view: center when it fits, otherwise clamp edges. */
export function clampTransform(t, imgW, imgH, vpW, vpH) {
  const sw = imgW * t.scale;
  const sh = imgH * t.scale;
  return {
    scale: t.scale,
    tx: sw <= vpW ? (vpW - sw) / 2 : clamp(t.tx, vpW - sw, 0),
    ty: sh <= vpH ? (vpH - sh) / 2 : clamp(t.ty, vpH - sh, 0),
  };
}

export function pan(t, dx, dy) {
  return { scale: t.scale, tx: t.tx + dx, ty: t.ty + dy };
}

/** Wheel event -> zoom factor. Trackpad pinches arrive with ctrlKey and small deltas. */
export function wheelZoomFactor(deltaY, pinch) {
  const k = pinch ? 0.01 : 0.0015;
  return Math.exp(-deltaY * k);
}

/** Map a client (screen) position into image pixel coordinates using the canvas's screen rect. */
export function imagePointFromClient(clientX, clientY, rect, imgW, imgH) {
  const sx = imgW / rect.width;
  const sy = imgH / rect.height;
  return {
    x: (clientX - rect.left) * sx,
    y: (clientY - rect.top) * sy,
    inside: clientX >= rect.left && clientX < rect.right && clientY >= rect.top && clientY < rect.bottom,
  };
}

/** On-screen diameter of a brush of `brushPx` image pixels at the given screen scale. */
export function brushPreviewDiameter(brushPx, scale) {
  return brushPx * scale;
}

export function clamp(v, lo, hi) {
  return Math.min(hi, Math.max(lo, v));
}
