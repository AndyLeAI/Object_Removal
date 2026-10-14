import * as api from './api.js?v=21';
import {
  fitTransform, zoomAt, clampTransform, pan, wheelZoomFactor,
  imagePointFromClient, brushPreviewDiameter, MIN_SCALE, MAX_SCALE,
} from './viewport.js?v=21';

const $ = (id) => document.getElementById(id);
const fileInput = $('fileInput');
const dropZone = $('dropZone');
const stage = $('stage');
const viewport = $('viewport');
const canvasWrap = $('canvasWrap');
const emptyState = $('emptyState');
const imageCanvas = $('imageCanvas');
const maskCanvas = $('maskCanvas');
const pointLayer = $('pointLayer');
const imageCtx = imageCanvas.getContext('2d', { willReadFrequently: true });
const maskCtx = maskCanvas.getContext('2d', { willReadFrequently: true });
const brushSize = $('brushSize');
const brushValue = $('brushValue');
const brushPreview = $('brushPreview');
const removeBtn = $('removeBtn');
const undoBtn = $('undoBtn');
const clearBtn = $('clearBtn');
const downloadBtn = $('downloadBtn');
const statusText = $('statusText');
const backendText = $('backendText');
const progressBar = $('progressBar');
const imageInfo = $('imageInfo');
const maskInfo = $('maskInfo');
const hintText = $('hintText');
const processingOverlay = $('processingOverlay');
const processingTitle = $('processingTitle');
const processingHint = $('processingHint');
const modeSmartBtn = $('modeSmart');
const modeBrushBtn = $('modeBrush');
const zoomInBtn = $('zoomInBtn');
const zoomOutBtn = $('zoomOutBtn');
const zoomFitBtn = $('zoomFitBtn');
const zoomValueBtn = $('zoomValueBtn');
const qualityButtons = Array.from(document.querySelectorAll('.quality-btn'));
const qualityStatus = $('qualityStatus');
const strengthWrap = document.querySelector('.strength-inline');
const strengthInput = $('strength');
const strengthValue = $('strengthValue');
const processingBar = $('processingBar');
const processingTime = $('processingTime');

const MASK_COLOR = [138, 63, 252];
const MAX_SIDE = 3072;
const HISTORY_LIMIT = 12;
const ZOOM_STEP = 1.25;

const HINTS = {
  smart: 'Click an object to select it · Shift+click to exclude · Scroll to zoom · Space+drag to pan',
  brush: 'Paint over the object · Scroll to zoom · Space+drag to pan',
};

let backendReady = false;
let quality = 'fast';
let refiners = {};            // name -> status from /api/health
let healthTimer = null;
let hasImage = false;
let hasMask = false;
let busy = false;
let mode = 'smart';
let transform = { scale: 1, tx: 0, ty: 0 };
let fitMode = true;
let spaceHeld = false;
let panState = null;        // { pointerId, startX, startY, tx, ty }
let isDrawing = false;
let lastPoint = null;
let lastPointerClient = null;
let imageBlobPromise = null; // cached PNG of the current image for the backend
let selection = null;       // { points: [{x,y,label}], base: ImageData }
let segmentRequest = 0;
const history = [];

// ---------- status helpers ----------

function setStatus(message, progress = null) {
  statusText.textContent = message;
  if (typeof progress === 'number') {
    progressBar.style.width = `${Math.max(0, Math.min(100, progress * 100))}%`;
  }
}

let busyTimer = null;
let busyStart = 0;

const STAGE_TEXT = {
  lama: 'Big-LaMa is rebuilding the area',
  sdxl: 'SDXL is refining the fill',
  flux: 'FLUX.1 Fill is repainting the area',
};

async function tickProgress() {
  const secs = Math.round((performance.now() - busyStart) / 1000);
  processingTime.textContent = `${secs} s`;
  try {
    const p = await api.progress();
    if (p.stage) processingHint.textContent = STAGE_TEXT[p.stage] || p.stage;
    if (p.total > 0) {
      processingBar.style.width = `${Math.round((p.step / p.total) * 100)}%`;
      processingTime.textContent = `${secs} s · step ${p.step}/${p.total}`;
    }
  } catch (_) { /* backend busy; keep the timer running */ }
}

function setBusy(next, title = 'Removing…', hint = 'AI is rebuilding the selected area') {
  busy = next;
  if (busy) {
    busyStart = performance.now();
    processingBar.style.width = '0%';
    processingTime.textContent = '0 s';
    clearInterval(busyTimer);
    busyTimer = setInterval(tickProgress, 1000);
  } else {
    clearInterval(busyTimer);
    busyTimer = null;
  }
  document.body.classList.toggle('busy', busy);
  removeBtn.classList.toggle('removing', busy);
  removeBtn.textContent = busy ? 'Removing…' : 'Remove Object';
  processingOverlay.hidden = !busy;
  processingTitle.textContent = title;
  processingHint.textContent = hint;
  maskCanvas.classList.toggle('pulsing', busy && hasMask);
  updateButtons();
}

function qualityReady() {
  if (quality === 'fast') return true;
  const r = refiners[quality];
  return Boolean(r && r.loaded);
}

function updateButtons() {
  removeBtn.disabled = busy || !backendReady || !hasImage || !hasMask || !qualityReady();
  clearBtn.disabled = busy || !hasImage || !hasMask;
  undoBtn.disabled = busy || history.length === 0;
  downloadBtn.disabled = busy || !hasImage;
  maskInfo.textContent = hasMask ? 'Mask: painted' : 'Mask: empty';
}

// ---------- backend ----------

function renderQuality() {
  for (const btn of qualityButtons) btn.classList.toggle('active', btn.dataset.quality === quality);
  strengthWrap.hidden = quality !== 'sdxl';
  qualityStatus.classList.remove('loading', 'error');
  if (quality === 'fast') {
    qualityStatus.textContent = 'Big-LaMa · seconds';
    return;
  }
  const r = refiners[quality];
  if (!r) {
    qualityStatus.textContent = 'Waiting for backend…';
  } else if (r.error) {
    qualityStatus.classList.add('error');
    qualityStatus.textContent = `Load failed: ${r.error}`;
    qualityStatus.title = r.error;
  } else if (r.loaded) {
    qualityStatus.textContent = quality === 'flux' ? 'FLUX.1 Fill ready · minutes per removal' : 'SDXL ready · ~1 min per removal';
  } else if (r.loading) {
    qualityStatus.classList.add('loading');
    qualityStatus.textContent = `Loading ${r.label}… first time downloads ~${r.download_gb} GB`;
  } else {
    qualityStatus.textContent = `${r.label} · ${r.download_gb} GB download`;
  }
}

async function requestRefiner(name) {
  try {
    refiners[name] = await api.loadModel(name);
  } catch (err) {
    setStatus(`Could not start loading ${name}: ${err.message}`);
  }
  renderQuality();
}

function setQuality(next) {
  quality = next;
  renderQuality();
  updateButtons();
  if (quality !== 'fast') {
    const r = refiners[quality];
    if (!r || (!r.loaded && !r.loading)) requestRefiner(quality);
  }
}

async function pollBackend() {
  for (;;) {
    try {
      const h = await api.health();
      const seg = h.models.segmenter;
      const inp = h.models.inpainter;
      refiners = h.refiners || {};
      renderQuality();
      backendText.textContent = `Backend: ${h.device} · ${inp.name} + ${seg.name}`;
      if (h.error) {
        setStatus(`Model error: ${h.error}`);
        backendReady = false;
      } else if (h.ready) {
        const wasReady = backendReady;
        backendReady = true;
        if (!wasReady && !busy) setStatus('Ready', 1);
        updateButtons();
        await new Promise((r) => setTimeout(r, quality !== 'fast' && !qualityReady() ? 1500 : 5000));
        continue;
      } else {
        const loaded = [inp, seg].filter((m) => m.loaded).length;
        setStatus(`Loading models (${loaded}/2)… first start downloads about 500 MB`, 0.1 + loaded * 0.4);
      }
    } catch (err) {
      backendReady = false;
      backendText.textContent = 'Backend: not reachable';
      setStatus('Backend not reachable. Start it with ./start_mac.sh (or start_windows.bat).');
    }
    await new Promise((r) => setTimeout(r, 1500));
  }
}

function invalidateImageBlob() {
  imageBlobPromise = null;
}

function getImageBlob() {
  if (!imageBlobPromise) {
    imageBlobPromise = new Promise((resolve, reject) => {
      imageCanvas.toBlob((blob) => (blob ? resolve(blob) : reject(new Error('could not encode image'))), 'image/png');
    });
  }
  return imageBlobPromise;
}

function getMaskBlob() {
  return new Promise((resolve, reject) => {
    maskCanvas.toBlob((blob) => (blob ? resolve(blob) : reject(new Error('could not encode mask'))), 'image/png');
  });
}

// ---------- history ----------

function pushHistory() {
  if (!hasImage) return;
  const image = imageCtx.getImageData(0, 0, imageCanvas.width, imageCanvas.height);
  const mask = maskCtx.getImageData(0, 0, maskCanvas.width, maskCanvas.height);
  history.push({ image, mask, hadMask: hasMask });
  if (history.length > HISTORY_LIMIT) history.shift();
  updateButtons();
}

function restoreState(state) {
  imageCtx.putImageData(state.image, 0, 0);
  maskCtx.putImageData(state.mask, 0, 0);
  hasMask = state.hadMask;
  invalidateImageBlob();
  endSelection();
  updateButtons();
}

function clearMask(save = true) {
  if (!hasImage) return;
  if (save && hasMask) pushHistory();
  maskCtx.clearRect(0, 0, maskCanvas.width, maskCanvas.height);
  hasMask = false;
  endSelection();
  updateButtons();
}

// ---------- image loading ----------

function loadFile(file) {
  if (!file || !file.type.startsWith('image/')) return;
  const url = URL.createObjectURL(file);
  const img = new Image();
  img.onload = () => {
    let width = img.naturalWidth;
    let height = img.naturalHeight;
    if (Math.max(width, height) > MAX_SIDE) {
      const scale = MAX_SIDE / Math.max(width, height);
      width = Math.round(width * scale);
      height = Math.round(height * scale);
    }

    imageCanvas.width = maskCanvas.width = width;
    imageCanvas.height = maskCanvas.height = height;
    canvasWrap.style.width = `${width}px`;
    canvasWrap.style.height = `${height}px`;
    imageCtx.clearRect(0, 0, width, height);
    maskCtx.clearRect(0, 0, width, height);
    imageCtx.drawImage(img, 0, 0, width, height);

    hasImage = true;
    hasMask = false;
    history.length = 0;
    invalidateImageBlob();
    endSelection();
    stage.classList.remove('empty');
    emptyState.style.display = 'none';
    imageInfo.textContent = `${file.name} · ${width}×${height}`;
    setStatus(backendReady ? 'Ready' : 'Image loaded. Waiting for the backend…', backendReady ? 1 : null);
    URL.revokeObjectURL(url);
    fitToView();
    updateButtons();
  };
  img.onerror = () => {
    URL.revokeObjectURL(url);
    setStatus('Could not read this image file');
  };
  img.src = url;
}

// ---------- viewport (zoom / pan) ----------

function viewportSize() {
  return { w: viewport.clientWidth, h: viewport.clientHeight };
}

function applyTransform(next, { keepFit = false } = {}) {
  const { w, h } = viewportSize();
  transform = clampTransform(next, imageCanvas.width, imageCanvas.height, w, h);
  if (!keepFit) fitMode = false;
  canvasWrap.style.transform = `translate(${transform.tx}px, ${transform.ty}px) scale(${transform.scale})`;
  canvasWrap.style.setProperty('--inv', String(1 / transform.scale));
  canvasWrap.classList.toggle('pixelated', transform.scale >= 4);
  zoomValueBtn.textContent = `${Math.round(transform.scale * 100)}%`;
  zoomInBtn.disabled = transform.scale >= MAX_SCALE;
  zoomOutBtn.disabled = transform.scale <= MIN_SCALE;
  if (lastPointerClient) updateBrushPreview(lastPointerClient.x, lastPointerClient.y);
}

function fitToView() {
  if (!hasImage) return;
  const { w, h } = viewportSize();
  fitMode = true;
  applyTransform(fitTransform(imageCanvas.width, imageCanvas.height, w, h, 24), { keepFit: true });
}

function zoomAroundCenter(factor) {
  const { w, h } = viewportSize();
  applyTransform(zoomAt(transform, factor, w / 2, h / 2));
}

function zoomTo(scale) {
  const { w, h } = viewportSize();
  applyTransform(zoomAt(transform, scale / transform.scale, w / 2, h / 2));
}

viewport.addEventListener('wheel', (event) => {
  if (!hasImage) return;
  event.preventDefault();
  const rect = viewport.getBoundingClientRect();
  const factor = wheelZoomFactor(event.deltaY, event.ctrlKey || event.metaKey);
  applyTransform(zoomAt(transform, factor, event.clientX - rect.left, event.clientY - rect.top));
}, { passive: false });

window.addEventListener('resize', () => {
  if (!hasImage) return;
  if (fitMode) fitToView();
  else applyTransform(transform);
});

zoomInBtn.addEventListener('click', () => zoomAroundCenter(ZOOM_STEP));
zoomOutBtn.addEventListener('click', () => zoomAroundCenter(1 / ZOOM_STEP));
zoomFitBtn.addEventListener('click', fitToView);
zoomValueBtn.addEventListener('click', () => zoomTo(1));

// ---------- pointer mapping / brush ----------

function screenScale() {
  const rect = maskCanvas.getBoundingClientRect();
  return rect.width / maskCanvas.width;
}

function pointFromEvent(event) {
  return imagePointFromClient(event.clientX, event.clientY, maskCanvas.getBoundingClientRect(), maskCanvas.width, maskCanvas.height);
}

function updateBrushPreview(clientX, clientY) {
  if (!hasImage || mode !== 'brush' || spaceHeld || panState) {
    brushPreview.style.display = 'none';
    return;
  }
  const vp = viewport.getBoundingClientRect();
  const inside = clientX >= vp.left && clientX < vp.right && clientY >= vp.top && clientY < vp.bottom;
  if (!inside) {
    brushPreview.style.display = 'none';
    return;
  }
  // The stroke is drawn in image pixels; the preview must shrink or grow with the on-screen scale.
  const diameter = brushPreviewDiameter(Number(brushSize.value), screenScale());
  brushPreview.style.width = `${diameter}px`;
  brushPreview.style.height = `${diameter}px`;
  brushPreview.style.left = `${clientX}px`;
  brushPreview.style.top = `${clientY}px`;
  brushPreview.style.display = 'block';
}

function drawAt(point) {
  const size = Number(brushSize.value);
  maskCtx.save();
  maskCtx.globalCompositeOperation = 'source-over';
  maskCtx.lineCap = 'round';
  maskCtx.lineJoin = 'round';
  maskCtx.strokeStyle = `rgb(${MASK_COLOR.join(',')})`;
  maskCtx.fillStyle = maskCtx.strokeStyle;
  maskCtx.lineWidth = size;
  if (lastPoint) {
    maskCtx.beginPath();
    maskCtx.moveTo(lastPoint.x, lastPoint.y);
    maskCtx.lineTo(point.x, point.y);
    maskCtx.stroke();
  } else {
    maskCtx.beginPath();
    maskCtx.arc(point.x, point.y, size / 2, 0, Math.PI * 2);
    maskCtx.fill();
  }
  maskCtx.restore();
  lastPoint = point;
  hasMask = true;
  updateButtons();
}

// ---------- smart selection ----------

function endSelection() {
  selection = null;
  segmentRequest += 1;
  pointLayer.replaceChildren();
  stage.classList.remove('selecting');
}

function renderPoints() {
  pointLayer.replaceChildren();
  if (!selection) return;
  for (const p of selection.points) {
    const el = document.createElement('div');
    el.className = `point-marker${p.label === 0 ? ' negative' : ''}`;
    el.style.left = `${p.x}px`;
    el.style.top = `${p.y}px`;
    pointLayer.appendChild(el);
  }
}

function blobToImage(blob) {
  return new Promise((resolve, reject) => {
    const url = URL.createObjectURL(blob);
    const img = new Image();
    img.onload = () => { URL.revokeObjectURL(url); resolve(img); };
    img.onerror = () => { URL.revokeObjectURL(url); reject(new Error('could not decode response image')); };
    img.src = url;
  });
}

/** Paint a grayscale mask image (white = selected) into the mask canvas in the mask color. */
function paintMaskImage(img) {
  const w = maskCanvas.width;
  const h = maskCanvas.height;
  const tmp = document.createElement('canvas');
  tmp.width = w;
  tmp.height = h;
  const tctx = tmp.getContext('2d', { willReadFrequently: true });
  tctx.drawImage(img, 0, 0, w, h);
  const src = tctx.getImageData(0, 0, w, h);
  const data = src.data;
  let count = 0;
  for (let i = 0; i < data.length; i += 4) {
    if (data[i] > 127) {
      data[i] = MASK_COLOR[0];
      data[i + 1] = MASK_COLOR[1];
      data[i + 2] = MASK_COLOR[2];
      data[i + 3] = 255;
      count += 1;
    } else {
      data[i + 3] = 0;
    }
  }
  tctx.putImageData(src, 0, 0);
  maskCtx.drawImage(tmp, 0, 0);
  return count;
}

async function runSegmentation() {
  if (!selection) return;
  const request = ++segmentRequest;
  const points = selection.points.map((p) => ({ x: Math.round(p.x), y: Math.round(p.y), label: p.label }));
  stage.classList.add('selecting');
  setStatus('Selecting object…');
  try {
    const blob = await getImageBlob();
    const maskBlob = await api.segment(blob, points);
    if (request !== segmentRequest || !selection) return;
    const img = await blobToImage(maskBlob);
    if (request !== segmentRequest || !selection) return;
    maskCtx.putImageData(selection.base, 0, 0);
    const count = paintMaskImage(img);
    hasMask = count > 0 || selection.baseHadMask;
    setStatus(count > 0 ? 'Object selected. Click Remove, or refine with more clicks.' : 'Nothing selected there. Try another point.', 1);
  } catch (err) {
    if (request !== segmentRequest) return;
    setStatus(`Selection failed: ${err.message}`);
  } finally {
    if (request === segmentRequest) stage.classList.remove('selecting');
    updateButtons();
  }
}

function smartClick(point, negative) {
  if (negative && selection) {
    selection.points.push({ x: point.x, y: point.y, label: 0 });
  } else if (!negative && selection && selection.points.length > 0 && selection.extend) {
    selection.points.push({ x: point.x, y: point.y, label: 1 });
  } else {
    // New object: remember the mask as it is now so refinements replace only this object's area.
    pushHistory();
    selection = {
      points: [{ x: point.x, y: point.y, label: 1 }],
      base: maskCtx.getImageData(0, 0, maskCanvas.width, maskCanvas.height),
      baseHadMask: hasMask,
      extend: false,
    };
  }
  renderPoints();
  runSegmentation();
}

// ---------- modes ----------

function setMode(next) {
  mode = next;
  stage.classList.toggle('mode-smart', mode === 'smart');
  stage.classList.toggle('mode-brush', mode === 'brush');
  modeSmartBtn.classList.toggle('active', mode === 'smart');
  modeBrushBtn.classList.toggle('active', mode === 'brush');
  hintText.textContent = HINTS[mode];
  if (mode === 'brush') endSelection();
  if (lastPointerClient) updateBrushPreview(lastPointerClient.x, lastPointerClient.y);
}

modeSmartBtn.addEventListener('click', () => setMode('smart'));
modeBrushBtn.addEventListener('click', () => setMode('brush'));

// ---------- pointer events ----------

viewport.addEventListener('pointerdown', (event) => {
  if (!hasImage || busy) return;
  const wantsPan = spaceHeld || event.button === 1;
  if (wantsPan) {
    event.preventDefault();
    panState = { pointerId: event.pointerId, startX: event.clientX, startY: event.clientY, tx: transform.tx, ty: transform.ty };
    try { viewport.setPointerCapture(event.pointerId); } catch (_) { /* synthetic event */ }
    stage.classList.add('panning');
    brushPreview.style.display = 'none';
    return;
  }
  if (event.button !== 0) return;
  const point = pointFromEvent(event);
  if (!point.inside) return;

  if (mode === 'brush') {
    pushHistory();
    endSelection();
    isDrawing = true;
    lastPoint = null;
    try { viewport.setPointerCapture(event.pointerId); } catch (_) { /* synthetic event */ }
    drawAt(point);
  } else if (backendReady) {
    smartClick(point, event.shiftKey);
  } else {
    setStatus('Smart select needs the backend models. Use the brush meanwhile.');
  }
});

viewport.addEventListener('pointermove', (event) => {
  lastPointerClient = { x: event.clientX, y: event.clientY };
  if (panState && event.pointerId === panState.pointerId) {
    const dx = event.clientX - panState.startX;
    const dy = event.clientY - panState.startY;
    applyTransform({ scale: transform.scale, tx: panState.tx + dx, ty: panState.ty + dy });
    return;
  }
  updateBrushPreview(event.clientX, event.clientY);
  if (!isDrawing) return;
  const point = pointFromEvent(event);
  drawAt(point);
});

function endPointer(event) {
  if (panState && event.pointerId === panState.pointerId) {
    panState = null;
    stage.classList.remove('panning');
    try { viewport.releasePointerCapture(event.pointerId); } catch (_) { /* already released */ }
    updateBrushPreview(event.clientX, event.clientY);
  }
  if (isDrawing) {
    isDrawing = false;
    lastPoint = null;
    try { viewport.releasePointerCapture(event.pointerId); } catch (_) { /* already released */ }
  }
}

viewport.addEventListener('pointerup', endPointer);
viewport.addEventListener('pointercancel', endPointer);
viewport.addEventListener('pointerleave', () => {
  lastPointerClient = null;
  brushPreview.style.display = 'none';
});
viewport.addEventListener('contextmenu', (event) => event.preventDefault());
viewport.addEventListener('auxclick', (event) => { if (event.button === 1) event.preventDefault(); });

// ---------- file input / drag & drop ----------

fileInput.addEventListener('change', (event) => loadFile(event.target.files?.[0]));

for (const target of [dropZone, stage]) {
  for (const eventName of ['dragenter', 'dragover']) {
    target.addEventListener(eventName, (event) => {
      event.preventDefault();
      dropZone.classList.add('dragover');
    });
  }
  for (const eventName of ['dragleave', 'drop']) {
    target.addEventListener(eventName, (event) => {
      event.preventDefault();
      dropZone.classList.remove('dragover');
    });
  }
  target.addEventListener('drop', (event) => loadFile(event.dataTransfer?.files?.[0]));
}

brushSize.addEventListener('input', () => {
  brushValue.textContent = brushSize.value;
  if (lastPointerClient) updateBrushPreview(lastPointerClient.x, lastPointerClient.y);
});

// ---------- actions ----------

clearBtn.addEventListener('click', () => clearMask(true));

undoBtn.addEventListener('click', () => {
  if (busy) return;
  const state = history.pop();
  if (state) restoreState(state);
});

removeBtn.addEventListener('click', async () => {
  if (!backendReady || !hasImage || !hasMask || busy || !qualityReady()) return;
  pushHistory();
  endSelection();
  const hints = {
    fast: 'Big-LaMa is rebuilding the area',
    sdxl: 'LaMa first, then SDXL refines the fill (about a minute)',
    flux: 'LaMa first, then FLUX.1 Fill repaints the area (several minutes)',
  };
  setBusy(true, 'Removing…', hints[quality]);
  setStatus('Removing object…', 1);
  try {
    const [imageBlob, maskBlob] = await Promise.all([getImageBlob(), getMaskBlob()]);
    const resultBlob = await api.inpaint(imageBlob, maskBlob, { quality, strength: Number(strengthInput.value) });
    const img = await blobToImage(resultBlob);
    imageCtx.drawImage(img, 0, 0, imageCanvas.width, imageCanvas.height);
    invalidateImageBlob();
    clearMask(false);
    setStatus('Done', 1);
  } catch (err) {
    setStatus(`Removal failed: ${err.message}`);
  } finally {
    setBusy(false);
  }
});

downloadBtn.addEventListener('click', () => {
  if (!hasImage) return;
  const a = document.createElement('a');
  a.href = imageCanvas.toDataURL('image/png');
  a.download = 'obj-remover-result.png';
  document.body.appendChild(a);
  a.click();
  a.remove();
});

// ---------- keyboard ----------

function isTypingTarget(el) {
  return el && (el.tagName === 'INPUT' || el.tagName === 'TEXTAREA' || el.isContentEditable);
}

window.addEventListener('keydown', (event) => {
  if (isTypingTarget(event.target) && event.target !== brushSize) return;
  const key = event.key.toLowerCase();
  if ((event.ctrlKey || event.metaKey) && key === 'z') {
    event.preventDefault();
    undoBtn.click();
    return;
  }
  if (event.ctrlKey || event.metaKey) return;
  if (event.code === 'Space') {
    if (!spaceHeld) {
      spaceHeld = true;
      stage.classList.add('pan-ready');
      brushPreview.style.display = 'none';
    }
    event.preventDefault();
    return;
  }
  if (key === 'b') setMode('brush');
  else if (key === 's') setMode('smart');
  else if (key === '+' || key === '=') zoomAroundCenter(ZOOM_STEP);
  else if (key === '-' || key === '_') zoomAroundCenter(1 / ZOOM_STEP);
  else if (key === '0') fitToView();
  else if (key === '1') zoomTo(1);
  else if (key === 'escape') endSelection();
  else if (key === '[') { brushSize.value = Math.max(Number(brushSize.min), Number(brushSize.value) - 4); brushSize.dispatchEvent(new Event('input')); }
  else if (key === ']') { brushSize.value = Math.min(Number(brushSize.max), Number(brushSize.value) + 4); brushSize.dispatchEvent(new Event('input')); }
});

window.addEventListener('keyup', (event) => {
  if (event.code === 'Space') {
    spaceHeld = false;
    stage.classList.remove('pan-ready');
    if (lastPointerClient) updateBrushPreview(lastPointerClient.x, lastPointerClient.y);
  }
});

window.addEventListener('blur', () => {
  spaceHeld = false;
  stage.classList.remove('pan-ready');
});

for (const btn of qualityButtons) btn.addEventListener('click', () => setQuality(btn.dataset.quality));
strengthInput.addEventListener('input', () => { strengthValue.textContent = Number(strengthInput.value).toFixed(2); });

// ---------- boot ----------
renderQuality();

setMode('smart');
updateButtons();
pollBackend();
