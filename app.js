(() => {
  const fileInput = document.getElementById('fileInput');
  const dropZone = document.getElementById('dropZone');
  const stage = document.getElementById('stage');
  const emptyState = document.getElementById('emptyState');
  const imageCanvas = document.getElementById('imageCanvas');
  const maskCanvas = document.getElementById('maskCanvas');
  const imageCtx = imageCanvas.getContext('2d', { willReadFrequently: true });
  const maskCtx = maskCanvas.getContext('2d', { willReadFrequently: true });
  const brushSize = document.getElementById('brushSize');
  const brushValue = document.getElementById('brushValue');
  const brushPreview = document.getElementById('brushPreview');
  const removeBtn = document.getElementById('removeBtn');
  const undoBtn = document.getElementById('undoBtn');
  const clearBtn = document.getElementById('clearBtn');
  const downloadBtn = document.getElementById('downloadBtn');
  const statusText = document.getElementById('statusText');
  const backendText = document.getElementById('backendText');
  const progressBar = document.getElementById('progressBar');
  const imageInfo = document.getElementById('imageInfo');
  const maskInfo = document.getElementById('maskInfo');
  const processingOverlay = document.getElementById('processingOverlay');

  let worker = null;
  let workerReady = false;
  let hasImage = false;
  let isDrawing = false;
  let hasMask = false;
  let lastPoint = null;
  const history = [];

  function setStatus(message, progress = null) {
    statusText.textContent = message;
    if (typeof progress === 'number') {
      progressBar.style.width = `${Math.max(0, Math.min(100, progress * 100))}%`;
    }
  }

  function setBusy(busy) {
    document.body.classList.toggle('busy', busy);
    removeBtn.classList.toggle('removing', busy);
    removeBtn.textContent = busy ? 'Removing…' : 'Remove Object';
    if (processingOverlay) processingOverlay.hidden = !busy;
    maskCanvas.classList.toggle('pulsing', busy && hasMask);
    removeBtn.disabled = busy || !workerReady || !hasImage || !hasMask;
    clearBtn.disabled = busy || !hasImage || !hasMask;
    undoBtn.disabled = busy || history.length === 0;
    downloadBtn.disabled = busy || !hasImage;
  }

  function updateButtons() {
    removeBtn.disabled = !workerReady || !hasImage || !hasMask;
    clearBtn.disabled = !hasImage || !hasMask;
    undoBtn.disabled = history.length === 0;
    downloadBtn.disabled = !hasImage;
    maskInfo.textContent = hasMask ? 'Mask: painted · 40% opacity' : 'Mask: empty';
  }

  function initWorker() {
    try {
      worker = new Worker('./worker.js?v=12');
    } catch (err) {
      setStatus(`Worker error: ${err.message}`);
      return;
    }

    worker.onmessage = (event) => {
      const data = event.data || {};
      if (data.type === 'modelProgress') {
        if (data.cached) setStatus('Model loaded from local cache', 1);
        else if (data.total) setStatus(`Loading model ${(data.received / 1024 / 1024).toFixed(1)} / ${(data.total / 1024 / 1024).toFixed(1)} MB`, data.ratio || 0);
        else setStatus('Loading local model…', data.ratio || 0);
      }

      if (data.type === 'ready') {
        workerReady = true;
        backendText.textContent = `Backend: ${data.backend || 'wasm'}${data.model ? ` / ${data.model}` : ''}`;
        setStatus('Ready', 1);
        updateButtons();
      }

      if (data.type === 'cropInfo') {
        setStatus(`Processing mask area ${data.maskW}×${data.maskH}…`, 1);
      }

      if (data.type === 'result') {
        const pixels = new Uint8ClampedArray(data.pixels);
        const out = new ImageData(pixels, data.origW, data.origH);
        imageCtx.putImageData(out, 0, 0);
        clearMask(false);
        setStatus('Done', 1);
        setBusy(false);
        updateButtons();
      }

      if (data.type === 'error') {
        setStatus(data.message || 'Unknown worker error');
        setBusy(false);
        updateButtons();
      }
    };

    worker.onerror = (event) => {
      setStatus(`Worker failed: ${event.message || 'unknown error'}`);
      setBusy(false);
    };

    setStatus('Loading model…', 0.05);
    worker.postMessage({ type: 'init', isMobile: false });
  }

  function pushHistory() {
    if (!hasImage) return;
    const image = imageCtx.getImageData(0, 0, imageCanvas.width, imageCanvas.height);
    const mask = maskCtx.getImageData(0, 0, maskCanvas.width, maskCanvas.height);
    history.push({ image, mask, hadMask: hasMask });
    if (history.length > 20) history.shift();
    updateButtons();
  }

  function restoreState(state) {
    imageCtx.putImageData(state.image, 0, 0);
    maskCtx.putImageData(state.mask, 0, 0);
    hasMask = state.hadMask;
    updateButtons();
  }

  function clearMask(save = true) {
    if (!hasImage) return;
    if (save && hasMask) pushHistory();
    maskCtx.clearRect(0, 0, maskCanvas.width, maskCanvas.height);
    hasMask = false;
    updateButtons();
  }

  function loadFile(file) {
    if (!file || !file.type.startsWith('image/')) return;
    const url = URL.createObjectURL(file);
    const img = new Image();
    img.onload = () => {
      const maxSide = 2048;
      let width = img.naturalWidth;
      let height = img.naturalHeight;
      if (Math.max(width, height) > maxSide) {
        const scale = maxSide / Math.max(width, height);
        width = Math.round(width * scale);
        height = Math.round(height * scale);
      }

      imageCanvas.width = maskCanvas.width = width;
      imageCanvas.height = maskCanvas.height = height;
      imageCtx.clearRect(0, 0, width, height);
      maskCtx.clearRect(0, 0, width, height);
      imageCtx.drawImage(img, 0, 0, width, height);

      hasImage = true;
      hasMask = false;
      history.length = 0;
      stage.classList.remove('empty');
      emptyState.style.display = 'none';
      imageInfo.textContent = `${file.name} · ${width}×${height}`;
      setStatus(workerReady ? 'Ready' : 'Image loaded. Waiting for model…', workerReady ? 1 : null);
      URL.revokeObjectURL(url);
      updateButtons();
    };
    img.onerror = () => {
      URL.revokeObjectURL(url);
      setStatus('Could not read this image file');
    };
    img.src = url;
  }

  function pointFromEvent(event) {
    const rect = maskCanvas.getBoundingClientRect();
    return {
      x: (event.clientX - rect.left) * (maskCanvas.width / rect.width),
      y: (event.clientY - rect.top) * (maskCanvas.height / rect.height),
      inside: event.clientX >= rect.left && event.clientX <= rect.right && event.clientY >= rect.top && event.clientY <= rect.bottom,
    };
  }

  function drawAt(point) {
    const size = Number(brushSize.value);
    maskCtx.save();
    maskCtx.globalCompositeOperation = 'source-over';
    maskCtx.lineCap = 'round';
    maskCtx.lineJoin = 'round';
    maskCtx.strokeStyle = 'rgba(138, 63, 252, 1)';
    maskCtx.fillStyle = 'rgba(138, 63, 252, 1)';
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

  function moveBrushPreview(event) {
    const point = pointFromEvent(event);
    if (!hasImage || !point.inside) {
      brushPreview.style.display = 'none';
      return;
    }
    const size = Number(brushSize.value);
    brushPreview.style.width = `${size}px`;
    brushPreview.style.height = `${size}px`;
    brushPreview.style.background = 'rgba(138, 63, 252, 0.40)';
    brushPreview.style.borderColor = 'rgba(138, 63, 252, 0.95)';
    brushPreview.style.left = `${event.clientX}px`;
    brushPreview.style.top = `${event.clientY}px`;
    brushPreview.style.display = 'block';
  }

  fileInput.addEventListener('change', (event) => loadFile(event.target.files?.[0]));

  for (const eventName of ['dragenter', 'dragover']) {
    dropZone.addEventListener(eventName, (event) => {
      event.preventDefault();
      dropZone.classList.add('dragover');
    });
  }
  for (const eventName of ['dragleave', 'drop']) {
    dropZone.addEventListener(eventName, (event) => {
      event.preventDefault();
      dropZone.classList.remove('dragover');
    });
  }
  dropZone.addEventListener('drop', (event) => loadFile(event.dataTransfer?.files?.[0]));

  brushSize.addEventListener('input', () => {
    brushValue.textContent = brushSize.value;
  });

  maskCanvas.addEventListener('pointerdown', (event) => {
    if (!hasImage) return;
    const point = pointFromEvent(event);
    if (!point.inside) return;
    pushHistory();
    isDrawing = true;
    lastPoint = null;
    maskCanvas.setPointerCapture(event.pointerId);
    drawAt(point);
  });

  maskCanvas.addEventListener('pointermove', (event) => {
    moveBrushPreview(event);
    if (!isDrawing) return;
    const point = pointFromEvent(event);
    if (point.inside) drawAt(point);
  });

  function stopDrawing(event) {
    if (!isDrawing) return;
    isDrawing = false;
    lastPoint = null;
    try { maskCanvas.releasePointerCapture(event.pointerId); } catch (_) {}
  }

  maskCanvas.addEventListener('pointerup', stopDrawing);
  maskCanvas.addEventListener('pointercancel', stopDrawing);
  maskCanvas.addEventListener('pointerleave', () => {
    brushPreview.style.display = 'none';
  });

  clearBtn.addEventListener('click', () => clearMask(true));

  undoBtn.addEventListener('click', () => {
    const state = history.pop();
    if (state) restoreState(state);
  });

  removeBtn.addEventListener('click', () => {
    if (!workerReady || !hasImage || !hasMask) return;
    pushHistory();
    setBusy(true);
    setStatus('Removing object…', 1);
    const image = imageCtx.getImageData(0, 0, imageCanvas.width, imageCanvas.height);
    const mask = maskCtx.getImageData(0, 0, maskCanvas.width, maskCanvas.height);
    worker.postMessage({
      type: 'inpaint',
      imagePixels: image.data.buffer,
      maskPixels: mask.data.buffer,
      origW: imageCanvas.width,
      origH: imageCanvas.height,
    }, [image.data.buffer, mask.data.buffer]);
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

  window.addEventListener('keydown', (event) => {
    if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === 'z') {
      event.preventDefault();
      undoBtn.click();
    }
  });

  initWorker();
  updateButtons();
})();
