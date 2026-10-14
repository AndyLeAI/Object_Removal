import test from 'node:test';
import assert from 'node:assert/strict';
import {
  fitTransform, zoomAt, clampTransform, pan, wheelZoomFactor,
  imagePointFromClient, brushPreviewDiameter, MIN_SCALE, MAX_SCALE,
} from '../viewport.js';

const close = (a, b, eps = 1e-9) => assert.ok(Math.abs(a - b) < eps, `${a} != ${b}`);

test('fitTransform centers a landscape image inside the viewport', () => {
  const t = fitTransform(2000, 1000, 1000, 800, 0);
  close(t.scale, 0.5);
  close(t.tx, 0);
  close(t.ty, 150);
});

test('fitTransform respects padding and scales up small images', () => {
  const t = fitTransform(100, 100, 1000, 1000, 100);
  close(t.scale, 8);
  close(t.tx, 100);
  close(t.ty, 100);
});

test('zoomAt keeps the image point under the cursor fixed', () => {
  const t = { scale: 1, tx: 10, ty: 20 };
  const px = 300, py = 200;
  const imgX = (px - t.tx) / t.scale;
  const imgY = (py - t.ty) / t.scale;
  const z = zoomAt(t, 2, px, py);
  close(z.scale, 2);
  close(z.tx + imgX * z.scale, px);
  close(z.ty + imgY * z.scale, py);
});

test('zoomAt clamps to min/max scale', () => {
  assert.equal(zoomAt({ scale: 1, tx: 0, ty: 0 }, 1e9, 0, 0).scale, MAX_SCALE);
  assert.equal(zoomAt({ scale: 1, tx: 0, ty: 0 }, 1e-9, 0, 0).scale, MIN_SCALE);
});

test('clampTransform centers images smaller than the viewport', () => {
  const c = clampTransform({ scale: 0.5, tx: -999, ty: 999 }, 400, 400, 1000, 600);
  close(c.tx, 400);
  close(c.ty, 200);
});

test('clampTransform stops large images from leaving a gap', () => {
  const c = clampTransform({ scale: 2, tx: 50, ty: -5000 }, 1000, 1000, 800, 600);
  close(c.tx, 0);
  close(c.ty, 600 - 2000);
  const mid = clampTransform({ scale: 2, tx: -300, ty: -300 }, 1000, 1000, 800, 600);
  close(mid.tx, -300);
  close(mid.ty, -300);
});

test('pan shifts translation only', () => {
  assert.deepEqual(pan({ scale: 3, tx: 1, ty: 2 }, 5, -7), { scale: 3, tx: 6, ty: -5 });
});

test('wheelZoomFactor zooms in on negative delta and is symmetric', () => {
  assert.ok(wheelZoomFactor(-100, false) > 1);
  assert.ok(wheelZoomFactor(100, false) < 1);
  close(wheelZoomFactor(-100, false) * wheelZoomFactor(100, false), 1);
  assert.ok(wheelZoomFactor(-10, true) > wheelZoomFactor(-10, false));
});

test('imagePointFromClient maps through the scaled rect', () => {
  const rect = { left: 100, top: 50, width: 500, height: 250, right: 600, bottom: 300 };
  const p = imagePointFromClient(350, 175, rect, 1000, 500);
  close(p.x, 500);
  close(p.y, 250);
  assert.equal(p.inside, true);
  assert.equal(imagePointFromClient(50, 50, rect, 1000, 500).inside, false);
});

test('brush preview diameter follows the on-screen scale', () => {
  // A 40px brush on an image shown at half size must look 20px wide.
  close(brushPreviewDiameter(40, 0.5), 20);
  close(brushPreviewDiameter(40, 3), 120);
});
