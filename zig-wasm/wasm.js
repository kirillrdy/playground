let wasmExports = null;
let wasmMemory = null;
let activeCanvas = null;
let activeCtx = null;

function getView(ptr, len) {
  return new Uint8Array(wasmMemory.buffer, ptr, len);
}

function readStrFromWasm(ptr, len) {
  return new TextDecoder().decode(getView(ptr, len));
}

function writeBytesToWasm(bytes) {
  const ptr = wasmExports.alloc(bytes.length);
  if (!ptr) throw new Error("wasm alloc failed");
  getView(ptr, bytes.length).set(bytes);
  return ptr;
}

function jsCanvasClear() {
  if (!activeCtx || !activeCanvas) return;
  activeCtx.clearRect(0, 0, activeCanvas.width, activeCanvas.height);
}

function jsCanvasDraw(x, y, w, h, labelPtr, labelLen, score) {
  if (!activeCtx) return;
  const label = readStrFromWasm(labelPtr, labelLen);
  activeCtx.lineWidth = 2;
  activeCtx.strokeStyle = "#ef4444";
  activeCtx.fillStyle = "#ef4444";
  activeCtx.font = "12px sans-serif";
  activeCtx.strokeRect(x, y, w, h);
  activeCtx.fillText(`${label} ${Number(score).toFixed(2)}`, x + 2, Math.max(12, y - 4));
}

async function ensureWasm() {
  if (wasmExports) return wasmExports;

  const imports = {
    env: {
      jsCanvasClear,
      jsCanvasDraw,
    },
  };

  const instance = await WebAssembly.instantiateStreaming(fetch("/main.wasm"), imports);
  wasmExports = instance.instance.exports;
  wasmMemory = wasmExports.memory;
  return wasmExports;
}

function fitCanvasToVideo(video, canvas) {
  const w = video.clientWidth || video.videoWidth || 0;
  const h = video.clientHeight || video.videoHeight || 0;
  if (!w || !h) return;
  canvas.width = w;
  canvas.height = h;
  canvas.style.width = `${w}px`;
  canvas.style.height = `${h}px`;
}

async function attachVideoOverlay(video, canvas, detectionsUrl) {
  const wasm = await ensureWasm();
  activeCanvas = canvas;
  activeCtx = canvas.getContext("2d");

  const raw = await fetch(detectionsUrl).then((r) => r.text());
  const bytes = new TextEncoder().encode(raw);

  wasm.clearDetections();
  const ptr = writeBytesToWasm(bytes);
  const ok = wasm.loadDetections(ptr, bytes.length);
  wasm.free(ptr, bytes.length);
  if (!ok) {
    console.warn("failed to parse detections in wasm");
  }

  const draw = () => {
    fitCanvasToVideo(video, canvas);
    wasm.renderAt(video.currentTime * 1000, video.duration * 1000, canvas.width, canvas.height);
  };

  video.addEventListener("loadedmetadata", draw);
  video.addEventListener("timeupdate", draw);
  video.addEventListener("seeked", draw);
  video.addEventListener("play", draw);
  video.addEventListener("pause", draw);
  window.addEventListener("resize", draw);

  const tick = () => {
    draw();
    requestAnimationFrame(tick);
  };
  requestAnimationFrame(tick);
}

window.videoOverlayWasm = {
  attachVideoOverlay,
};
