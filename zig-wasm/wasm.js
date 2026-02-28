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

  wasm.clearDetections();

  const loadedSegments = new Set();
  const fetchingSegments = new Set();
  const SEGMENT_DURATION = 2.0;

  async function fetchSegment(startTime) {
    const segmentId = Math.floor(startTime / SEGMENT_DURATION);
    if (loadedSegments.has(segmentId) || fetchingSegments.has(segmentId)) return;

    fetchingSegments.add(segmentId);
    try {
      const start = segmentId * SEGMENT_DURATION;
      const end = start + SEGMENT_DURATION;
      const url = `${detectionsUrl}?start=${start.toFixed(2)}&end=${end.toFixed(2)}`;
      const response = await fetch(url);
      if (!response.ok) throw new Error(`failed to fetch segment ${segmentId}`);
      
      const raw = await response.text();
      const bytes = new TextEncoder().encode(raw);
      const ptr = writeBytesToWasm(bytes);
      wasm.loadDetections(ptr, bytes.length);
      wasm.free(ptr, bytes.length);
      
      loadedSegments.add(segmentId);
    } catch (err) {
      console.warn("failed to fetch detections segment:", err);
    } finally {
      fetchingSegments.delete(segmentId);
    }
  }

  const draw = () => {
    fitCanvasToVideo(video, canvas);
    const currentTime = video.currentTime;
    
    // Fetch current and next segment
    fetchSegment(currentTime);
    fetchSegment(currentTime + SEGMENT_DURATION);

    wasm.renderAt(currentTime * 1000, video.duration * 1000, canvas.width, canvas.height);
  };

  video.addEventListener("loadedmetadata", draw);
  video.addEventListener("timeupdate", draw);
  video.addEventListener("seeked", () => {
    // On seek, we might want to clear or just let it fetch new segments
    draw();
  });
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
