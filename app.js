import { loadLiteRt, loadAndCompile, Tensor } from "https://cdn.jsdelivr.net/npm/@litertjs/core@2.6.0/+esm";

const MODEL_URL = new URL("./models/MobileNetV2_Deteccion_Frutas_Fijo.tflite", import.meta.url).href;
const IMAGE_SIZE = 224;
const CLASSES = ["AVOCADO", "MANGO", "PAPAYA", "TOMATE"];
const FRAME_INTERVAL_MS = 100;

const video = document.querySelector("#camera");
const overlay = document.querySelector("#overlay");
const ctx = overlay.getContext("2d");
const inputCanvas = document.createElement("canvas");
inputCanvas.width = IMAGE_SIZE;
inputCanvas.height = IMAGE_SIZE;
const inputCtx = inputCanvas.getContext("2d", { willReadFrequently: true });

const modelState = document.querySelector("#modelState");
const statusText = document.querySelector("#statusText");
const startBtn = document.querySelector("#startBtn");
const stopBtn = document.querySelector("#stopBtn");
const placeholder = document.querySelector("#cameraPlaceholder");
const className = document.querySelector("#className");
const confidenceValue = document.querySelector("#confidenceValue");
const confidenceBar = document.querySelector("#confidenceBar");
const inferenceTime = document.querySelector("#inferenceTime");
const engineName = document.querySelector("#engineName");
const fpsLabel = document.querySelector("#fpsLabel");

let model = null;
let stream = null;
let running = false;
let busy = false;
let rafId = 0;
let lastFrameAt = 0;
let lastFpsAt = performance.now();
let fpsFrames = 0;

function setModelState(text, kind = "loading") {
  modelState.textContent = text;
  modelState.className = "state state-" + kind;
}

async function initializeModel() {
  try {
    setModelState("Cargando modelo", "loading");
    statusText.textContent = "Descargando MobileNetV2 al dispositivo...";

    await loadLiteRt("https://cdn.jsdelivr.net/npm/@litertjs/core@2.6.0/wasm/");

    try {
      model = await loadAndCompile(MODEL_URL, { accelerator: "webgpu" });
      engineName.textContent = "WebGPU";
    } catch (gpuError) {
      console.warn("WebGPU no disponible; se utilizará WASM:", gpuError);
      model = await loadAndCompile(MODEL_URL, { accelerator: "wasm" });
      engineName.textContent = "WASM";
    }

    const inputs = model.getInputDetails?.() ?? [];
    const outputs = model.getOutputDetails?.() ?? [];
    console.info("Entradas LiteRT:", inputs);
    console.info("Salidas LiteRT:", outputs);

    setModelState("Modelo listo", "ready");
    statusText.textContent = "Modelo cargado en el dispositivo. Puedes iniciar la cámara.";
    startBtn.disabled = false;
  } catch (error) {
    console.error(error);
    setModelState("Error del modelo", "error");
    statusText.textContent = "No se pudo cargar el modelo. Comprueba que exista el archivo .tflite en /models/.";
    startBtn.disabled = true;
  }
}

function makeInputTensor() {
  // El entrenamiento redimensionó las imágenes a 224x224 y aplicó
  // mobilenet_v2.preprocess_input: RGB de [0,255] a [-1,1].
  inputCtx.drawImage(video, 0, 0, IMAGE_SIZE, IMAGE_SIZE);
  const { data } = inputCtx.getImageData(0, 0, IMAGE_SIZE, IMAGE_SIZE);
  const rgb = new Float32Array(IMAGE_SIZE * IMAGE_SIZE * 3);
  let p = 0;
  for (let i = 0; i < data.length; i += 4) {
    rgb[p++] = data[i] / 127.5 - 1;
    rgb[p++] = data[i + 1] / 127.5 - 1;
    rgb[p++] = data[i + 2] / 127.5 - 1;
  }
  return new Tensor(rgb, [1, IMAGE_SIZE, IMAGE_SIZE, 3]);
}

async function runInference() {
  if (!running || busy || video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA) return;
  busy = true;
  const startedAt = performance.now();
  let inputTensor = null;
  let outputs = [];

  try {
    inputTensor = makeInputTensor();

    // IMPORTANTE: model.run es asíncrono. Hay que esperarlo antes de leer outputs.
    outputs = await model.run([inputTensor]);

    // Orden confirmado en el modelo LiteRT fijo: [clase, bounding_box].
    if (!outputs || outputs.length < 2) {
      throw new Error("El modelo no devolvió las dos salidas esperadas.");
    }

    const probabilities = Array.from(await outputs[0].data());
    const box = Array.from(await outputs[1].data());

    if (probabilities.length !== 4 || box.length !== 4) {
      throw new Error("Dimensiones de salida inesperadas: " + probabilities.length + " / " + box.length);
    }
    if (!probabilities.every(Number.isFinite) || !box.every(Number.isFinite)) {
      throw new Error("El modelo devolvió valores no válidos.");
    }

    let bestIndex = 0;
    for (let i = 1; i < probabilities.length; i++) {
      if (probabilities[i] > probabilities[bestIndex]) bestIndex = i;
    }
    const confidence = Math.max(0, Math.min(1, probabilities[bestIndex]));

    className.textContent = CLASSES[bestIndex];
    confidenceValue.textContent = (confidence * 100).toFixed(1) + "%";
    confidenceBar.style.width = (confidence * 100).toFixed(1) + "%";
    inferenceTime.textContent = (performance.now() - startedAt).toFixed(0) + " ms";

    drawBox(box);

    // Ayuda a detectar errores de preprocesamiento/salida sin detener la cámara.
    console.debug("Predicción:", CLASSES[bestIndex], probabilities, "Caja:", box);
  } catch (error) {
    console.error("Error durante inferencia:", error);
    statusText.textContent = "Error de inferencia: " + (error?.message ?? error);
  } finally {
    inputTensor?.delete?.();
    for (const output of outputs) output?.delete?.();
    busy = false;
  }
}

function drawBox(values) {
  ctx.clearRect(0, 0, overlay.width, overlay.height);
  if (!values || values.length !== 4) return;

  // El notebook de entrenamiento define las cajas como [x_min,y_min,x_max,y_max],
  // normalizadas entre 0 y 1. Se ordenan esquinas por seguridad.
  const x1 = Math.max(0, Math.min(1, Math.min(values[0], values[2])));
  const y1 = Math.max(0, Math.min(1, Math.min(values[1], values[3])));
  const x2 = Math.max(0, Math.min(1, Math.max(values[0], values[2])));
  const y2 = Math.max(0, Math.min(1, Math.max(values[1], values[3])));

  const left = x1 * overlay.width;
  const top = y1 * overlay.height;
  const width = (x2 - x1) * overlay.width;
  const height = (y2 - y1) * overlay.height;
  if (width < 2 || height < 2) return;

  ctx.strokeStyle = "#70f59a";
  ctx.lineWidth = Math.max(3, overlay.width * 0.005);
  ctx.strokeRect(left, top, width, height);
  ctx.fillStyle = "#70f59a";
  ctx.font = "bold 18px Arial";
  ctx.shadowColor = "#001108";
  ctx.shadowBlur = 5;
  ctx.fillText(className.textContent, left + 4, Math.max(22, top - 8));
  ctx.shadowBlur = 0;
}

function frameLoop(now) {
  if (!running) return;
  if (!busy && now - lastFrameAt >= FRAME_INTERVAL_MS) {
    lastFrameAt = now;
    void runInference();
    fpsFrames++;
  }
  if (now - lastFpsAt >= 1000) {
    fpsLabel.textContent = fpsFrames + " FPS";
    fpsFrames = 0;
    lastFpsAt = now;
  }
  rafId = requestAnimationFrame(frameLoop);
}

async function startCamera() {
  try {
    if (!model) throw new Error("El modelo todavía no está listo.");
    stream = await navigator.mediaDevices.getUserMedia({
      audio: false,
      video: {
        facingMode: { ideal: "environment" },
        width: { ideal: 1280 },
        height: { ideal: 720 }
      }
    });
    video.srcObject = stream;
    await video.play();

    overlay.width = video.videoWidth;
    overlay.height = video.videoHeight;
    placeholder.hidden = true;
    running = true;
    startBtn.disabled = true;
    stopBtn.disabled = false;
    statusText.textContent = "Cámara activa. Muestra una fruta frente a la cámara.";
    lastFrameAt = 0;
    lastFpsAt = performance.now();
    fpsFrames = 0;
    rafId = requestAnimationFrame(frameLoop);
  } catch (error) {
    console.error(error);
    statusText.textContent = "No se pudo iniciar la cámara: " + (error?.message ?? error);
  }
}

function stopCamera() {
  running = false;
  cancelAnimationFrame(rafId);
  if (stream) stream.getTracks().forEach(track => track.stop());
  stream = null;
  video.srcObject = null;
  ctx.clearRect(0, 0, overlay.width, overlay.height);
  placeholder.hidden = false;
  startBtn.disabled = !model;
  stopBtn.disabled = true;
  className.textContent = "Esperando cámara";
  confidenceValue.textContent = "--%";
  confidenceBar.style.width = "0%";
  inferenceTime.textContent = "-- ms";
  fpsLabel.textContent = "-- FPS";
  statusText.textContent = "Cámara detenida.";
}

startBtn.addEventListener("click", startCamera);
stopBtn.addEventListener("click", stopCamera);
startBtn.disabled = true;
stopBtn.disabled = true;

if ("serviceWorker" in navigator && location.protocol.startsWith("http")) {
  window.addEventListener("load", () => {
    navigator.serviceWorker.register("./sw.js").catch(error => console.warn("Service worker no registrado:", error));
  });
}

void initializeModel();
