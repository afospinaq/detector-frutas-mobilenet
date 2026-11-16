import {
  loadLiteRt,
  loadAndCompile,
  Tensor
} from "https://cdn.jsdelivr.net/npm/@litertjs/core@2.6.0/+esm";

// ============================================================
// MODELO
// ============================================================

const MODEL_URL = new URL(
  "./models/MobileNetV2_Deteccion_Frutas_Batch1.tflite",
  import.meta.url
).href;

const WASM_URL =
  "https://cdn.jsdelivr.net/npm/@litertjs/core@2.6.0/wasm/";

const SIZE = 224;
const CLASSES = ["AVOCADO", "MANGO", "PAPAYA", "TOMATE"];

const CLASS_OUTPUT_INDEX = 0;
const BOX_OUTPUT_INDEX = 1;

// Filtro provisional de abstención. El modelo actual solo conoce
// las cuatro frutas, por lo que este filtro NO sustituye una clase
// de fondo entrenada.
const MIN_CONFIDENCE = 0.72;
const MIN_CLASS_MARGIN = 0.18;
const STABILITY_WINDOW = 4;
const REQUIRED_STABLE_VOTES = 3;
const MIN_BOX_AREA = 0.012;
const MAX_BOX_AREA = 0.88;

// ============================================================
// ELEMENTOS
// ============================================================

const video = document.getElementById("camera");
const overlay = document.getElementById("overlay");
const ctx = overlay.getContext("2d");
const cameraStage = document.getElementById("cameraStage");
const placeholder = document.getElementById("cameraPlaceholder");

const startBtn = document.getElementById("startBtn");
const stopBtn = document.getElementById("stopBtn");
const statusText = document.getElementById("statusText");
const modelState = document.getElementById("modelState");

const className = document.getElementById("className");
const confidenceValue = document.getElementById("confidenceValue");
const confidenceBar = document.getElementById("confidenceBar");
const inferenceTime = document.getElementById("inferenceTime");
const engineName = document.getElementById("engineName");
const fpsLabel = document.getElementById("fpsLabel");

const inputCanvas = document.createElement("canvas");
inputCanvas.width = SIZE;
inputCanvas.height = SIZE;
const inputCtx = inputCanvas.getContext("2d", {
  willReadFrequently: true
});

const scoreElements = Object.fromEntries(
  CLASSES.map(name => [
    name,
    document.getElementById("score" + name)
  ])
);

// ============================================================
// ESTADO
// ============================================================

let model = null;
let cameraStream = null;
let running = false;
let cameraStarting = false;
let processing = false;
let nextTimer = null;
let cancelMetadataWait = null;
let cameraOperationId = 0;
let lastInferenceFinished = 0;
let inferenceCount = 0;
let predictionHistory = [];

// ============================================================
// INTERFAZ
// ============================================================

function setModelState(message, type = "loading") {
  modelState.textContent = message;
  modelState.className = "state state-" + type;
}

function showPlaceholder(show) {
  placeholder.hidden = !show;
  placeholder.style.display = show ? "flex" : "none";
  placeholder.setAttribute("aria-hidden", show ? "false" : "true");
}

function clearOverlay() {
  ctx.clearRect(0, 0, overlay.width, overlay.height);
}

function clearScores() {
  for (const name of CLASSES) {
    if (scoreElements[name]) scoreElements[name].textContent = "--%";
  }
}

function updateScores(values) {
  CLASSES.forEach((name, i) => {
    const value = Math.max(0, Math.min(1, Number(values[i]) || 0));
    if (scoreElements[name]) {
      scoreElements[name].textContent = (value * 100).toFixed(1) + "%";
    }
  });
}

function showNoReliableDetection(message = "Sin detección confiable") {
  className.textContent = "Sin detección confiable";
  confidenceValue.textContent = "--%";
  confidenceBar.style.width = "0%";
  clearOverlay();
  statusText.textContent = message;
}

// ============================================================
// CARGA DEL MODELO
// ============================================================

async function initializeModel() {
  try {
    setModelState("Cargando modelo", "loading");
    statusText.textContent = "Cargando MobileNetV2 en este dispositivo...";
    startBtn.disabled = true;
    stopBtn.disabled = true;

    await loadLiteRt(WASM_URL);
    model = await loadAndCompile(MODEL_URL, {
      accelerator: "wasm"
    });

    const inputDetails = model.getInputDetails();
    const outputDetails = model.getOutputDetails();

    console.info("ENTRADA DEL MODELO:", inputDetails);
    console.info("SALIDAS DEL MODELO:", outputDetails);

    if (!inputDetails?.length || !outputDetails || outputDetails.length !== 2) {
      throw new Error("El modelo no presenta una entrada y dos salidas como se esperaba.");
    }

    const inputShape = Array.from(inputDetails[0].shape);
    if (inputShape.join(",") !== "1,224,224,3") {
      throw new Error("La entrada no coincide con [1,224,224,3]: " + inputShape);
    }

    engineName.textContent = "WASM";
    setModelState("Modelo listo", "ready");
    statusText.textContent = "Modelo cargado. Inicia la cámara.";
    startBtn.disabled = false;

  } catch (error) {
    console.error("ERROR CARGANDO MODELO:", error);
    model = null;
    setModelState("Error del modelo", "error");
    statusText.textContent = "No se pudo cargar el modelo: " + error.message;
    startBtn.disabled = true;
  }
}

// ============================================================
// PREPROCESAMIENTO
// ============================================================

function createInputTensor() {
  inputCtx.drawImage(video, 0, 0, SIZE, SIZE);
  const pixels = inputCtx.getImageData(0, 0, SIZE, SIZE).data;
  const values = new Float32Array(SIZE * SIZE * 3);

  let j = 0;
  for (let i = 0; i < pixels.length; i += 4) {
    values[j++] = pixels[i] / 127.5 - 1;
    values[j++] = pixels[i + 1] / 127.5 - 1;
    values[j++] = pixels[i + 2] / 127.5 - 1;
  }

  return new Tensor(values, [1, SIZE, SIZE, 3]);
}

// ============================================================
// VALIDAR CAJA Y ESTABILIDAD
// ============================================================

function normalizeBox(rawBox) {
  let [x1, y1, x2, y2] = rawBox.map(Number);

  if (![x1, y1, x2, y2].every(Number.isFinite)) return null;

  if (Math.max(Math.abs(x1), Math.abs(y1), Math.abs(x2), Math.abs(y2)) > 1.5) {
    x1 /= SIZE;
    y1 /= SIZE;
    x2 /= SIZE;
    y2 /= SIZE;
  }

  const left = Math.max(0, Math.min(1, Math.min(x1, x2)));
  const top = Math.max(0, Math.min(1, Math.min(y1, y2)));
  const right = Math.max(0, Math.min(1, Math.max(x1, x2)));
  const bottom = Math.max(0, Math.min(1, Math.max(y1, y2)));

  const width = right - left;
  const height = bottom - top;
  const area = width * height;

  if (width < 0.05 || height < 0.05) return null;
  if (area < MIN_BOX_AREA || area > MAX_BOX_AREA) return null;

  return [left, top, right, bottom];
}

function addPredictionCandidate(candidate) {
  predictionHistory.push(candidate);
  if (predictionHistory.length > STABILITY_WINDOW) {
    predictionHistory.shift();
  }

  if (!candidate) return null;

  const matching = predictionHistory.filter(
    item => item && item.classIndex === candidate.classIndex
  );

  if (matching.length < REQUIRED_STABLE_VOTES) return null;

  const averageConfidence =
    matching.reduce((sum, item) => sum + item.confidence, 0) / matching.length;

  const averageBox = [0, 1, 2, 3].map(index =>
    matching.reduce((sum, item) => sum + item.box[index], 0) / matching.length
  );

  return {
    classIndex: candidate.classIndex,
    confidence: averageConfidence,
    box: averageBox
  };
}

// ============================================================
// INFERENCIA CONTINUA
// ============================================================

async function inferOneFrame() {
  if (!model || !running || processing || video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA) {
    return;
  }

  processing = true;
  const startedAt = performance.now();

  let inputTensor = null;
  let outputs = null;

  try {
    inputTensor = createInputTensor();
    outputs = await model.run(inputTensor);

    if (!Array.isArray(outputs) || outputs.length !== 2) {
      throw new Error("La inferencia no devolvió dos salidas.");
    }

    const rawClasses = Array.from(await outputs[CLASS_OUTPUT_INDEX].data()).slice(0, 4);
    const rawBox = Array.from(await outputs[BOX_OUTPUT_INDEX].data()).slice(0, 4);

    if (!running) return;

    if (rawClasses.length !== 4 || rawBox.length !== 4) {
      throw new Error("Dimensiones de salida incorrectas.");
    }

    if (![...rawClasses, ...rawBox].every(Number.isFinite)) {
      throw new Error("El modelo devolvió valores no válidos.");
    }

    updateScores(rawClasses);

    const ranked = rawClasses
      .map((score, index) => ({ score, index }))
      .sort((a, b) => b.score - a.score);

    const top = ranked[0];
    const second = ranked[1];
    const margin = top.score - second.score;
    const box = normalizeBox(rawBox);

    let candidate = null;
    if (
      top.score >= MIN_CONFIDENCE &&
      margin >= MIN_CLASS_MARGIN &&
      box !== null
    ) {
      candidate = {
        classIndex: top.index,
        confidence: top.score,
        box
      };
    }

    const stable = addPredictionCandidate(candidate);

    const finishedAt = performance.now();
    inferenceTime.textContent = (finishedAt - startedAt).toFixed(0) + " ms";

    if (lastInferenceFinished > 0) {
      const interval = finishedAt - lastInferenceFinished;
      fpsLabel.textContent = (1000 / Math.max(interval, 1)).toFixed(1) + " FPS";
    } else {
      fpsLabel.textContent = "Calculando...";
    }

    lastInferenceFinished = finishedAt;
    inferenceCount++;

    if (stable) {
      className.textContent = CLASSES[stable.classIndex];
      confidenceValue.textContent = (stable.confidence * 100).toFixed(1) + "%";
      confidenceBar.style.width = (stable.confidence * 100) + "%";
      drawBoundingBox(stable.box);

      statusText.textContent =
        "Detección estable · " + inferenceCount + " inferencias";
    } else {
      showNoReliableDetection(
        candidate
          ? "Verificando estabilidad de la fruta..."
          : "Esperando una fruta clara y una predicción confiable..."
      );
    }

    console.debug("INFERENCIA", {
      scores: rawClasses,
      topClass: CLASSES[top.index],
      topConfidence: top.score,
      margin,
      box: rawBox,
      accepted: Boolean(stable)
    });

  } catch (error) {
    console.error("ERROR DE INFERENCIA:", error);
    predictionHistory = [];
    showNoReliableDetection("Error de inferencia: " + error.message);

  } finally {
    if (Array.isArray(outputs)) {
      for (const tensor of outputs) {
        try { tensor?.delete?.(); } catch (error) {
          console.warn("No se pudo liberar un tensor de salida:", error);
        }
      }
    }

    try { inputTensor?.delete?.(); } catch (error) {
      console.warn("No se pudo liberar el tensor de entrada:", error);
    }

    processing = false;

    // La próxima inferencia comienza solo después de terminar y liberar
    // los recursos de la actual.
    if (running) {
      const elapsed = performance.now() - startedAt;
      nextTimer = setTimeout(
        inferOneFrame,
        Math.max(40, 140 - elapsed)
      );
    }
  }
}

// ============================================================
// DIBUJAR BOUNDING BOX
// ============================================================

function drawBoundingBox(box) {
  clearOverlay();

  const [leftN, topN, rightN, bottomN] = box;
  const left = leftN * overlay.width;
  const top = topN * overlay.height;
  const width = (rightN - leftN) * overlay.width;
  const height = (bottomN - topN) * overlay.height;

  if (width < 2 || height < 2) return;

  ctx.lineWidth = Math.max(3, overlay.width * 0.005);
  ctx.strokeStyle = "#70F59A";
  ctx.strokeRect(left, top, width, height);

  const label = className.textContent;
  ctx.font = "bold 18px Arial";

  const labelWidth = ctx.measureText(label).width + 14;
  const labelHeight = 27;
  const labelTop = Math.max(0, top - labelHeight);

  ctx.fillStyle = "#70F59A";
  ctx.fillRect(left, labelTop, labelWidth, labelHeight);
  ctx.fillStyle = "#082014";
  ctx.fillText(label, left + 7, labelTop + 19);
}

// ============================================================
// INICIO ROBUSTO DE CÁMARA
// ============================================================

function waitForVideoMetadata(stream, operationId) {
  if (
    video.readyState >= HTMLMediaElement.HAVE_METADATA &&
    video.videoWidth > 0
  ) {
    return Promise.resolve();
  }

  return new Promise((resolve, reject) => {
    let settled = false;

    const cleanup = () => {
      clearTimeout(timeoutId);
      video.removeEventListener("loadedmetadata", onLoaded);
      video.removeEventListener("error", onError);
      const track = stream.getVideoTracks()[0];
      track?.removeEventListener("ended", onTrackEnded);
      if (cancelMetadataWait === cancel) cancelMetadataWait = null;
    };

    const finish = (callback, value) => {
      if (settled) return;
      settled = true;
      cleanup();
      callback(value);
    };

    const cancel = () => {
      finish(
        reject,
        new DOMException("Inicio de cámara cancelado.", "AbortError")
      );
    };

    const onLoaded = () => {
      if (
        operationId !== cameraOperationId ||
        video.srcObject !== stream
      ) {
        cancel();
        return;
      }
      if (video.videoWidth > 0) finish(resolve);
    };

    const onError = () => {
      finish(reject, new Error("El elemento de vídeo no pudo cargar la cámara."));
    };

    const onTrackEnded = () => {
      finish(
        reject,
        new DOMException("La cámara se detuvo durante el inicio.", "AbortError")
      );
    };

    const timeoutId = setTimeout(() => {
      finish(reject, new Error("La cámara no entregó metadatos de vídeo a tiempo."));
    }, 12000);

    cancelMetadataWait = cancel;
    video.addEventListener("loadedmetadata", onLoaded);
    video.addEventListener("error", onError);
    stream.getVideoTracks()[0]?.addEventListener("ended", onTrackEnded, { once: true });

    // Comprobar de nuevo por si loadedmetadata ocurrió antes de registrar el listener.
    onLoaded();
  });
}

async function startCamera() {
  if (!model || running || cameraStarting) return;

  const operationId = ++cameraOperationId;
  cameraStarting = true;
  startBtn.disabled = true;
  stopBtn.disabled = false;

  statusText.textContent = "Solicitando acceso a la cámara...";
  setModelState("Cámara iniciando", "loading");

  let newStream = null;

  try {
    newStream = await navigator.mediaDevices.getUserMedia({
      audio: false,
      video: {
        facingMode: { ideal: "environment" },
        width: { ideal: 1280 },
        height: { ideal: 720 },
        frameRate: { ideal: 30, max: 30 }
      }
    });

    if (operationId !== cameraOperationId) {
      newStream.getTracks().forEach(track => track.stop());
      return;
    }

    cameraStream = newStream;

    // autoplay se desactiva en HTML; solo hacemos un play() explícito.
    video.muted = true;
    video.playsInline = true;
    video.srcObject = newStream;

    await waitForVideoMetadata(newStream, operationId);

    if (
      operationId !== cameraOperationId ||
      video.srcObject !== newStream
    ) {
      newStream.getTracks().forEach(track => track.stop());
      return;
    }

    // No se solicita otra reproducción ni se cambia srcObject mientras
    // esta promesa está pendiente.
    await video.play();

    if (
      operationId !== cameraOperationId ||
      video.srcObject !== newStream
    ) {
      newStream.getTracks().forEach(track => track.stop());
      return;
    }

    const width = video.videoWidth || 1280;
    const height = video.videoHeight || 720;

    cameraStage.style.aspectRatio = width + " / " + height;
    overlay.width = width;
    overlay.height = height;

    showPlaceholder(false);

    running = true;
    processing = false;
    predictionHistory = [];
    inferenceCount = 0;
    lastInferenceFinished = 0;

    setModelState("Modelo listo", "ready");
    startBtn.disabled = true;
    stopBtn.disabled = false;

    className.textContent = "Buscando fruta...";
    confidenceValue.textContent = "--%";
    clearScores();

    statusText.textContent = "Cámara activa · buscando una fruta...";
    fpsLabel.textContent = "Calculando...";
    nextTimer = setTimeout(inferOneFrame, 80);

  } catch (error) {
    if (operationId === cameraOperationId) {
      console.error("ERROR DE CÁMARA:", error);

      if (newStream) {
        newStream.getTracks().forEach(track => track.stop());
      }

      if (video.srcObject === newStream) {
        video.pause();
        video.srcObject = null;
      }

      cameraStream = null;
      running = false;
      predictionHistory = [];

      const isAbort =
        error?.name === "AbortError" ||
        String(error?.message || "").includes("interrupted");

      statusText.textContent = isAbort
        ? "El inicio de cámara se interrumpió. Pulsa Iniciar cámara una sola vez."
        : "No se pudo iniciar la cámara: " + error.message;

      setModelState("Modelo listo", "ready");
      showPlaceholder(true);
    }

  } finally {
    if (operationId === cameraOperationId) {
      cameraStarting = false;
      startBtn.disabled = !model || running;
      stopBtn.disabled = !running;
    }
  }
}

function stopCamera(message = "Cámara detenida.") {
  // Invalida cualquier getUserMedia / play pendiente.
  cameraOperationId++;

  if (cancelMetadataWait) {
    const cancel = cancelMetadataWait;
    cancelMetadataWait = null;
    cancel();
  }

  cameraStarting = false;
  running = false;
  processing = false;

  if (nextTimer) {
    clearTimeout(nextTimer);
    nextTimer = null;
  }

  const oldStream = cameraStream;
  cameraStream = null;

  if (oldStream) {
    oldStream.getTracks().forEach(track => track.stop());
  }

  // Pausar y desmontar una sola vez al detener; nunca durante el inicio.
  try { video.pause(); } catch (error) {
    console.warn("No se pudo pausar el vídeo:", error);
  }

  video.srcObject = null;

  clearOverlay();
  showPlaceholder(true);
  predictionHistory = [];

  startBtn.disabled = !model;
  stopBtn.disabled = true;

  className.textContent = "Esperando cámara";
  confidenceValue.textContent = "--%";
  confidenceBar.style.width = "0%";
  inferenceTime.textContent = "-- ms";
  fpsLabel.textContent = "-- FPS";
  clearScores();

  statusText.textContent = message;
  setModelState(model ? "Modelo listo" : "Sin modelo", model ? "ready" : "error");
}

// ============================================================
// EVENTOS
// ============================================================

startBtn.addEventListener("click", startCamera);
stopBtn.addEventListener("click", () => stopCamera());

document.addEventListener("visibilitychange", () => {
  if (document.hidden && (running || cameraStarting)) {
    stopCamera("Cámara detenida porque saliste de esta pestaña. Pulsa Iniciar cámara al regresar.");
  }
});

window.addEventListener("pagehide", () => {
  stopCamera("Página cerrada.");
});

startBtn.disabled = true;
stopBtn.disabled = true;
showPlaceholder(true);
clearScores();

void initializeModel();
