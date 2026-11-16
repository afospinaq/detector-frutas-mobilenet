import {
  loadLiteRt,
  loadAndCompile,
  Tensor
} from "https://cdn.jsdelivr.net/npm/@litertjs/core@2.6.0/+esm";

// ============================================================
// MODELO Y CONFIGURACIÓN
// ============================================================

const MODEL_URL = new URL(
  "./models/MobileNetV2_Deteccion_Frutas_Batch1.tflite",
  import.meta.url
).href;

const WASM_URL =
  "https://cdn.jsdelivr.net/npm/@litertjs/core@2.6.0/wasm/";

const SIZE = 224;
const CLASSES = ["AVOCADO", "MANGO", "PAPAYA", "TOMATE"];
const TARGET_INTERVAL_MS = 120;

// El conversor validó: salida 0 = clase; salida 1 = bounding_box.
// La comparación Keras/TFLite dio errores menores a 0.000001.
const CLASS_OUTPUT_INDEX = 0;
const BOX_OUTPUT_INDEX = 1;

// Formato mostrado actualmente. El orden de coordenadas de la cabeza
// de bounding box debe cotejarse con las etiquetas usadas al entrenar.
const BOX_FORMAT = "xyxy";

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
let processing = false;
let nextTimer = null;
let lastInferenceFinished = 0;
let inferenceCount = 0;

// ============================================================
// INTERFAZ Y AYUDAS
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

// ============================================================
// CARGA DEL MODELO
// ============================================================

async function initializeModel() {
  try {
    setModelState("Cargando modelo", "loading");
    statusText.textContent = "Cargando MobileNetV2 en este dispositivo...";
    startBtn.disabled = true;

    await loadLiteRt(WASM_URL);

    // WASM es el backend estable que ya logró cargar el modelo en este móvil.
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
      throw new Error("La entrada del modelo no coincide con [1,224,224,3]: " + inputShape);
    }

    engineName.textContent = "WASM";
    setModelState("Modelo listo", "ready");
    statusText.textContent = "Modelo cargado. Inicia la cámara para detectar.";
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
// PREPROCESAMIENTO DE LA IMAGEN
// ============================================================

function createInputTensor() {
  // Mantiene el mismo procedimiento usado al validar el modelo:
  // redimensionado RGB 224x224 y MobileNetV2 preprocess_input [-1,1].
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
// INFERENCIA CONTINUA Y LIBERACIÓN DE MEMORIA
// ============================================================

async function inferOneFrame() {
  if (!model || !running || processing || video.readyState < 2) return;

  processing = true;
  const startedAt = performance.now();

  let inputTensor = null;
  let outputs = null;

  try {
    inputTensor = createInputTensor();

    // LiteRT.js devuelve un arreglo de tensores. La lectura oficial es
    // await output.data(); cada tensor se elimina por separado.
    outputs = await model.run(inputTensor);

    if (!Array.isArray(outputs) || outputs.length !== 2) {
      throw new Error("La inferencia no devolvió las dos salidas esperadas.");
    }

    const classValuesRaw = await outputs[CLASS_OUTPUT_INDEX].data();
    const boxValuesRaw = await outputs[BOX_OUTPUT_INDEX].data();

    const probabilities = Array.from(classValuesRaw).slice(0, 4);
    const box = Array.from(boxValuesRaw).slice(0, 4);

    // Si el usuario detuvo la cámara durante la inferencia,
    // no actualices una interfaz que ya se detuvo.
    if (!running) return;

    if (probabilities.length !== 4 || box.length !== 4) {
      throw new Error("Dimensiones de salida incorrectas.");
    }

    if (![...probabilities, ...box].every(Number.isFinite)) {
      throw new Error("El modelo devolvió números no válidos.");
    }

    let bestIndex = 0;
    for (let i = 1; i < probabilities.length; i++) {
      if (probabilities[i] > probabilities[bestIndex]) bestIndex = i;
    }

    const confidence = Math.max(0, Math.min(1, probabilities[bestIndex]));

    className.textContent = CLASSES[bestIndex];
    confidenceValue.textContent = (confidence * 100).toFixed(1) + "%";
    confidenceBar.style.width = (confidence * 100) + "%";
    updateScores(probabilities);

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

    drawBoundingBox(box);

    statusText.textContent =
      "Cámara activa · " + inferenceCount + " inferencias completadas";

    console.debug("PREDICCIÓN", {
      clase: CLASSES[bestIndex],
      probabilidades,
      boundingBox: box,
      inferenciaMs: (finishedAt - startedAt).toFixed(1)
    });

  } catch (error) {
    console.error("ERROR DE INFERENCIA:", error);
    statusText.textContent = "Error de inferencia: " + error.message;
  } finally {
    // Liberar siempre todos los tensores, también cuando ocurra un error.
    try {
      // LiteRT.js devuelve un contenedor de salidas con su propio delete().
      // Se elimina el contenedor completo para liberar todos los buffers.
      outputs?.delete?.();
    } catch (e) {
      console.warn("No se pudo liberar el contenedor de salidas:", e);
    }

    try { inputTensor?.delete?.(); } catch (e) {
      console.warn("No se pudo liberar la entrada:", e);
    }

    processing = false;

    // Programar el siguiente fotograma después de concluir la inferencia.
    // Así nunca se solapan ejecuciones ni se acumulan tensores en memoria.
    if (running) {
      const elapsed = performance.now() - startedAt;
      const wait = Math.max(30, TARGET_INTERVAL_MS - elapsed);
      nextTimer = setTimeout(inferOneFrame, wait);
    }
  }
}

// ============================================================
// BOUNDING BOX
// ============================================================

function drawBoundingBox(rawBox) {
  ctx.clearRect(0, 0, overlay.width, overlay.height);

  let a = Number(rawBox[0]);
  let b = Number(rawBox[1]);
  let c = Number(rawBox[2]);
  let d = Number(rawBox[3]);

  // La cabeza actual entrega coordenadas normalizadas según la conversión.
  // Si vinieran en píxeles de 224, convertirlas a proporciones.
  if (Math.max(Math.abs(a), Math.abs(b), Math.abs(c), Math.abs(d)) > 1.5) {
    a /= SIZE;
    b /= SIZE;
    c /= SIZE;
    d /= SIZE;
  }

  let leftN, topN, rightN, bottomN;

  if (BOX_FORMAT === "xyxy") {
    leftN = Math.min(a, c);
    topN = Math.min(b, d);
    rightN = Math.max(a, c);
    bottomN = Math.max(b, d);
  } else {
    // Alternativa si las etiquetas del entrenamiento fueron cx,cy,w,h.
    leftN = a - c / 2;
    topN = b - d / 2;
    rightN = a + c / 2;
    bottomN = b + d / 2;
  }

  leftN = Math.max(0, Math.min(1, leftN));
  topN = Math.max(0, Math.min(1, topN));
  rightN = Math.max(0, Math.min(1, rightN));
  bottomN = Math.max(0, Math.min(1, bottomN));

  const left = leftN * overlay.width;
  const top = topN * overlay.height;
  const width = (rightN - leftN) * overlay.width;
  const height = (bottomN - topN) * overlay.height;

  if (width < 2 || height < 2) return;

  const lineWidth = Math.max(3, overlay.width * 0.005);
  ctx.lineWidth = lineWidth;
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
// CÁMARA
// ============================================================

async function startCamera() {
  if (!model || running) return;

  try {
    statusText.textContent = "Solicitando cámara trasera...";

    cameraStream = await navigator.mediaDevices.getUserMedia({
      audio: false,
      video: {
        facingMode: { ideal: "environment" },
        width: { ideal: 1280 },
        height: { ideal: 720 },
        frameRate: { ideal: 30, max: 30 }
      }
    });

    video.srcObject = cameraStream;
    await video.play();

    const width = video.videoWidth || 1280;
    const height = video.videoHeight || 720;

    // El contenedor usa la proporción nativa para que vídeo y caja coincidan.
    cameraStage.style.aspectRatio = width + " / " + height;
    overlay.width = width;
    overlay.height = height;

    showPlaceholder(false);

    running = true;
    processing = false;
    inferenceCount = 0;
    lastInferenceFinished = 0;

    startBtn.disabled = true;
    stopBtn.disabled = false;

    statusText.textContent =
      "Cámara activa · preparando primera inferencia...";

    fpsLabel.textContent = "Calculando...";
    nextTimer = setTimeout(inferOneFrame, 50);

  } catch (error) {
    console.error("ERROR DE CÁMARA:", error);
    statusText.textContent = "No se pudo iniciar la cámara: " + error.message;

    if (cameraStream) {
      cameraStream.getTracks().forEach(track => track.stop());
      cameraStream = null;
    }
  }
}

function stopCamera() {
  running = false;

  if (nextTimer) {
    clearTimeout(nextTimer);
    nextTimer = null;
  }

  if (cameraStream) {
    cameraStream.getTracks().forEach(track => track.stop());
    cameraStream = null;
  }

  video.pause();
  video.srcObject = null;

  ctx.clearRect(0, 0, overlay.width, overlay.height);
  showPlaceholder(true);

  startBtn.disabled = !model;
  stopBtn.disabled = true;

  className.textContent = "Esperando cámara";
  confidenceValue.textContent = "--%";
  confidenceBar.style.width = "0%";
  inferenceTime.textContent = "-- ms";
  fpsLabel.textContent = "-- FPS";
  clearScores();

  statusText.textContent = "Cámara detenida.";
}

// ============================================================
// EVENTOS
// ============================================================

startBtn.addEventListener("click", startCamera);
stopBtn.addEventListener("click", stopCamera);

window.addEventListener("pagehide", () => {
  running = false;
  if (nextTimer) clearTimeout(nextTimer);
  if (cameraStream) cameraStream.getTracks().forEach(track => track.stop());
});

startBtn.disabled = true;
stopBtn.disabled = true;
showPlaceholder(true);
clearScores();

void initializeModel();
