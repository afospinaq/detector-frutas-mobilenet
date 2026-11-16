
import {
    loadLiteRt,
    loadAndCompile,
    Tensor
} from "https://cdn.jsdelivr.net/npm/@litertjs/core@2.6.0/+esm";

// ======================================================
// CONFIGURACIÓN
// ======================================================

const MODEL_URL = new URL(
    "./models/MobileNetV2_Deteccion_Frutas_Batch1.tflite",
    import.meta.url
).href;

const WASM_URL =
    "https://cdn.jsdelivr.net/npm/@litertjs/core@2.6.0/wasm/";

const IMAGE_SIZE = 224;

const CLASSES = [
    "AVOCADO",
    "MANGO",
    "PAPAYA",
    "TOMATE"
];

const FRAME_INTERVAL = 100;

// ======================================================
// ELEMENTOS HTML
// ======================================================

const video = document.getElementById("camera");
const canvas = document.getElementById("overlay");
const ctx = canvas.getContext("2d");

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

const placeholder = document.getElementById("cameraPlaceholder");
const cameraStage = document.getElementById("cameraStage");

// ======================================================
// CANVAS PARA PREPROCESAMIENTO
// ======================================================

const inputCanvas = document.createElement("canvas");

inputCanvas.width = IMAGE_SIZE;
inputCanvas.height = IMAGE_SIZE;

const inputCtx = inputCanvas.getContext("2d", {
    willReadFrequently: true
});

// ======================================================
// VARIABLES GLOBALES
// ======================================================

let model = null;
let cameraStream = null;

let running = false;
let processing = false;

let animationId = null;

let lastInference = 0;

let fpsFrames = 0;
let lastFpsUpdate = performance.now();

// ======================================================
// ESTADO VISUAL
// ======================================================

function updateModelState(message, type = "loading") {
    modelState.textContent = message;
    modelState.className = "state state-" + type;
}

// ======================================================
// CARGAR MODELO
// ======================================================

async function initializeModel() {

    try {

        updateModelState("Cargando modelo", "loading");

        statusText.textContent =
            "Preparando el modelo MobileNetV2...";

        startBtn.disabled = true;

        // Cargar runtime de LiteRT.
        await loadLiteRt(WASM_URL);

        // Intentar aceleración mediante GPU.
        try {

            model = await loadAndCompile(
                MODEL_URL,
                { accelerator: "webgpu" }
            );

            engineName.textContent = "WebGPU";

        } catch (gpuError) {

            console.warn(
                "WebGPU no disponible. Se utilizará WASM.",
                gpuError
            );

            model = await loadAndCompile(
                MODEL_URL,
                { accelerator: "wasm" }
            );

            engineName.textContent = "WASM";
        }

        // Inspeccionar las salidas del modelo.
        console.log(
            "ENTRADAS:",
            model.getInputDetails()
        );

        console.log(
            "SALIDAS:",
            model.getOutputDetails()
        );

        updateModelState("Modelo listo", "ready");

        statusText.textContent =
            "MobileNetV2 cargado. Puedes iniciar la cámara.";

        startBtn.disabled = false;

    } catch (error) {

        console.error("ERROR AL CARGAR MODELO:", error);

        updateModelState("Error del modelo", "error");

        statusText.textContent =
            "No se pudo cargar el modelo. Verifica la ruta y el archivo TFLite.";

    }
}

// ======================================================
// PREPROCESAMIENTO
// ======================================================

function preprocessFrame() {

    // Redimensionar a 224x224.
    inputCtx.drawImage(
        video,
        0,
        0,
        IMAGE_SIZE,
        IMAGE_SIZE
    );

    const imageData = inputCtx.getImageData(
        0,
        0,
        IMAGE_SIZE,
        IMAGE_SIZE
    );

    const pixels = imageData.data;

    const inputData = new Float32Array(
        IMAGE_SIZE * IMAGE_SIZE * 3
    );

    let index = 0;

    // RGB y normalización MobileNetV2 [-1, 1].
    for (let i = 0; i < pixels.length; i += 4) {

        inputData[index++] =
            pixels[i] / 127.5 - 1;

        inputData[index++] =
            pixels[i + 1] / 127.5 - 1;

        inputData[index++] =
            pixels[i + 2] / 127.5 - 1;
    }

    return inputData;
}

// ======================================================
// IDENTIFICAR LAS DOS SALIDAS
// ======================================================

function identifyOutputs(values0, values1) {

    const details = model.getOutputDetails();

    const names = details.map(
        item => String(item.name || "").toLowerCase()
    );

    let classIndex = names.findIndex(
        name =>
            name.includes("clase") ||
            name.includes("class") ||
            name.includes("softmax") ||
            name.includes("prob")
    );

    let boxIndex = names.findIndex(
        name =>
            name.includes("bounding") ||
            name.includes("bbox") ||
            name.includes("box")
    );

    // Si los nombres no identifican las salidas,
    // buscar cuál tiene comportamiento de softmax.
    if (classIndex === -1 || boxIndex === -1) {

        function probabilityScore(values) {

            const valid = values.every(
                value =>
                    Number.isFinite(value) &&
                    value >= 0 &&
                    value <= 1
            );

            const sum = values.reduce(
                (total, value) => total + value,
                0
            );

            return valid
                ? Math.abs(sum - 1)
                : Infinity;
        }

        const score0 = probabilityScore(values0);
        const score1 = probabilityScore(values1);

        if (score0 < score1) {

            classIndex = 0;
            boxIndex = 1;

        } else if (score1 < score0) {

            classIndex = 1;
            boxIndex = 0;

        } else {

            // Orden de salida original del modelo Keras:
            // bounding_box, clase.
            classIndex = 1;
            boxIndex = 0;
        }
    }

    return {
        classIndex,
        boxIndex
    };
}

// ======================================================
// INFERENCIA
// ======================================================

async function classifyFrame() {

    if (
        !model ||
        !running ||
        processing ||
        video.readyState < 2
    ) {
        return;
    }

    processing = true;

    const startedAt = performance.now();

    let inputTensor = null;
    let results = null;

    let output0 = null;
    let output1 = null;

    try {

        // Preparar imagen.
        const inputData = preprocessFrame();

        inputTensor = new Tensor(
            inputData,
            [1, IMAGE_SIZE, IMAGE_SIZE, 3]
        );

        // Ejecutar modelo.
        results = await model.run(inputTensor);

        if (!results || results.length !== 2) {
            throw new Error(
                "Se esperaban dos salidas del modelo."
            );
        }

        // Pasar las salidas a memoria WASM.
        output0 = await results[0].moveTo("wasm");
        output1 = await results[1].moveTo("wasm");

        const values0 = Array.from(
            output0.toTypedArray()
        );

        const values1 = Array.from(
            output1.toTypedArray()
        );

        if (
            values0.length !== 4 ||
            values1.length !== 4
        ) {
            throw new Error(
                "Las salidas no tienen cuatro valores."
            );
        }

        const indexes = identifyOutputs(
            values0,
            values1
        );

        const probabilities =
            indexes.classIndex === 0
                ? values0
                : values1;

        const boundingBox =
            indexes.boxIndex === 0
                ? values0
                : values1;

        // Obtener clase más probable.
        let bestIndex = 0;

        for (let i = 1; i < probabilities.length; i++) {

            if (
                probabilities[i] >
                probabilities[bestIndex]
            ) {
                bestIndex = i;
            }
        }

        const confidence = Math.max(
            0,
            Math.min(1, probabilities[bestIndex])
        );

        // Actualizar interfaz.
        className.textContent = CLASSES[bestIndex];

        confidenceValue.textContent =
            (confidence * 100).toFixed(2) + "%";

        confidenceBar.style.width =
            (confidence * 100) + "%";

        inferenceTime.textContent =
            (performance.now() - startedAt).toFixed(0) + " ms";

        // Dibujar bounding box.
        drawBoundingBox(boundingBox);
        fpsFrames++;

        console.log(
            "CLASE:",
            CLASSES[bestIndex],
            "PROBABILIDADES:",
            probabilities,
            "BOUNDING BOX:",
            boundingBox
        );

    } catch (error) {

        console.error("ERROR DE INFERENCIA:", error);

        statusText.textContent =
            "Error de inferencia: " + error.message;

    } finally {

        // Liberar memoria.
        inputTensor?.delete();

        output0?.delete();
        output1?.delete();

        results?.delete();

        processing = false;
    }
}

// ======================================================
// DIBUJAR BOUNDING BOX
// ======================================================

function drawBoundingBox(values) {

    ctx.clearRect(
        0,
        0,
        canvas.width,
        canvas.height
    );

    if (!values || values.length !== 4) {
        return;
    }

    let [x1, y1, x2, y2] = values;

    // Si las coordenadas están expresadas entre 0 y 224,
    // convertirlas a coordenadas normalizadas.
    if (Math.max(
        Math.abs(x1),
        Math.abs(y1),
        Math.abs(x2),
        Math.abs(y2)
    ) > 1.5) {

        x1 /= IMAGE_SIZE;
        y1 /= IMAGE_SIZE;
        x2 /= IMAGE_SIZE;
        y2 /= IMAGE_SIZE;
    }

    // Coordenadas normalizadas.
    const leftNorm = Math.max(
        0,
        Math.min(1, Math.min(x1, x2))
    );

    const topNorm = Math.max(
        0,
        Math.min(1, Math.min(y1, y2))
    );

    const rightNorm = Math.max(
        0,
        Math.min(1, Math.max(x1, x2))
    );

    const bottomNorm = Math.max(
        0,
        Math.min(1, Math.max(y1, y2))
    );

    const left = leftNorm * canvas.width;
    const top = topNorm * canvas.height;

    const width =
        (rightNorm - leftNorm) * canvas.width;

    const height =
        (bottomNorm - topNorm) * canvas.height;

    if (width < 2 || height < 2) {
        return;
    }

    ctx.strokeStyle = "#70F59A";

    ctx.lineWidth = Math.max(
        3,
        canvas.width * 0.005
    );

    ctx.strokeRect(
        left,
        top,
        width,
        height
    );

    // Etiqueta.
    ctx.fillStyle = "#70F59A";
    ctx.font = "bold 18px Arial";

    ctx.fillText(
        className.textContent,
        left + 4,
        Math.max(22, top - 8)
    );
}

// ======================================================
// BUCLE DE CLASIFICACIÓN CONTINUA
// ======================================================

function processFrames(now) {

    if (!running) {
        return;
    }

    if (
        !processing &&
        now - lastInference >= FRAME_INTERVAL
    ) {

        lastInference = now;

        classifyFrame();
    }

    if (now - lastFpsUpdate >= 1000) {

        fpsLabel.textContent =
            fpsFrames + " FPS";

        fpsFrames = 0;

        lastFpsUpdate = now;
    }

    animationId = requestAnimationFrame(
        processFrames
    );
}

// ======================================================
// INICIAR CÁMARA
// ======================================================

async function startCamera() {

    try {

        if (!model) {
            throw new Error(
                "El modelo todavía no está cargado."
            );
        }

        cameraStream =
            await navigator.mediaDevices.getUserMedia({

                audio: false,

                video: {
                    facingMode: {
                        ideal: "environment"
                    },

                    width: {
                        ideal: 1280
                    },

                    height: {
                        ideal: 720
                    }
                }
            });

        video.srcObject = cameraStream;

        await video.play();

        // Relación de aspecto real de la cámara: evita deformar la vista.
        const cameraSettings =
            cameraStream.getVideoTracks()[0]?.getSettings() || {};

        const frameWidth =
            video.videoWidth || cameraSettings.width || 1280;

        const frameHeight =
            video.videoHeight || cameraSettings.height || 720;

        cameraStage.style.aspectRatio =
            `${frameWidth} / ${frameHeight}`;

        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;

        placeholder.hidden = true;

        running = true;

        startBtn.disabled = true;
        stopBtn.disabled = false;

        statusText.textContent =
            "Cámara activa. Muestra una fruta.";

        lastInference = 0;
        lastFpsUpdate = performance.now();
        fpsFrames = 0;

        animationId = requestAnimationFrame(
            processFrames
        );

    } catch (error) {

        console.error("ERROR DE CÁMARA:", error);

        statusText.textContent =
            "No se pudo iniciar la cámara: " +
            error.message;
    }
}

// ======================================================
// DETENER CÁMARA
// ======================================================

function stopCamera() {

    running = false;

    cancelAnimationFrame(animationId);

    if (cameraStream) {

        cameraStream.getTracks().forEach(
            track => track.stop()
        );

        cameraStream = null;
    }

    video.srcObject = null;

    ctx.clearRect(
        0,
        0,
        canvas.width,
        canvas.height
    );

    placeholder.hidden = false;

    startBtn.disabled = !model;
    stopBtn.disabled = true;

    className.textContent = "Esperando cámara";

    confidenceValue.textContent = "--%";
    confidenceBar.style.width = "0%";

    inferenceTime.textContent = "-- ms";
    fpsLabel.textContent = "-- FPS";

    statusText.textContent =
        "Cámara detenida.";
}

// ======================================================
// EVENTOS
// ======================================================

startBtn.addEventListener(
    "click",
    startCamera
);

stopBtn.addEventListener(
    "click",
    stopCamera
);

startBtn.disabled = true;
stopBtn.disabled = true;

// ======================================================
// INICIAR APLICACIÓN
// ======================================================

void initializeModel();
