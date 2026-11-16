# Detector de frutas MobileNetV2

Aplicación web para ejecutar MobileNetV2 localmente en el navegador Android mediante LiteRT.js y WebAssembly.

Clases: AVOCADO, MANGO, PAPAYA y TOMATE.

## Archivos del modelo necesarios

Coloca el modelo en `models/MobileNetV2_Deteccion_Frutas_Fijo.tflite` y los archivos WASM de LiteRT.js en `wasm/`.

## Publicación

En GitHub: Settings → Pages → Build and deployment → Deploy from a branch → main → /(root).

La cámara requiere HTTPS (GitHub Pages lo proporciona) y permiso del usuario.
