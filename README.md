# Detector de frutas MobileNetV2

Aplicación web para clasificar **AVOCADO, MANGO, PAPAYA y TOMATE** en tiempo real con la cámara de Android. La inferencia se realiza en el navegador con LiteRT.js; los fotogramas no se envían a un servidor.

## Modelo

Sube el archivo del modelo a esta ruta exacta:

`models/MobileNetV2_Deteccion_Frutas_Fijo.tflite`

Los archivos WebAssembly de LiteRT.js se cargan desde jsDelivr; no es necesario subir la carpeta `wasm/`.

## Publicar con GitHub Pages

1. Abre **Settings → Pages**.
2. En **Build and deployment → Source**, selecciona **GitHub Actions**.
3. El workflow `.github/workflows/deploy.yml` publica automáticamente el sitio cuando se actualiza `main`.
4. Revisa la pestaña **Actions** hasta que el workflow finalice correctamente.

Después abre:

`https://afospinaq.github.io/detector-frutas-mobilenet/`

## Ejecución

Chrome Android solicita permiso para usar la cámara. El navegador descarga el modelo al dispositivo y ejecuta las predicciones localmente. La primera descarga del modelo requiere Internet; el service worker almacena los recursos propios del sitio en la caché del navegador.

## Modelo y preprocesamiento

- Entrada: `[1, 224, 224, 3]`, RGB float32.
- Preprocesamiento: MobileNetV2 `preprocess_input`, es decir, canal RGB transformado de [0,255] a [-1,1].
- Salida 0: probabilidades de clase en el orden AVOCADO, MANGO, PAPAYA, TOMATE.
- Salida 1: bounding box normalizado `[x_min, y_min, x_max, y_max]`.
