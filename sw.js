const CACHE_NAME = "fruit-detector-v1";
const APP_FILES = ["./", "./index.html", "./styles.css", "./app.js", "./manifest.webmanifest"];

self.addEventListener("install", event => {
  event.waitUntil(caches.open(CACHE_NAME).then(cache => cache.addAll(APP_FILES)));
  self.skipWaiting();
});

self.addEventListener("activate", event => {
  event.waitUntil((async () => {
    const names = await caches.keys();
    await Promise.all(names.filter(name => name.startsWith("fruit-detector-") && name !== CACHE_NAME).map(name => caches.delete(name)));
    await self.clients.claim();
  })());
});

self.addEventListener("fetch", event => {
  const request = event.request;
  const url = new URL(request.url);
  if (request.method !== "GET" || url.origin !== self.location.origin) return;

  // Guarda el modelo después de descargarlo; las siguientes aperturas pueden
  // reutilizar la copia local del navegador.
  const isAppAsset = url.pathname.endsWith(".tflite") ||
    url.pathname.endsWith(".js") ||
    url.pathname.endsWith(".css") ||
    url.pathname.endsWith(".html") ||
    url.pathname.endsWith(".webmanifest");

  if (!isAppAsset) return;

  event.respondWith((async () => {
    const cache = await caches.open(CACHE_NAME);
    const cached = await cache.match(request);
    if (cached) return cached;

    const response = await fetch(request);
    if (response.ok) event.waitUntil(cache.put(request, response.clone()));
    return response;
  })());
});
