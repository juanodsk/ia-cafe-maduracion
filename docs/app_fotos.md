# Aplicación local para analizar fotos de café

La interfaz está hecha con React y Vite. Un servidor FastAPI procesa las fotos con el mismo `CameraPipeline` de la cámara: YOLO11n localiza frutos y el clasificador elegido predice su madurez. La app usa los pesos existentes; no vuelve a entrenarlos ni cambia la partición 70 % / 15 % / 15 %.

## Abrir

En la carpeta del proyecto, ejecuta `iniciar_app.cmd` o usa PowerShell:

```powershell
.\.venv\Scripts\python.exe -m cafe_ml.web_api
```

Abre **http://127.0.0.1:8765** y conserva la consola abierta. Ctrl+C detiene el servidor. Si el puerto está ocupado, utiliza `--port 8766` y abre ese puerto.

1. Selecciona o arrastra una foto JPG, PNG o WEBP.
2. Elige un modelo y pulsa **Analizar madurez**. La primera carga de los pesos tarda más.
3. Consulta los recuadros numerados y los resultados por fruto. Selecciona un recuadro o una fila para resaltarlo.
4. Descarga la foto anotada o el informe JSON si deseas guardarlos.

El botón de ejemplo utiliza una foto del conjunto de **validación**, identificada así en la interfaz. Es una demostración del flujo y no una evaluación independiente. Para comprobar el comportamiento con fotos nuevas, carga tus propias imágenes de frutos verdes, pintones y maduros bajo distintas condiciones.

## Qué representan los resultados

Los estados son Verde, Pintón, Maduro, Sobremaduro y Seco. Cuando la mayor puntuación del clasificador es menor al **55 %**, la etiqueta es **Incierto** y se indica la clase candidata. El porcentaje corresponde al softmax del clasificador para ese fruto, o al promedio de softmax para el ensemble. No es una certeza calibrada ni la precisión global del sistema. Los recuentos reflejan únicamente los frutos detectados; algunos pueden no localizarse.

EfficientNet B0 es la opción predeterminada y utiliza el checkpoint de ajuste fino. También están disponibles ResNet50, DenseNet121, ViT B/16 y Swin T, más el ensemble de los cinco. La localización utiliza `experimentos/detector_01/weights/best.pt`. Los clasificadores se restauran mediante `cafe_ml.predict.load_classifier`, igual que en la cámara. Se usa CUDA cuando está disponible y CPU en caso contrario.

## Tratamiento de imágenes

El servidor escucha únicamente en `127.0.0.1`. Las fotos se mantienen en memoria y no se envían a servicios externos ni se guardan automáticamente. Los límites son 15 MB y 20 megapíxeles. Se corrige la orientación EXIF y se reduce el lado mayor a un máximo de 1920 píxeles, conservando la proporción. Los recuadros y las coordenadas JSON corresponden a esta imagen procesada.

El sistema es el mismo detector de cajas utilizado en la cámara: señala cada fruto con un rectángulo, sin segmentar su contorno exacto. Las solicitudes de inferencia se atienden una a la vez para proteger la GPU y el estado de Ultralytics. Cambiar de clasificador libera el anterior antes de cargar otro. La app no usa ni modifica el conjunto de prueba.

## Instalación y compilación

La interfaz ya compilada queda en `web/dist`. Para reconstruirla, con Node.js y pnpm instalados:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-web.txt
pnpm --dir web install --frozen-lockfile
pnpm --dir web build
.\.venv\Scripts\python.exe -m cafe_ml.web_api
```

Los pesos y la caché de los backbones deben estar presentes, como en este equipo. Tras instalar y compilar, no se requiere conexión a Internet para analizar fotos. El lockfile de la interfaz registra sus versiones resueltas.

Para desarrollar, ejecuta el servidor Python y, en otra consola, `pnpm --dir web dev`. Vite redirige `/api` al servidor local en el puerto 8765. La app de uso normal se sirve desde FastAPI y no necesita mantener Vite activo.

## Verificación

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -p test_web_api.py -v
```

Estas pruebas cubren carga, formato, tamaño, orientación, conversión de color, ausencia de detecciones y errores de pesos sin cargar modelos reales. La prueba funcional con la foto de ejemplo debe ejecutarse adicionalmente contra los pesos del equipo.

```powershell
.\.venv\Scripts\python.exe scripts\verificar_app_fotos.py --all-models
```

Verificación realizada el 25 de noviembre de 2026: compilación de producción correcta; seis pruebas de API aprobadas; inferencia real correcta para los cinco clasificadores y el ensemble. Todos localizaron los mismos nueve frutos de la foto de demostración, con diferencias de clasificación entre modelos. Con EfficientNet B0 se obtuvieron cinco maduros, tres pintones y uno verde. Esto comprueba el funcionamiento, no mide exactitud con nuevas etiquetas.

También se comprobó en el navegador la selección de un archivo local, el análisis, la selección de una fila, el ejemplo integrado y la descarga efectiva de JPG y JSON. La comprobación HTTP guarda una copia explícita de su resultado de demostración en `pruebas/app_fotos/`; la app no guarda las cargas normales.
