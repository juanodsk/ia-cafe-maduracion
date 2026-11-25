# Cámara para selección de café por madurez

El programa se inicia desde consola y abre una ventana de vídeo con recuadros de colores. Localiza los frutos mediante un detector YOLO11n de una clase y clasifica cada recorte con los modelos de madurez guardados. El clasificador predeterminado es EfficientNet B0 afinado; el ensemble está disponible como opción con mayor coste por fotograma.

## Conectar y ejecutar

Conecta una cámara USB y cierra otras aplicaciones que la estén utilizando. Desde PowerShell:

```powershell
cd "C:\Users\juanp\Desktop\juan-castro"
.\.venv\Scripts\python.exe -m cafe_ml.camera --fuente 0
```

Si la cámara externa es la segunda cámara del equipo:

```powershell
.\.venv\Scripts\python.exe -m cafe_ml.camera --fuente 1
```

No se graba vídeo por defecto. `Q` o `Esc` cierra la ventana; `Ctrl+C` detiene el programa desde consola. La tecla `S` guarda el fotograma con sus recuadros en `pruebas/camara/`.

| Color | Interpretación |
| --- | --- |
| Verde | Verde o inmaduro |
| Amarillo | Pintón o semimaduro |
| Rojo | Maduro |
| Morado | Sobremaduro |
| Marrón | Seco |
| Gris | El clasificador no supera el umbral de puntuación |

Los porcentajes dibujados son puntuaciones softmax del clasificador, no porcentajes de aciertos garantizados. La clasificación depende de que el detector haya localizado correctamente un fruto.

## Opciones

```powershell
# Usar los cinco clasificadores; puede reducir los FPS
.\.venv\Scripts\python.exe -m cafe_ml.camera --fuente 0 --modelo ensemble

# Usar una cámara IP compatible con OpenCV
.\.venv\Scripts\python.exe -m cafe_ml.camera --fuente "rtsp://DIRECCION_DE_TU_CAMARA"

# Procesar un vídeo existente y guardar el resultado sin abrir ventana
.\.venv\Scripts\python.exe -m cafe_ml.camera --fuente "mi_video.mp4" --sin-ventana --grabar "pruebas/camara/resultado.mp4"

# Procesar una fotografía completa y dibujar frutos localizados
.\.venv\Scripts\python.exe -m cafe_ml.camera --imagen "mi_foto.jpg" --salida "pruebas/camara/foto_resultado.jpg"

# Clasificar un único fruto dentro de una zona fija, sin detector automático
.\.venv\Scripts\python.exe -m cafe_ml.camera --fuente 0 --manual
```

`--conf-deteccion 0.3` controla el umbral del localizador y `--conf-madurez 0.55` el del clasificador. `--manual --roi X Y ANCHO ALTO` establece una zona fija en píxeles. `--device cpu` permite usar el programa sin GPU. El modo manual no busca frutos ni demuestra detección automática.

Acerca frutos que ocupen suficientes píxeles, evita desenfoque y reflejos fuertes y mantén iluminación natural. Si no se abre la cámara, cambia el índice, revisa los permisos de cámara de Windows y comprueba que otra aplicación no la tenga ocupada.

## Entrenamiento del localizador

Los cinco clasificadores originales siguen siendo ResNet50, EfficientNet B0, DenseNet121, ViT B 16 y Swin T. YOLO11n es un componente adicional de localización, con una sola clase `fruit`; sus métricas de detección son distintas de las métricas de clasificación de madurez.

`dataset_deteccion/` conserva las mismas 1.651 fotografías de entrenamiento y 354 de validación. El conjunto de prueba permanece reservado. Se agrupan todas las madureces en la clase de localización y se limitan las cajas al borde. Ultralytics elimina cajas idénticas repetidas al cargar las etiquetas; esos conteos se registran por separado. La conversión no cambia las anotaciones ni las fotografías de `dataset_preparado/`.

```powershell
.\.venv\Scripts\python.exe -m cafe_ml.detector prepare
.\.venv\Scripts\python.exe -m cafe_ml.detector download
.\.venv\Scripts\python.exe -m cafe_ml.detector train --epochs 15
# Reanudar una ejecución interrumpida
.\.venv\Scripts\python.exe -m cafe_ml.detector train --resume
```

El checkpoint seleccionado se guarda en `experimentos/detector_01/weights/best.pt`. El piloto conserva el backbone del detector congelado y ajusta sus capas restantes, con resolución 640, semilla 42 y sin cambios de tono, saturación ni exposición. No utiliza sintéticos de la GAN. Las pruebas con fotos o vídeos del dataset verifican la integración; el comportamiento con una cámara física necesita comprobarse en sus condiciones de captura.

Dependencias adicionales: `requirements-camera.txt`. Referencias: [entrenamiento de Ultralytics](https://docs.ultralytics.com/modes/train/), [predicción de Ultralytics](https://docs.ultralytics.com/modes/predict/) y [VideoCapture de OpenCV](https://docs.opencv.org/4.x/d8/dfe/classcv_1_1VideoCapture.html).

## Verificación de esta entrega

El detector completó 15 épocas. En validación obtuvo precisión de localización 0,9069, recall 0,9325, mAP50 0,9502 y mAP50 a 95 0,7844. Estas métricas corresponden a localizar frutos, no al rendimiento conjunto de localizar y clasificar la madurez en una cámara nueva.

Se conservaron 23.641 cajas únicas de entrenamiento y 5.007 de validación, tras eliminar 237 y 109 cajas de localización repetidas. Las fotografías y anotaciones de clasificación originales permanecen intactas.

Pasaron cuatro pruebas de coordenadas, conversión RGB, recuadro manual, rechazo de baja puntuación y ausencia de detecciones. La prueba integrada encontró nueve frutos en la imagen demostrativa de validación y produjo cinco fotogramas anotados en el vídeo de salida. La cámara física de índice 0 entregó fotogramas y se abrió la ventana en vivo; no se verificó todavía una muestra física positiva de café frente a esa cámara.

- [Imagen con recuadros de ejemplo](../pruebas/camara/demo_auto.jpg).
- [Vídeo de ejemplo](../pruebas/camara/demo_salida.mp4).
- [Comprobación guardada](../experimentos/detector_01/verificacion_camara.json).
- [Métricas del localizador](../experimentos/detector_01/resumen.json).
