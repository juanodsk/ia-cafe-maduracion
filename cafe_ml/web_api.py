"""Local photo app: serve built React assets and reuse the camera inference pipeline."""
from __future__ import annotations

import argparse
import base64
import gc
import io
import logging
import os
import threading
import time
import warnings
from collections import Counter
from typing import Literal

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps, UnidentifiedImageError
from starlette.concurrency import run_in_threadpool

from .camera import CameraPipeline, COLORS, LABELS
from .experiment import ROOT, torch

MAX_BYTES = 15 * 1024 * 1024
MAX_PIXELS = 20_000_000
MODEL_LABELS = {
    'efficientnet_b0': 'EfficientNet B0 · ajuste fino',
    'resnet50': 'ResNet50', 'densenet121': 'DenseNet121',
    'vit_b_16': 'ViT B/16', 'swin_t': 'Swin T',
    'ensemble': 'Ensemble · cinco modelos',
}
WEB_LABELS = {**LABELS, 'semi_ripe': 'Pintón', 'unknown': 'Incierto'}
ModelName = Literal['efficientnet_b0', 'resnet50', 'densenet121', 'vit_b_16', 'swin_t', 'ensemble']
app = FastAPI(title='Café · Madurez', docs_url=None, redoc_url=None)
_lock = threading.Lock()
_analysis_slot = threading.BoundedSemaphore(1)
_pipeline = None
_loaded_model = None
logger = logging.getLogger(__name__)


def decode_photo(payload: bytes) -> Image.Image:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('error', Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(payload)) as source:
                if source.format not in ('JPEG', 'PNG', 'WEBP'):
                    raise HTTPException(415, 'Usa una imagen JPG, PNG o WEBP.')
                if source.width * source.height > MAX_PIXELS:
                    raise HTTPException(413, 'La foto supera 20 megapíxeles. Reduce su tamaño e intenta de nuevo.')
                # Transpose before discarding EXIF so portrait phone photos retain orientation.
                image = ImageOps.exif_transpose(source).convert('RGB')
    except HTTPException:
        raise
    except (Image.DecompressionBombWarning, Image.DecompressionBombError):
        raise HTTPException(413, 'La foto es demasiado grande. El máximo es 20 megapíxeles.') from None
    except (UnidentifiedImageError, OSError, ValueError):
        raise HTTPException(422, 'No se pudo leer la foto. Prueba con otro archivo JPG, PNG o WEBP.') from None
    image.thumbnail((1920, 1920), Image.Resampling.LANCZOS)
    return image


def image_url(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format='JPEG', quality=92)
    return 'data:image/jpeg;base64,' + base64.b64encode(buffer.getvalue()).decode('ascii')


def get_pipeline(model_name: str):
    """Call under _lock: both loading and Ultralytics prediction are serialized."""
    global _pipeline, _loaded_model
    if _loaded_model != model_name or _pipeline is None:
        _pipeline = None
        _loaded_model = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        torch.set_num_threads(max(1,int(os.environ.get('CAFE_NUM_THREADS','2'))))
        _pipeline = CameraPipeline(
            ROOT / 'experimentos/detector_01/weights/best.pt', model_name,
            'cuda' if torch.cuda.is_available() else 'cpu',
            classifier_batch_size=int(os.environ.get('CAFE_INFERENCE_BATCH_SIZE','4')),
        )
        _loaded_model = model_name
    return _pipeline


def analyze_photo(image: Image.Image, model_name: str) -> dict:
    started = time.perf_counter()
    frame = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
    with _lock:
        annotations = get_pipeline(model_name).process(frame)
    rendered = frame.copy()
    detections = []
    for index, ann in enumerate(annotations, 1):
        label = WEB_LABELS[ann['class']]
        item = {**ann, 'id': index, 'label': label,
                'candidate_label': WEB_LABELS[ann['candidate_class']]}
        detections.append(item)
        x1, y1, x2, y2 = ann['bbox']
        color = COLORS[ann['class']]
        cv2.rectangle(rendered, (x1, y1), (x2, y2), color, 2)
        caption = f"{index}. {LABELS[ann['class']]} {ann['score'] * 100:.1f}%"
        font_scale = max(.45, min(.8, image.width / 1300))
        (text_w, text_h), baseline = cv2.getTextSize(caption, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
        text_x = max(0, min(x1, image.width - text_w - 8))
        text_y = min(image.height - baseline - 4, max(text_h + 8, y1))
        cv2.rectangle(rendered, (text_x, text_y - text_h - 6),
                      (min(image.width, text_x + text_w + 8), text_y + baseline + 3), (20, 30, 25), -1)
        cv2.putText(rendered, caption, (text_x + 4, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1, cv2.LINE_AA)
    counts = Counter(d['class'] for d in detections)
    return {
        'image': {'data_url': image_url(image), 'width': image.width, 'height': image.height},
        'annotated_image': image_url(Image.fromarray(cv2.cvtColor(rendered, cv2.COLOR_BGR2RGB))),
        'detections': detections, 'counts': dict(counts), 'total': len(detections),
        'model': model_name, 'model_label': MODEL_LABELS[model_name],
        'device': 'GPU' if torch.cuda.is_available() else 'CPU',
        'processing_ms': round((time.perf_counter() - started) * 1000),
        'maturity_threshold': .55,
        'score_note': 'La puntuación del modelo no es una certeza calibrada ni la precisión del sistema. '
                      'Por debajo del 55 % se muestra Incierto.',
    }


@app.get('/api/health')
def health():
    return {'status': 'ok', 'loaded_model': _loaded_model, 'models': MODEL_LABELS,
            'deployment': 'server' if os.environ.get('CAFE_DEPLOYMENT') == 'server' else 'local',
            'max_bytes': MAX_BYTES, 'max_pixels': MAX_PIXELS}


@app.get('/api/example')
def example():
    path = ROOT / 'pruebas/camara/demo_validacion.jpg'
    if not path.is_file():
        raise HTTPException(404, 'La foto de ejemplo no está disponible.')
    return FileResponse(path, media_type='image/jpeg', filename='ejemplo_validacion.jpg')


@app.post('/api/analyze')
async def analyze(file: UploadFile = File(...), model: ModelName = Form('efficientnet_b0')):
    if not _analysis_slot.acquire(blocking=False):
        await file.close()
        raise HTTPException(429, 'Ya hay un análisis en curso. Espera a que termine y vuelve a intentarlo.',
                            headers={'Retry-After': '5'})
    try:
        try:
            payload = await file.read(MAX_BYTES + 1)
        finally:
            await file.close()
        if len(payload) > MAX_BYTES:
            raise HTTPException(413, 'El archivo supera 15 MB. Reduce su tamaño e intenta de nuevo.')
        if not payload:
            raise HTTPException(422, 'Selecciona una foto que no esté vacía.')
        image = await run_in_threadpool(decode_photo, payload)
        try:
            return await run_in_threadpool(analyze_photo, image, model)
        except FileNotFoundError:
            logger.exception('Missing model checkpoint')
            raise HTTPException(503, 'Faltan pesos del modelo. Revisa la consola del servidor.') from None
        except Exception:
            logger.exception('Photo inference failed')
            raise HTTPException(503, 'No se pudo completar el análisis. Revisa la consola e intenta de nuevo.') from None
    finally:
        _analysis_slot.release()


dist = ROOT / 'web/dist'
if dist.is_dir():
    app.mount('/', StaticFiles(directory=dist, html=True), name='frontend')
else:
    @app.get('/')
    def missing_frontend():
        raise HTTPException(503, 'Primero compila la interfaz: pnpm --dir web build. Consulta docs/app_fotos.md.')


def main():
    import uvicorn
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=8765)
    args = parser.parse_args()
    uvicorn.run(app, host='127.0.0.1', port=args.port)


if __name__ == '__main__':
    main()
