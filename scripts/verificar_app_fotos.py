"""Smoke-test the running local app with real weights and the validation demo."""
import argparse
import base64
import json
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', default='http://127.0.0.1:8765')
    parser.add_argument('--all-models', action='store_true', help='Also run each classifier and the ensemble')
    parser.add_argument('--output', type=Path, default=ROOT / 'pruebas/app_fotos')
    args = parser.parse_args()
    with httpx.Client(base_url=args.url, timeout=180, trust_env=False) as client:
        home = client.get('/')
        home.raise_for_status()
        assert 'id="root"' in home.text
        health = client.get('/api/health')
        health.raise_for_status()
        assert health.json()['status'] == 'ok'
        example = client.get('/api/example')
        example.raise_for_status()
        started = time.monotonic()
        response = client.post('/api/analyze',
                               files={'file': ('ejemplo_validacion.jpg', example.content, 'image/jpeg')},
                               data={'model': 'efficientnet_b0'})
        response.raise_for_status()
        result = response.json()
        default_seconds = round(time.monotonic() - started, 2)
        verified_models = [{'model': result['model'], 'total': result['total'],
                            'counts': result['counts'], 'processing_ms': result['processing_ms']}]
        assert result['total'] > 0, 'Expected coffee fruit in the demo photo'
        assert result['total'] == len(result['detections']) == sum(result['counts'].values())
        for item in result['detections']:
            assert 0 <= item['score'] <= 1
            x1, y1, x2, y2 = item['bbox']
            assert 0 <= x1 < x2 <= result['image']['width']
            assert 0 <= y1 < y2 <= result['image']['height']
        invalid = client.post('/api/analyze', files={'file': ('mal.jpg', b'invalid', 'image/jpeg')})
        assert invalid.status_code == 422
        if args.all_models:
            for model in health.json()['models']:
                if model == 'efficientnet_b0':
                    continue
                other = client.post('/api/analyze',
                                    files={'file': ('ejemplo_validacion.jpg', example.content, 'image/jpeg')},
                                    data={'model': model})
                other.raise_for_status()
                other_result = other.json()
                assert other_result['total'] == result['total']
                assert all(0 <= d['score'] <= 1 for d in other_result['detections'])
                verified_models.append({'model': model, 'total': other_result['total'],
                                        'counts': other_result['counts'], 'processing_ms': other_result['processing_ms']})
                print(json.dumps({'model': model, 'frutos': other_result['total'],
                                  'counts': other_result['counts'], 'processing_ms': other_result['processing_ms']}), flush=True)
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    (output / 'demo_resultado.jpg').write_bytes(base64.b64decode(result['annotated_image'].split(',', 1)[1]))
    result.pop('annotated_image')
    result['image'].pop('data_url')
    result['source'] = 'Foto de validación: demostración funcional, no prueba independiente.'
    result['verification'] = {'deployment': health.json().get('deployment'), 'models': verified_models}
    (output / 'demo_resultado.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'ok': True, 'frutos': result['total'], 'madurez': result['counts'],
                      'segundos': default_seconds, 'modelo': result['model']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
