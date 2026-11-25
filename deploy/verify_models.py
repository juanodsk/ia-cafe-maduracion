"""Validate the exact inference assets; stdlib only, also run during image build."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ASSETS = (
    'experimentos/detector_01/weights/best.pt',
    'experimentos/finetune_01/efficientnet_b0/checkpoint.pt',
    'experimentos/finetune_01/swin_t/checkpoint.pt',
    'experimentos/transfer_01/resnet50/checkpoint.pt',
    'experimentos/transfer_01/densenet121/checkpoint.pt',
    'experimentos/transfer_01/vit_b_16/checkpoint.pt',
    '.cache/torch/hub/checkpoints/resnet50-11ad3fa6.pth',
    '.cache/torch/hub/checkpoints/densenet121-a639ec97.pth',
    '.cache/torch/hub/checkpoints/vit_b_16-c867db91.pth',
)


def inspect(root=ROOT):
    records = []
    for relative in ASSETS:
        path = root / relative
        with path.open('rb') as handle:
            digest = hashlib.file_digest(handle, 'sha256').hexdigest()
        records.append({'path': relative, 'bytes': path.stat().st_size, 'sha256': digest})
    return records


def verify(root=ROOT):
    expected = json.loads((root / 'deploy/model_assets.json').read_text(encoding='utf-8'))
    actual = inspect(root)
    if actual != expected['files']:
        raise RuntimeError('Los pesos no coinciden con deploy/model_assets.json. No se pueden garantizar las predicciones originales.')
    return actual


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--write-manifest', action='store_true', help='Explicitly record the current trained inference weights')
    args = parser.parse_args()
    if args.write_manifest:
        target = ROOT / 'deploy/model_assets.json'
        target.write_text(json.dumps({'schema_version': 1, 'files': inspect()}, indent=2) + '\n', encoding='utf-8')
    records = verify()
    print(f"OK: {len(records)} archivos de modelos; {sum(r['bytes'] for r in records) / 1024**2:.1f} MiB")
