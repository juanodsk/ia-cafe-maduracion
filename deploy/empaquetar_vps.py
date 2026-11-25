"""Create a portable Docker build ZIP including only inference files and weights."""
from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path

from verify_models import ROOT, ASSETS, verify


def source_files():
    fixed = [
        'Dockerfile', '.dockerignore', 'compose.yaml', '.env.example', 'requirements-docker.txt',
        'deploy/verify_models.py', 'deploy/model_assets.json', 'deploy/empaquetar_vps.py',
        'deploy/nginx.conf.example', 'docs/docker_vps.md',
        'web/package.json', 'web/pnpm-lock.yaml', 'web/vite.config.js', 'web/index.html',
        'pruebas/camara/demo_validacion.jpg', 'scripts/verificar_app_fotos.py',
        'pruebas/docker_app/demo_resultado.json', 'pruebas/docker_app/demo_resultado.jpg',
        'tests/test_web_api.py',
        *ASSETS,
    ]
    files = [ROOT / name for name in fixed]
    files.extend(sorted((ROOT / 'cafe_ml').glob('*.py')))
    files.extend(sorted(path for path in (ROOT / 'web/src').rglob('*') if path.is_file()))
    return files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'dist_deploy/cafe-madurez-vps.zip')
    args = parser.parse_args()
    verify()
    files = source_files()
    for path in files:
        if not path.is_file():
            raise FileNotFoundError(path)
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f'El archivo ya existe: {output}. Elige otra ruta con --output.')
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix('.zip.tmp')
    try:
        with zipfile.ZipFile(temporary, mode='w', compression=zipfile.ZIP_DEFLATED, compresslevel=1) as archive:
            for path in files:
                archive.write(path, 'cafe-madurez/' + path.relative_to(ROOT).as_posix())
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    with output.open('rb') as handle:
        digest = hashlib.file_digest(handle, 'sha256').hexdigest()
    report = {'archive': output.name, 'bytes': output.stat().st_size, 'sha256': digest,
              'file_count': len(files), 'model_assets': len(ASSETS)}
    output.with_suffix('.zip.sha256').write_text(f'{digest}  {output.name}\n', encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
