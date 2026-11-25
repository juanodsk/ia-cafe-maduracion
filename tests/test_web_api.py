import io
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from fastapi.testclient import TestClient
from PIL import Image

from cafe_ml import web_api


def photo_bytes(mode='RGB', size=(100, 80), exif=None):
    output = io.BytesIO()
    image = Image.new(mode, size, 'red' if mode == 'RGB' else 128)
    image.save(output, format='JPEG', **({'exif': exif} if exif else {}))
    return output.getvalue()


class WebApiTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(web_api.app)

    def upload(self, data, **fields):
        return self.client.post('/api/analyze', files={'file': ('foto.jpg', data, 'image/jpeg')}, data=fields)

    def test_upload_rgb_inference_response_and_download(self):
        def process(frame):
            self.assertEqual(frame.shape, (80, 100, 3))
            self.assertGreater(int(frame[0, 0, 2]), 240)  # RGB upload -> BGR pipeline.
            return [{'bbox': [10, 10, 50, 60], 'class': 'ripe', 'candidate_class': 'ripe',
                     'score': .87, 'detector_score': .93}]
        with patch.object(web_api, 'get_pipeline', return_value=SimpleNamespace(process=process)):
            response = self.upload(photo_bytes())
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data['counts'], {'ripe': 1})
        self.assertEqual(data['detections'][0]['label'], 'Maduro')
        self.assertEqual(data['detections'][0]['score'], .87)
        self.assertEqual(data['image']['width'], 100)
        self.assertTrue(data['annotated_image'].startswith('data:image/jpeg;base64,'))

    def test_invalid_empty_and_unknown_model_do_not_run_inference(self):
        with patch.object(web_api, 'get_pipeline') as load:
            self.assertEqual(self.upload(b'').status_code, 422)
            self.assertEqual(self.upload(b'not an image').status_code, 422)
            self.assertEqual(self.upload(photo_bytes(), model='inventado').status_code, 422)
            load.assert_not_called()

    def test_oversize_is_rejected_before_inference(self):
        with patch.object(web_api, 'MAX_BYTES', 100):
            self.assertEqual(self.upload(b'x' * 101).status_code, 413)
        with patch.object(web_api, 'MAX_PIXELS', 100):
            self.assertEqual(self.upload(photo_bytes()).status_code, 413)

    def test_phone_orientation_grayscale_and_max_dimension(self):
        exif = Image.Exif(); exif[274] = 6
        image = web_api.decode_photo(photo_bytes(exif=exif))
        self.assertEqual(image.size, (80, 100))
        self.assertEqual(web_api.decode_photo(photo_bytes(mode='L')).mode, 'RGB')
        self.assertEqual(web_api.decode_photo(photo_bytes(size=(2400, 1200))).size, (1920, 960))

    def test_no_detections_and_missing_checkpoint(self):
        with patch.object(web_api, 'get_pipeline', return_value=SimpleNamespace(process=lambda _: [])):
            data = self.upload(photo_bytes()).json()
            self.assertEqual(data['total'], 0)
            self.assertEqual(data['counts'], {})
        with patch.object(web_api, 'get_pipeline', side_effect=FileNotFoundError('expected test')):
            with self.assertLogs(web_api.logger, level='ERROR'):
                self.assertEqual(self.upload(photo_bytes()).status_code, 503)

    def test_health_and_validation_example(self):
        self.assertEqual(self.client.get('/api/health').json()['status'], 'ok')
        self.assertEqual(self.client.get('/api/example').headers['content-type'], 'image/jpeg')

    def test_health_identifies_remote_deployment_for_photo_notice(self):
        with patch.dict(web_api.os.environ, {'CAFE_DEPLOYMENT': 'server'}):
            self.assertEqual(self.client.get('/api/health').json()['deployment'], 'server')
        with patch.dict(web_api.os.environ, {'CAFE_DEPLOYMENT': 'local'}):
            self.assertEqual(self.client.get('/api/health').json()['deployment'], 'local')


if __name__ == '__main__':
    unittest.main()
