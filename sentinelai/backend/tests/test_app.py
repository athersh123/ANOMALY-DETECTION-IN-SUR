import io
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

cv2_stub = types.ModuleType("cv2")
for name in [
    "setNumThreads", "imshow", "imwrite", "imread", "waitKey", "destroyAllWindows",
    "cvtColor", "resize", "VideoCapture", "putText", "rectangle", "line",
    "imencode", "getTextSize", "VideoWriter_fourcc", "VideoWriter",
]:
    setattr(cv2_stub, name, lambda *args, **kwargs: None)
cv2_stub.imdecode = lambda *args, **kwargs: None
cv2_stub.IMREAD_COLOR = 1
cv2_stub.IMWRITE_JPEG_QUALITY = 90
cv2_stub.FONT_HERSHEY_SIMPLEX = 0
cv2_stub.FONT_HERSHEY_DUPLEX = 0
cv2_stub.COLOR_RGB2BGR = 0
cv2_stub.CAP_PROP_FRAME_WIDTH = 0
cv2_stub.CAP_PROP_FRAME_HEIGHT = 0
cv2_stub.CAP_PROP_FPS = 0
cv2_stub.CAP_PROP_FRAME_COUNT = 0
cv2_stub.LINE_AA = 0
sys.modules["cv2"] = cv2_stub

ultralytics_stub = types.ModuleType("ultralytics")
ultralytics_stub.YOLO = lambda *args, **kwargs: types.SimpleNamespace(predict=lambda *a, **k: [])
sys.modules["ultralytics"] = ultralytics_stub

tensorflow_stub = types.ModuleType("tensorflow")
class _Logger:
    def setLevel(self, *_args, **_kwargs):
        return None

tensorflow_stub.get_logger = lambda: _Logger()
sys.modules.setdefault("tensorflow", tensorflow_stub)

keras_stub = types.ModuleType("tensorflow.keras")
models_stub = types.ModuleType("tensorflow.keras.models")
models_stub.load_model = lambda *args, **kwargs: types.SimpleNamespace(predict=lambda *a, **k: [None])
sys.modules.setdefault("tensorflow.keras", keras_stub)
sys.modules.setdefault("tensorflow.keras.models", models_stub)

import app as backend_app


class AppUploadTests(unittest.TestCase):
    def test_upload_wildlife_image_initializes_detector_when_missing(self):
        backend_app.wildlife_detector = None
        backend_app.anomaly_detector = None

        fake_detector = types.SimpleNamespace(
            run_on_image=lambda frame: {
                "animal": "Tiger",
                "confidence": 92.4,
                "threat": "Critical",
                "active": True,
                "bbox": [0, 0, 10, 10],
                "timestamp": "2024-01-01 00:00:00",
                "detection_time_ms": 12,
                "annotated_b64": "dGVzdA==",
            },
            start=lambda: None,
        )

        with patch.object(backend_app, "WildlifeDetector", return_value=fake_detector), \
             patch.object(backend_app, "_read_uploaded_image", return_value=None), \
             patch.object(backend_app, "_save_snapshot", return_value="snapshots/test.jpg"), \
             patch.object(backend_app.db, "insert_detection", return_value=None):
            with backend_app.app.test_client() as client:
                response = client.post(
                    "/api/wildlife/upload-image",
                    data={"file": (io.BytesIO(b"fake-image-data"), "test.jpg")},
                    content_type="multipart/form-data",
                )

        self.assertEqual(response.status_code, 422)
        payload = response.get_json()
        self.assertIn("Cannot decode image", payload["error"])


if __name__ == "__main__":
    unittest.main()
