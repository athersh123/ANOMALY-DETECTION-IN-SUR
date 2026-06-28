import sys
import types
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
cv2.VideoCapture = lambda *args, **kwargs: types.SimpleNamespace(
    isOpened=lambda: False,
    release=lambda: None,
    read=lambda: (False, None)
)

# Real tensorflow is imported natively

from anomaly_detector import classify_anomaly_type


class AnomalyDetectorClassificationTests(unittest.TestCase):
    def test_low_error_is_normal(self):
        self.assertEqual(classify_anomaly_type(0.0005), "Normal")

    def test_moderate_error_is_generic_suspicious(self):
        self.assertEqual(classify_anomaly_type(0.0019), "Suspicious Activity")

    def test_high_error_is_generic_anomaly(self):
        self.assertEqual(classify_anomaly_type(0.0035), "Anomaly Detected")


if __name__ == "__main__":
    unittest.main()
