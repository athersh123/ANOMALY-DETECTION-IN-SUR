import unittest
from pathlib import Path

from model_utils import resolve_model_path


class ModelPathResolutionTests(unittest.TestCase):
    def test_resolves_workspace_relative_wildlife_weights(self):
        resolved = resolve_model_path("weights/best.pt")
        self.assertTrue(Path(resolved).exists())

    def test_resolves_workspace_relative_anomaly_model(self):
        resolved = resolve_model_path("ana_backend/best_model.keras", "ana_backend/final_model.keras")
        self.assertTrue(Path(resolved).exists())


if __name__ == "__main__":
    unittest.main()
