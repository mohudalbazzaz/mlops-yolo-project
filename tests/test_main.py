import numpy as np
from fastapi.testclient import TestClient
from src.main import app

client = TestClient(app)

def test_banana_ripeness_classifier(monkeypatch):

    def fake_preprocess_image(_):
        return np.random.rand(128, 128, 3)

    monkeypatch.setattr("src.main.preprocess_image", fake_preprocess_image)

    class DummyModel:
        def predict(self, _):
            return np.array([[0.0, 1.0, 0.0]])

    dummy_model = DummyModel()
    monkeypatch.setattr("src.main.model", dummy_model)

    def fake_compute(label):
        return f"{label}: Expires in 3 days"

    monkeypatch.setattr("src.main.compute_cumulative_ripening", fake_compute)

    response = client.post(
        "/banana_ripeness_classifier",
        files={"file": ("test.jpg", b"fake_bytes", "image/jpeg")}
    )

    assert response.status_code == 200
    assert response.json() == "Ripe: Expires in 3 days"