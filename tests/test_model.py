import io
import pytest
import torch
from PIL import Image
from fastapi.testclient import TestClient

from model import get_model
from serve import app

def test_model_architecture():
    model = get_model(architecture="resnet18", num_classes=10, pretrained=False)
    dummy_input = torch.randn(2, 3, 32, 32)
    output = model(dummy_input)
    assert output.shape == (2, 10)

def test_health_endpoint():
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code in (200, 503)

def test_predict_endpoint_invalid_file():
    client = TestClient(app)
    response = client.post(
        "/predict",
        files={"image": ("test.txt", b"dummy text content", "text/plain")}
    )
    assert response.status_code == 400
