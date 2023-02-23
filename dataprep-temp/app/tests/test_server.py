from fastapi.testclient import TestClient

from qpiai_data_prep.config import sample_input
from app.main import app

client = TestClient(app)


def test_server_start():
    response = client.get("/")
    assert response.status_code == 200


def test_api_process_endpoint():
    response = client.post("/api", json=sample_input)
    assert response.status_code == 202
