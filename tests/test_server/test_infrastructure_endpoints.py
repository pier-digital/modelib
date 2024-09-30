import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from modelib.server import infrastructure


@pytest.fixture
def client():
    app = FastAPI()
    infrastructure.init_app(app)
    return TestClient(app)


def test_healthz(client):
    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_readyz(client):
    response = client.get("/readyz")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_livez(client):
    response = client.get("/livez")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
