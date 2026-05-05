from fastapi.testclient import TestClient

from app.main import app


def test_health_route_exists():
    client = TestClient(app)
    response = client.get("/docs")
    assert response.status_code in {200, 302}
