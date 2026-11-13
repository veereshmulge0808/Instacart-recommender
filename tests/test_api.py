from fastapi.testclient import TestClient
from app.main import app

def test_health_endpoint():
    """Verify /health returns 200 and loaded model metadata."""
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "optimal_threshold" in data
        assert "model_version" in data

def test_recommendation_valid_user():
    """Verify /recommend returns valid ranked predictions and latency."""
    with TestClient(app) as client:
        payload = {
            "user_id": 1,
            "order_dow": 1,
            "order_hour": 10,
            "days_since_prior": 7.0,
            "top_n": 5
        }
        response = client.post("/recommend", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["user_id"] == 1
        assert "latency_ms" in data
        assert isinstance(data["recommendations"], list)
        if len(data["recommendations"]) > 0:
            item = data["recommendations"][0]
            assert "product_id" in item
            assert "product_name" in item
            assert 0.0 <= item["predicted_probability"] <= 1.0

def test_recommendation_invalid_payload():
    """Verify Pydantic blocks out-of-range inputs with HTTP 422."""
    with TestClient(app) as client:
        bad_payload = {
            "user_id": 1,
            "order_dow": 99,  # Invalid: must be 0-6
            "top_n": 5
        }
        response = client.post("/recommend", json=bad_payload)
        assert response.status_code == 422
