"""
Tests for FastAPI endpoints and the RealTimeEngine.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from inference.api import create_app
from inference.real_time_engine import RealTimeEngine


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def client():
    """TestClient for the FastAPI app with a default (no-model) engine."""
    app = create_app(engine=RealTimeEngine())
    return TestClient(app)


def _make_txn(**kwargs):
    """Return a minimal valid transaction payload."""
    base = {
        "transaction_id": "TXN00001",
        "amount": 50.0,
        "hour_of_day": 14,
        "day_of_week": 2,
        "is_weekend": 0,
        "is_international": 0,
        "latitude": 40.7,
        "longitude": -74.0,
    }
    base.update(kwargs)
    return base


# ---------------------------------------------------------------------------
# Health endpoint
# ---------------------------------------------------------------------------


class TestHealthEndpoint:
    def test_health_ok(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["engine_ready"] is True

    def test_health_contains_uptime(self, client):
        resp = client.get("/health")
        data = resp.json()
        assert "uptime_s" in data
        assert data["uptime_s"] >= 0.0


# ---------------------------------------------------------------------------
# Model info endpoint
# ---------------------------------------------------------------------------


class TestModelInfoEndpoint:
    def test_model_info_ok(self, client):
        resp = client.get("/model/info")
        assert resp.status_code == 200
        data = resp.json()
        assert "version" in data
        assert "threshold" in data

    def test_model_info_version(self, client):
        resp = client.get("/model/info")
        data = resp.json()
        assert data["version"] == "1.0.0"


# ---------------------------------------------------------------------------
# /predict endpoint
# ---------------------------------------------------------------------------


class TestPredictEndpoint:
    def test_predict_returns_200(self, client):
        resp = client.post("/predict", json=_make_txn())
        assert resp.status_code == 200

    def test_predict_response_fields(self, client):
        resp = client.post("/predict", json=_make_txn())
        data = resp.json()
        assert "transaction_id" in data
        assert "fraud_probability" in data
        assert "is_fraud" in data
        assert "gan_score" in data
        assert "gnn_prob" in data
        assert "threshold" in data
        assert "latency_ms" in data

    def test_predict_probability_range(self, client):
        resp = client.post("/predict", json=_make_txn())
        data = resp.json()
        assert 0.0 <= data["fraud_probability"] <= 1.0

    def test_predict_negative_amount_rejected(self, client):
        resp = client.post("/predict", json=_make_txn(amount=-10.0))
        assert resp.status_code == 422  # Unprocessable Entity

    def test_predict_invalid_hour_rejected(self, client):
        resp = client.post("/predict", json=_make_txn(hour_of_day=25))
        assert resp.status_code == 422

    def test_predict_missing_required_field(self, client):
        txn = _make_txn()
        del txn["amount"]
        resp = client.post("/predict", json=txn)
        assert resp.status_code == 422

    def test_predict_zero_amount(self, client):
        resp = client.post("/predict", json=_make_txn(amount=0.0))
        assert resp.status_code == 200

    def test_predict_large_amount(self, client):
        resp = client.post("/predict", json=_make_txn(amount=99999.99))
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# /batch_predict endpoint
# ---------------------------------------------------------------------------


class TestBatchPredictEndpoint:
    def test_batch_predict_returns_200(self, client):
        payload = {"transactions": [_make_txn(transaction_id=f"T{i}") for i in range(5)]}
        resp = client.post("/batch_predict", json=payload)
        assert resp.status_code == 200

    def test_batch_predict_response_structure(self, client):
        n = 3
        payload = {"transactions": [_make_txn(transaction_id=f"T{i}") for i in range(n)]}
        resp = client.post("/batch_predict", json=payload)
        data = resp.json()
        assert data["n_transactions"] == n
        assert "results" in data
        assert len(data["results"]) == n

    def test_batch_predict_counts_flagged(self, client):
        payload = {"transactions": [_make_txn(transaction_id=f"T{i}") for i in range(4)]}
        resp = client.post("/batch_predict", json=payload)
        data = resp.json()
        assert isinstance(data["n_flagged"], int)
        assert 0 <= data["n_flagged"] <= data["n_transactions"]

    def test_batch_predict_empty_fails(self, client):
        resp = client.post("/batch_predict", json={"transactions": []})
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# RealTimeEngine unit tests
# ---------------------------------------------------------------------------


class TestRealTimeEngine:
    def test_process_transaction_returns_dict(self):
        engine = RealTimeEngine()
        result = engine.process_transaction(_make_txn())
        assert isinstance(result, dict)

    def test_process_transaction_keys(self):
        engine = RealTimeEngine()
        result = engine.process_transaction(_make_txn())
        assert "fraud_probability" in result
        assert "is_fraud" in result

    def test_process_batch_length(self):
        engine = RealTimeEngine()
        txns = [_make_txn(transaction_id=f"T{i}") for i in range(5)]
        results = engine.process_batch(txns)
        assert len(results) == 5

    def test_window_fills(self):
        engine = RealTimeEngine(window_size=3)
        for i in range(5):
            engine.process_transaction(_make_txn(transaction_id=f"T{i}"))
        assert len(engine._window) == 3  # Only last 3 kept

    def test_threshold_applied(self):
        engine = RealTimeEngine(threshold=1.0)  # Nothing will be flagged
        result = engine.process_transaction(_make_txn())
        assert result["is_fraud"] is False

    def test_thread_safety(self):
        import threading

        engine = RealTimeEngine()
        errors = []

        def score():
            try:
                engine.process_transaction(_make_txn())
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=score) for _ in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == []
