"""
FastAPI application for credit card fraud detection inference.

Endpoints:
  POST /predict          — Single transaction fraud scoring
  POST /batch_predict    — Batch transaction scoring
  GET  /health           — Health check
  GET  /model/info       — Model metadata and version
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

from inference.real_time_engine import RealTimeEngine
from utils.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------


class TransactionRequest(BaseModel):
    """Single transaction feature payload."""

    transaction_id: str = Field(default="", description="Unique transaction identifier")
    amount: float = Field(..., ge=0.0, description="Transaction amount in USD")
    hour_of_day: int = Field(..., ge=0, le=23, description="Hour of the day (0–23)")
    day_of_week: int = Field(..., ge=0, le=6, description="Day of the week (0=Mon, 6=Sun)")
    is_weekend: int = Field(default=0, ge=0, le=1)
    is_international: int = Field(default=0, ge=0, le=1)
    latitude: float = Field(default=0.0, ge=-90.0, le=90.0)
    longitude: float = Field(default=0.0, ge=-180.0, le=180.0)
    timestamp: Optional[str] = Field(default=None)
    card_id: Optional[str] = Field(default=None)
    merchant_id: Optional[str] = Field(default=None)
    merchant_category: Optional[str] = Field(default=None)
    card_type: Optional[str] = Field(default=None)

    @field_validator("amount")
    @classmethod
    def amount_must_be_positive(cls, v: float) -> float:
        if v < 0:
            raise ValueError("amount must be non-negative")
        return v


class FraudPrediction(BaseModel):
    """Fraud scoring result for a single transaction."""

    transaction_id: str
    fraud_probability: float
    is_fraud: bool
    gan_score: float
    gnn_prob: float
    threshold: float
    latency_ms: float


class BatchTransactionRequest(BaseModel):
    """Batch of transactions for scoring."""

    transactions: List[TransactionRequest] = Field(..., min_length=1, max_length=1000)


class BatchPredictionResponse(BaseModel):
    """Response for batch scoring."""

    results: List[FraudPrediction]
    n_transactions: int
    n_flagged: int
    latency_ms: float


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    uptime_s: float
    engine_ready: bool


class ModelInfoResponse(BaseModel):
    """Model metadata response."""

    version: str
    fusion_method: str
    threshold: float
    engine_window_size: int


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

_START_TIME = time.time()


def create_app(engine: Optional[RealTimeEngine] = None) -> FastAPI:
    """
    Create and configure the FastAPI application.

    Args:
        engine: Optional pre-initialized RealTimeEngine. If None, a default
                engine with no models is used (useful for testing).

    Returns:
        Configured FastAPI application instance.
    """
    app = FastAPI(
        title="Credit Card Fraud Detection API",
        description="Real-time fraud scoring using GAN + GNN hybrid model.",
        version="1.0.0",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Use a lightweight default engine if none provided
    _engine = engine or RealTimeEngine()

    # ---- Routes ----

    @app.get("/health", response_model=HealthResponse, tags=["System"])
    async def health() -> HealthResponse:
        """Return service health status."""
        return HealthResponse(
            status="ok",
            uptime_s=round(time.time() - _START_TIME, 2),
            engine_ready=True,
        )

    @app.get("/model/info", response_model=ModelInfoResponse, tags=["System"])
    async def model_info() -> ModelInfoResponse:
        """Return model metadata."""
        fusion = "none"
        if _engine.ensemble is not None:
            fusion = getattr(_engine.ensemble, "fusion_method", "unknown")
        return ModelInfoResponse(
            version="1.0.0",
            fusion_method=fusion,
            threshold=_engine.threshold,
            engine_window_size=_engine.window_size,
        )

    @app.post("/predict", response_model=FraudPrediction, tags=["Inference"])
    async def predict(request: TransactionRequest) -> FraudPrediction:
        """
        Score a single transaction for fraud.

        Returns the fraud probability, decision, and component scores.
        """
        t0 = time.perf_counter()
        txn = request.model_dump()

        try:
            result = _engine.process_transaction(txn)
        except Exception as exc:
            logger.error(f"Prediction failed: {exc}")
            raise HTTPException(status_code=500, detail=f"Prediction error: {str(exc)}")

        latency = round((time.perf_counter() - t0) * 1000, 2)
        return FraudPrediction(latency_ms=latency, **result)

    @app.post("/batch_predict", response_model=BatchPredictionResponse, tags=["Inference"])
    async def batch_predict(request: BatchTransactionRequest) -> BatchPredictionResponse:
        """Score a batch of transactions."""
        t0 = time.perf_counter()
        txns = [t.model_dump() for t in request.transactions]

        try:
            results_raw = _engine.process_batch(txns)
        except Exception as exc:
            logger.error(f"Batch prediction failed: {exc}")
            raise HTTPException(status_code=500, detail=f"Batch prediction error: {str(exc)}")

        latency = round((time.perf_counter() - t0) * 1000, 2)
        predictions = [FraudPrediction(latency_ms=latency, **r) for r in results_raw]
        n_flagged = sum(1 for p in predictions if p.is_fraud)

        return BatchPredictionResponse(
            results=predictions,
            n_transactions=len(predictions),
            n_flagged=n_flagged,
            latency_ms=latency,
        )

    return app


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

app = create_app()

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("inference.api:app", host="0.0.0.0", port=8000, reload=False)
