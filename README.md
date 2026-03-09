# Credit Card Fraud Detection — GNN + GAN Hybrid Model

A **production-grade** hybrid fraud detection system combining:

- **WGAN-GP** (Wasserstein GAN with Gradient Penalty) for real-time anomaly detection and synthetic data generation
- **GraphSAGE + GAT** (Graph Neural Network) for context-aware, graph-based fraud classification
- **Elastic Weight Consolidation (EWC)** + **Experience Replay** to prevent catastrophic forgetting
- **FastAPI** inference server for real-time scoring

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                    FRAUD DETECTION PIPELINE                          │
│                                                                      │
│  Raw Transactions                                                    │
│       │                                                              │
│       ▼                                                              │
│  ┌──────────┐   ┌─────────────────┐   ┌────────────────────────┐   │
│  │ Cleaner  │──▶│Feature Engineer │──▶│   Graph Builder (PyG)  │   │
│  └──────────┘   └─────────────────┘   └────────────────────────┘   │
│                                                  │                   │
│              ┌───────────────┐                   ▼                  │
│              │  GAN (WGAN-GP)│         ┌──────────────────┐        │
│              │  Generator    │         │  GNN Model       │        │
│              │  Discriminator│◀───────▶│  GraphSAGE + GAT │        │
│              └───────┬───────┘         │  EWC + Replay    │        │
│                      │                 └────────┬─────────┘        │
│                      │ anomaly score            │ fraud prob        │
│                      ▼                          ▼                   │
│              ┌─────────────────────────────────────┐               │
│              │        Ensemble Fusion Layer         │               │
│              │   (weighted / learned / stacking)    │               │
│              └────────────────┬────────────────────┘               │
│                               │ fraud probability                   │
│                               ▼                                     │
│                    ┌──────────────────┐                             │
│                    │   Decision       │  → is_fraud (0/1)           │
│                    └──────────────────┘                             │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Project Structure

```
credit-card-fraud-detection-GNN-GAN/
├── config/
│   └── config.yaml                     # All hyperparameters and paths
├── data/
│   ├── download_data.py                # Multi-source data acquisition
│   ├── synthetic_generator.py          # Realistic synthetic transaction generator
│   └── dataset.py                      # PyTorch Dataset / DataLoader classes
├── preprocessing/
│   ├── cleaner.py                      # Missing values, duplicates, outliers
│   ├── feature_engineering.py          # Time, velocity, aggregation features
│   ├── graph_builder.py                # PyG graph construction
│   └── pipeline.py                     # End-to-end preprocessing orchestrator
├── models/
│   ├── gan/
│   │   ├── generator.py                # WGAN-GP Generator
│   │   ├── discriminator.py            # WGAN-GP Discriminator / Critic
│   │   └── trainer.py                  # GAN training loop + GP
│   ├── gnn/
│   │   ├── layers.py                   # GraphSAGE, GAT, TemporalEncoding layers
│   │   ├── model.py                    # Full GNN model with skip connections
│   │   ├── memory_buffer.py            # Experience replay buffer
│   │   └── trainer.py                  # GNN training with EWC + replay
│   └── ensemble.py                     # Fusion of GAN + GNN scores
├── inference/
│   ├── real_time_engine.py             # Sliding-window streaming inference
│   └── api.py                          # FastAPI REST API
├── evaluation/
│   ├── metrics.py                      # AUPRC, ROC-AUC, F1, MCC, etc.
│   └── visualization.py               # Loss curves, ROC, PR, t-SNE plots
├── utils/
│   ├── logger.py                       # Structured logging
│   └── helpers.py                      # Seed, device, checkpointing
├── tests/
│   ├── test_preprocessing.py
│   ├── test_models.py
│   └── test_inference.py
├── train.py                            # Main training entrypoint
├── requirements.txt
├── setup.py
├── Dockerfile
└── README.md
```

---

## Installation

### Prerequisites

- Python 3.9+
- (Optional) CUDA GPU

### Steps

```bash
# Clone the repository
git clone https://github.com/HardikMahajan-97/credit-card-fraud-detection-GNN-GAN.git
cd credit-card-fraud-detection-GNN-GAN

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# (Optional) Install PyTorch Geometric for graph support
pip install torch-geometric
```

---

## Quick Start

### 1. Train with synthetic data (no external datasets required)

```bash
python train.py --data-source synthetic --gan-epochs 10 --gnn-epochs 5
```

### 2. Start the inference API

```bash
uvicorn inference.api:app --host 0.0.0.0 --port 8000
```

### 3. Score a transaction

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "transaction_id": "TXN12345",
    "amount": 2500.00,
    "hour_of_day": 3,
    "day_of_week": 6,
    "is_weekend": 1,
    "is_international": 1,
    "latitude": 51.5,
    "longitude": -0.1
  }'
```

**Response:**
```json
{
  "transaction_id": "TXN12345",
  "fraud_probability": 0.842,
  "is_fraud": true,
  "gan_score": 0.791,
  "gnn_prob": 0.863,
  "threshold": 0.5,
  "latency_ms": 4.2
}
```

---

## Dataset Preparation

### Option 1: Synthetic (Default — no credentials needed)

```python
from data.download_data import get_data
df = get_data(source="synthetic", n_samples=100_000)
```

### Option 2: Kaggle Credit Card Fraud Dataset

```bash
pip install kagglehub
python -c "from data.download_data import get_data; get_data(source='kaggle')"
```

### Option 3: HuggingFace Hub

```bash
pip install datasets
python -c "from data.download_data import get_data; get_data(source='huggingface')"
```

### Option 4: Mixed (All available sources)

```python
df = get_data(source="mixed")
```

---

## Training

```bash
python train.py \
  --config config/config.yaml \
  --data-source synthetic \
  --device auto \
  --output-dir results
```

---

## API Usage

### Batch Prediction

```bash
curl -X POST http://localhost:8000/batch_predict \
  -H "Content-Type: application/json" \
  -d '{
    "transactions": [
      {"transaction_id": "T1", "amount": 50.0,  "hour_of_day": 12, "day_of_week": 1},
      {"transaction_id": "T2", "amount": 9999.0, "hour_of_day": 3,  "day_of_week": 6, "is_international": 1}
    ]
  }'
```

### Health Check

```bash
curl http://localhost:8000/health
```

---

## Model Architecture Details

### GAN (WGAN-GP)

| Component | Architecture |
|-----------|-------------|
| Generator | Linear → BatchNorm → LeakyReLU (×3) → Tanh |
| Discriminator | Linear → LayerNorm → LeakyReLU → Dropout (×3) → Linear |
| Loss | Wasserstein loss + Gradient Penalty (λ=10) |
| Optimizer | Adam (β₁=0, β₂=0.9, lr=1e-4) |

### GNN (GraphSAGE + GAT)

| Component | Details |
|-----------|---------|
| Graph structure | Bipartite: card nodes ↔ merchant nodes |
| GraphSAGE | 2 layers, mean aggregation, skip connections |
| GAT | 2 layers, 4 attention heads |
| Continual learning | EWC (λ=5000) + Experience Replay (10K buffer) |

---

## Configuration Reference

See `config/config.yaml` for all configurable parameters.

---

## Docker Deployment

```bash
docker build -t fraud-detection .
docker run -p 8000:8000 fraud-detection
```

---

## Running Tests

```bash
pip install pytest
pytest tests/ -v
```

---

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Open a Pull Request

---

## License

MIT License