FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Install package
RUN pip install --no-cache-dir -e .

# Expose API port
EXPOSE 8000

# Run the FastAPI application
CMD ["uvicorn", "inference.api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
