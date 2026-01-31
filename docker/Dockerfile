FROM python:3.9-slim

LABEL maintainer="Tim Fatykhov <tfatykhov@gmail.com>"
LABEL description="Membrain - Neuromorphic Memory Bridge for LLM Agents"

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    LOIHI_BACKEND=SIM \
    MEMBRAIN_PORT=50051

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY pyproject.toml README.md LICENSE ./
COPY src/ ./src/
COPY protos/ ./protos/

# Install Python dependencies
RUN pip install --no-cache-dir -e .

# Generate protobuf stubs
RUN python -m grpc_tools.protoc \
    -I./protos \
    --python_out=./src/membrain/proto \
    --grpc_python_out=./src/membrain/proto \
    ./protos/memory_a2a.proto

# Expose gRPC port
EXPOSE 50051

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import grpc; channel = grpc.insecure_channel('localhost:50051'); channel.close()" || exit 1

# Run the server
CMD ["python", "-m", "membrain.server"]
