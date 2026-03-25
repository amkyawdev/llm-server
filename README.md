# LLM Server

A high-performance LLM inference server with OpenAI-compatible API.

## Features

- **OpenAI-compatible API**: Works with OpenAI client libraries
- **Multiple model support**: Llama 2, Mistral, and more
- **Quantization**: 4-bit and 8-bit quantization support
- **Caching**: Redis and disk cache for fast responses
- **Rate limiting**: Configurable rate limiting
- **Metrics**: Prometheus-compatible metrics
- **Docker support**: Easy deployment with Docker

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Copy environment file
cp .env.example .env

# Edit .env with your settings
```

## Quick Start

```bash
# Start the server
python -m uvicorn src.main:app --host 0.0.0.0 --port 8000

# Or use the start script
./scripts/start_server.sh
```

## API Endpoints

- `GET /health` - Health check
- `POST /api/v1/chat/completions` - Chat completions
- `POST /api/v1/completions` - Text completions
- `POST /api/v1/embeddings` - Create embeddings
- `GET /api/v1/models` - List models

## Docker

```bash
# Build and run with Docker
cd docker
docker-compose up -d
```

## Configuration

See `.env.example` for all configuration options.