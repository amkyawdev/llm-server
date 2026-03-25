"""
LLM Server - FastAPI application for serving LLM models.

Main entry point for the LLM inference server.
"""

import time
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from starlette.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse

from loguru import logger

from config import settings
from src.api.routes import (
    chat_router,
    embeddings_router,
    models_router,
    health_router,
    admin_router,
)
from src.api.middleware import setup_cors
from src.utils.logger import setup_logger
from src.utils.metrics import get_metrics_collector





@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    logger.info("Starting LLM Server...")
    logger.info(f"Server config: host={settings.host}, port={settings.port}")
    logger.info(f"Model: {settings.model_name}, device: {settings.model_device}")
    
    # Initialize metrics
    metrics = get_metrics_collector()
    metrics.gauge("server_start_time", time.time())
    
    yield
    
    # Cleanup
    logger.info("Shutting down LLM Server...")


# Create FastAPI application
app = FastAPI(
    title="LLM Server",
    description="High-performance LLM inference server with OpenAI-compatible API",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan,
)

# Setup CORS
setup_cors(app)

# Add Gzip compression
app.add_middleware(GZipMiddleware, minimum_size=1000)


# Request metrics middleware
@app.middleware("http")
async def add_metrics_middleware(request: Request, call_next):
    """Add request metrics."""
    start_time = time.time()
    
    # Get metrics
    metrics = get_metrics_collector()
    
    # Record request
    request_id = request.headers.get("X-Request-ID", f"req-{int(time.time())}")
    metrics.record_request(
        request_id=request_id,
        endpoint=request.url.path,
        method=request.method,
    )
    
    # Process request
    response = await call_next(request)
    
    # Record response
    metrics.record_response(
        request_id=request_id,
        status_code=response.status_code,
    )
    
    # Add timing header
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    
    return response


# Exception handlers
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler."""
    logger.error(f"Unhandled exception: {exc}")
    
    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "message": "Internal server error",
                "type": "internal_error",
                "code": 500,
            }
        },
    )


# Include routers
app.include_router(health_router)
app.include_router(chat_router, prefix="/api/v1")
app.include_router(embeddings_router, prefix="/api/v1")
app.include_router(models_router, prefix="/api/v1")
app.include_router(admin_router, prefix="/api/v1")


# Root endpoint
@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "name": "LLM Server",
        "version": "0.1.0",
        "docs": "/docs",
        "health": "/health",
    }


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "src.main:app",
        host=settings.host,
        port=settings.port,
        reload=False,
        log_level=settings.log_level.lower(),
        workers=1,  # Use 1 for development
    )