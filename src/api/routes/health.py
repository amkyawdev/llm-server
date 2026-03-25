"""Health API route - Handles health check and status endpoints."""

from typing import Dict, Any
import time
import platform
from datetime import datetime
import torch

from fastapi import APIRouter
from pydantic import BaseModel, Field

from src.core.gpu_manager import get_gpu_monitor
from src.core.cache_manager import get_cache_manager

router = APIRouter(prefix="/health", tags=["health"])


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = Field(..., description="Health status")
    timestamp: int = Field(..., description="Unix timestamp")
    version: str = Field("0.1.0", description="Server version")


@router.get("", response_model=HealthResponse)
async def health_check():
    """Basic health check endpoint."""
    return HealthResponse(
        status="healthy",
        timestamp=int(time.time()),
        version="0.1.0",
    )


@router.get("/live")
async def liveness():
    """Liveness probe for Kubernetes."""
    return {"status": "alive"}


@router.get("/ready")
async def readiness():
    """Readiness probe - checks if server is ready to accept requests."""
    gpu_monitor = get_gpu_monitor()
    
    ready = True
    reason = None
    
    # Check if GPU is available (if enabled)
    if not gpu_monitor.is_available:
        ready = True  # CPU mode is always ready
        reason = "running on CPU"
    
    return {
        "ready": ready,
        "reason": reason,
    }


@router.get("/status")
async def detailed_status():
    """Detailed status information."""
    gpu_monitor = get_gpu_monitor()
    cache_manager = get_cache_manager()
    
    # Get GPU info
    gpu_info = []
    if gpu_monitor.is_available:
        for info in gpu_monitor.get_all_gpu_info():
            gpu_info.append({
                "device_id": info.device_id,
                "name": info.name,
                "memory_total_mb": info.memory_total // (1024 * 1024),
                "memory_allocated_mb": info.memory_allocated // (1024 * 1024),
                "memory_free_mb": (info.memory_total - info.memory_reserved) // (1024 * 1024),
                "utilization_percent": info.utilization,
            })
    
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "platform": {
            "python_version": platform.python_version(),
            "pytorch_version": torch.__version__,
            "system": platform.system(),
        },
        "gpu": {
            "available": gpu_monitor.is_available,
            "device_count": gpu_monitor.device_count,
            "devices": gpu_info,
        },
        "cache": cache_manager.stats(),
    }


@router.get("/metrics")
async def metrics():
    """Prometheus-compatible metrics endpoint."""
    gpu_monitor = get_gpu_monitor()
    
    # Build metrics in Prometheus format
    metrics_lines = []
    
    # Python metrics
    metrics_lines.append('# HELP python_info Python version info')
    metrics_lines.append('# TYPE python_info gauge')
    metrics_lines.append(f'python_info{{version="{platform.python_version()}"}} 1')
    
    # GPU metrics
    if gpu_monitor.is_available:
        metrics_lines.append('# HELP gpu_available Number of available GPUs')
        metrics_lines.append('# TYPE gpu_available gauge')
        metrics_lines.append(f'gpu_available {gpu_monitor.device_count}')
        
        for info in gpu_monitor.get_all_gpu_info():
            metrics_lines.append('# HELP gpu_memory_used_bytes GPU memory used in bytes')
            metrics_lines.append('# TYPE gpu_memory_used_bytes gauge')
            metrics_lines.append(f'gpu_memory_used_bytes{{device="{info.device_id}"}} {info.memory_allocated}')
    
    return "\n".join(metrics_lines)


@router.post("/reload")
async def reload_config():
    """Reload configuration (admin only in production)."""
    # This would reload settings from config files
    return {"status": "reloaded"}


@router.get("/debug/memory")
async def debug_memory():
    """Debug memory usage."""
    import psutil
    
    process = psutil.Process()
    mem_info = process.memory_info()
    
    return {
        "rss_mb": mem_info.rss / (1024 * 1024),
        "vms_mb": mem_info.vms / (1024 * 1024),
    }