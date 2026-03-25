"""GPU Manager - Monitors and manages GPU resources."""

from typing import Optional, Dict, List, Any
from dataclasses import dataclass
import os

import torch
from loguru import logger


@dataclass
class GPUInfo:
    """GPU information container."""

    device_id: int
    name: str
    memory_total: int
    memory_allocated: int
    memory_reserved: int
    utilization: float
    temperature: Optional[float] = None


class GPUMonitor:
    """Monitors and manages GPU resources."""

    def __init__(self):
        self._available = torch.cuda.is_available()
        self._device_count = torch.cuda.device_count() if self._available else 0
        logger.info(
            f"GPUMonitor initialized: {self._device_count} GPUs available"
        )

    @property
    def is_available(self) -> bool:
        """Check if GPU is available."""
        return self._available

    @property
    def device_count(self) -> int:
        """Get the number of available GPUs."""
        return self._device_count

    def get_device(self, device_id: int = 0) -> torch.device:
        """Get torch device."""
        if self._available and device_id < self._device_count:
            return torch.device(f"cuda:{device_id}")
        return torch.device("cpu")

    def get_gpu_info(self, device_id: int = 0) -> GPUInfo:
        """Get information about a specific GPU.

        Args:
            device_id: GPU device ID

        Returns:
            GPU information
        """
        if not self._available:
            return GPUInfo(
                device_id=0,
                name="CPU",
                memory_total=0,
                memory_allocated=0,
                memory_reserved=0,
                utilization=0.0,
            )

        props = torch.cuda.get_device_properties(device_id)
        memory_total = props.total_memory
        memory_allocated = torch.cuda.memory_allocated(device_id)
        memory_reserved = torch.cuda.memory_reserved(device_id)

        # Try to get utilization (may not be available on all systems)
        utilization = 0.0
        try:
            import pynvml

            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)
            utilization = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
            pynvml.nvmlShutdown()
        except ImportError:
            pass

        return GPUInfo(
            device_id=device_id,
            name=props.name,
            memory_total=memory_total,
            memory_allocated=memory_allocated,
            memory_reserved=memory_reserved,
            utilization=utilization,
        )

    def get_all_gpu_info(self) -> List[GPUInfo]:
        """Get information about all GPUs.

        Returns:
            List of GPU information
        """
        return [self.get_gpu_info(i) for i in range(self._device_count)]

    def set_device(self, device_id: int) -> None:
        """Set the current CUDA device.

        Args:
            device_id: Device ID to set
        """
        if self._available and device_id < self._device_count:
            torch.cuda.set_device(device_id)
            logger.info(f"Set CUDA device to {device_id}")

    def get_memory_info(self, device_id: int = 0) -> Dict[str, int]:
        """Get memory information for a GPU.

        Args:
            device_id: GPU device ID

        Returns:
            Dictionary with memory information
        """
        if not self._available:
            return {
                "total": 0,
                "allocated": 0,
                "reserved": 0,
                "free": 0,
            }

        info = self.get_gpu_info(device_id)
        return {
            "total": info.memory_total,
            "allocated": info.memory_allocated,
            "reserved": info.memory_reserved,
            "free": info.memory_total - info.memory_reserved,
        }

    def clear_cache(self) -> None:
        """Clear CUDA cache."""
        if self._available:
            torch.cuda.empty_cache()
            logger.info("GPU cache cleared")

    def synchronize(self, device_id: int = 0) -> None:
        """Synchronize CUDA operations.

        Args:
            device_id: Device ID to synchronize
        """
        if self._available:
            torch.cuda.synchronize(device_id)

    def set_memory_fraction(self, fraction: float, device_id: int = 0) -> None:
        """Set memory fraction for a device.

        Args:
            fraction: Memory fraction (0.0 to 1.0)
            device_id: Device ID
        """
        if self._available:
            torch.cuda.set_per_process_memory_fraction(fraction, device_id)
            logger.info(f"Set memory fraction to {fraction} for device {device_id}")


# Global GPU monitor instance
_gpu_monitor: Optional[GPUMonitor] = None


def get_gpu_monitor() -> GPUMonitor:
    """Get the global GPU monitor instance."""
    global _gpu_monitor
    if _gpu_monitor is None:
        _gpu_monitor = GPUMonitor()
    return _gpu_monitor