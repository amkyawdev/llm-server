"""Metrics collection and monitoring."""

import time
from typing import Dict, Optional, List
from dataclasses import dataclass, field
from collections import defaultdict
import threading


@dataclass
class RequestMetrics:
    """Metrics for a single request."""

    request_id: str
    endpoint: str
    method: str
    start_time: float
    end_time: Optional[float] = None
    status_code: Optional[int] = None
    error: Optional[str] = None
    tokens_in: int = 0
    tokens_out: int = 0


class MetricsCollector:
    """Collects and aggregates metrics."""

    def __init__(self):
        self._requests: List[RequestMetrics] = []
        self._counters: Dict[str, int] = defaultdict(int)
        self._gauges: Dict[str, float] = {}
        self._histograms: Dict[str, List[float]] = defaultdict(list)
        self._lock = threading.Lock()
        self._start_time = time.time()

    def record_request(
        self,
        request_id: str,
        endpoint: str,
        method: str,
    ) -> RequestMetrics:
        """Record the start of a request."""
        metrics = RequestMetrics(
            request_id=request_id,
            endpoint=endpoint,
            method=method,
            start_time=time.time(),
        )

        with self._lock:
            self._requests.append(metrics)
            self._counters[f"requests_{method}_{endpoint}"] += 1
            self._counters["requests_total"] += 1

        return metrics

    def record_response(
        self,
        request_id: str,
        status_code: int,
        tokens_in: int = 0,
        tokens_out: int = 0,
    ) -> None:
        """Record the response for a request."""
        with self._lock:
            for req in self._requests:
                if req.request_id == request_id:
                    req.end_time = time.time()
                    req.status_code = status_code
                    req.tokens_in = tokens_in
                    req.tokens_out = tokens_out
                    break

            # Update histograms
            if status_code >= 200 and status_code < 300:
                self._histograms["response_time_success"].append(
                    time.time() - self._requests[-1].start_time
                )
            else:
                self._histograms["response_time_error"].append(
                    time.time() - self._requests[-1].start_time
                )

            # Update counters
            if status_code >= 500:
                self._counters["errors_5xx"] += 1
            elif status_code >= 400:
                self._counters["errors_4xx"] += 1

    def record_error(self, request_id: str, error: str) -> None:
        """Record an error for a request."""
        with self._lock:
            for req in self._requests:
                if req.request_id == request_id:
                    req.error = error
                    self._counters["errors_total"] += 1
                    break

    def increment(self, name: str, value: int = 1) -> None:
        """Increment a counter."""
        with self._lock:
            self._counters[name] += value

    def gauge(self, name: str, value: float) -> None:
        """Set a gauge value."""
        with self._lock:
            self._gauges[name] = value

    def histogram(self, name: str, value: float) -> None:
        """Record a histogram value."""
        with self._lock:
            self._histograms[name].append(value)

    def get_metrics(self) -> Dict:
        """Get all metrics."""
        with self._lock:
            uptime = time.time() - self._start_time

            return {
                "uptime_seconds": uptime,
                "counters": dict(self._counters),
                "gauges": dict(self._gauges),
                "histograms": {
                    name: {
                        "count": len(values),
                        "min": min(values) if values else 0,
                        "max": max(values) if values else 0,
                        "avg": sum(values) / len(values) if values else 0,
                    }
                    for name, values in self._histograms.items()
                },
                "requests": {
                    "total": len(self._requests),
                    "completed": sum(1 for r in self._requests if r.end_time),
                },
            }

    def reset(self) -> None:
        """Reset all metrics."""
        with self._lock:
            self._requests.clear()
            self._counters.clear()
            self._gauges.clear()
            self._histograms.clear()
            self._start_time = time.time()


# Global metrics collector
_metrics_collector: Optional[MetricsCollector] = None


def get_metrics_collector() -> MetricsCollector:
    """Get the metrics collector singleton."""
    global _metrics_collector
    if _metrics_collector is None:
        _metrics_collector = MetricsCollector()
    return _metrics_collector