"""Queue Manager - Manages task queues for processing."""

from typing import Optional, Any, Dict, List
from queue import Queue, Empty
from threading import Thread, Lock
from dataclasses import dataclass
from datetime import datetime
import time
import uuid

from loguru import logger


@dataclass
class Task:
    """Task container."""

    id: str
    func: Any
    args: tuple
    kwargs: dict
    priority: int
    created_at: float
    status: str = "pending"
    result: Any = None
    error: Optional[str] = None


class QueueManager:
    """Manages task queues for async processing."""

    def __init__(self, max_size: int = 1000, num_workers: int = 4):
        self.max_size = max_size
        self.num_workers = num_workers
        self._queues: Dict[str, Queue] = {}
        self._workers: List[Thread] = []
        self._running = False
        self._lock = Lock()
        logger.info(f"QueueManager initialized (workers={num_workers})")

    def create_queue(self, name: str) -> None:
        """Create a named queue.

        Args:
            name: Queue name
        """
        with self._lock:
            if name not in self._queues:
                self._queues[name] = Queue(maxsize=self.max_size)
                logger.info(f"Queue created: {name}")

    def enqueue(
        self,
        queue_name: str,
        func: Any,
        *args,
        priority: int = 0,
        **kwargs,
    ) -> str:
        """Add a task to the queue.

        Args:
            queue_name: Queue name
            func: Function to execute
            *args: Positional arguments
            priority: Task priority (higher = more urgent)
            **kwargs: Keyword arguments

        Returns:
            Task ID
        """
        if queue_name not in self._queues:
            self.create_queue(queue_name)

        task = Task(
            id=str(uuid.uuid4())[:8],
            func=func,
            args=args,
            kwargs=kwargs,
            priority=priority,
            created_at=time.time(),
        )

        self._queues[queue_name].put(task)
        logger.debug(f"Task {task.id} enqueued to {queue_name}")

        return task.id

    def dequeue(self, queue_name: str, timeout: float = 1.0) -> Optional[Task]:
        """Get a task from the queue.

        Args:
            queue_name: Queue name
            timeout: Timeout in seconds

        Returns:
            Task or None
        """
        if queue_name not in self._queues:
            return None

        try:
            return self._queues[queue_name].get(timeout=timeout)
        except Empty:
            return None

    def get_queue_size(self, queue_name: str) -> int:
        """Get the size of a queue.

        Args:
            queue_name: Queue name

        Returns:
            Queue size
        """
        if queue_name in self._queues:
            return self._queues[queue_name].qsize()
        return 0

    def start_workers(self) -> None:
        """Start worker threads."""
        if self._running:
            return

        self._running = True

        for i in range(self.num_workers):
            worker = Thread(target=self._worker_loop, args=(i,), daemon=True)
            worker.start()
            self._workers.append(worker)

        logger.info(f"Started {self.num_workers} worker threads")

    def _worker_loop(self, worker_id: int) -> None:
        """Worker thread loop.

        Args:
            worker_id: Worker ID
        """
        logger.info(f"Worker {worker_id} started")

        while self._running:
            task = None

            # Try to get task from any queue (priority order)
            for queue_name in self._queues:
                task = self.dequeue(queue_name, timeout=1.0)
                if task:
                    break

            if task:
                try:
                    task.status = "running"
                    task.result = task.func(*task.args, **task.kwargs)
                    task.status = "completed"
                    logger.debug(f"Task {task.id} completed")
                except Exception as e:
                    task.status = "failed"
                    task.error = str(e)
                    logger.error(f"Task {task.id} failed: {e}")
            else:
                time.sleep(0.1)

        logger.info(f"Worker {worker_id} stopped")

    def stop_workers(self) -> None:
        """Stop worker threads."""
        self._running = False

        for worker in self._workers:
            worker.join(timeout=5.0)

        self._workers.clear()
        logger.info("Worker threads stopped")

    def get_task_status(self, queue_name: str, task_id: str) -> Optional[Dict]:
        """Get status of a task.

        Args:
            queue_name: Queue name
            task_id: Task ID

        Returns:
            Task status dict or None
        """
        # This would need to track tasks in a more sophisticated way
        # Simplified implementation
        return None

    def clear_queue(self, queue_name: str) -> int:
        """Clear all tasks from a queue.

        Args:
            queue_name: Queue name

        Returns:
            Number of tasks cleared
        """
        if queue_name not in self._queues:
            return 0

        count = 0
        while not self._queues[queue_name].empty():
            try:
                self._queues[queue_name].get_nowait()
                count += 1
            except Empty:
                break

        logger.info(f"Cleared {count} tasks from queue {queue_name}")
        return count


# Global queue manager instance
_queue_manager: Optional[QueueManager] = None


def get_queue_manager() -> QueueManager:
    """Get the queue manager singleton."""
    global _queue_manager
    if _queue_manager is None:
        _queue_manager = QueueManager()
    return _queue_manager