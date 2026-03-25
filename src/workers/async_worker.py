"""Async Worker - Handles asynchronous task execution."""

import asyncio
from typing import Optional, Callable, Any, Dict
from concurrent.futures import ThreadPoolExecutor
from loguru import logger


class AsyncWorker:
    """Handles asynchronous task execution with worker pool."""

    def __init__(self, max_workers: int = 4):
        self.max_workers = max_workers
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._tasks: Dict[str, asyncio.Task] = {}
        logger.info(f"AsyncWorker initialized with {max_workers} workers")

    async def run_task(
        self,
        task_id: str,
        func: Callable,
        *args,
        **kwargs,
    ) -> Any:
        """Run a task asynchronously.

        Args:
            task_id: Unique task identifier
            func: Function to run
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            Task result
        """
        loop = asyncio.get_event_loop()
        task = loop.run_in_executor(
            self._executor,
            lambda: func(*args, **kwargs),
        )
        self._tasks[task_id] = task

        try:
            result = await task
            return result
        finally:
            self._tasks.pop(task_id, None)

    async def run_tasks(self, tasks: list) -> list:
        """Run multiple tasks concurrently.

        Args:
            tasks: List of (func, args, kwargs) tuples

        Returns:
            List of results
        """
        coroutines = []
        for task in tasks:
            func, args, kwargs = task
            coroutines.append(
                asyncio.get_event_loop().run_in_executor(
                    self._executor,
                    lambda f=f, a=args, k=kwargs: f(*a, **k),
                )
            )

        return await asyncio.gather(*coroutines, return_exceptions=True)

    def submit_background(
        self,
        task_id: str,
        func: Callable,
        *args,
        **kwargs,
    ) -> None:
        """Submit a task to run in the background.

        Args:
            task_id: Unique task identifier
            func: Function to run
            *args: Positional arguments
            **kwargs: Keyword arguments
        """
        loop = asyncio.get_event_loop()
        task = loop.run_in_executor(
            self._executor,
            lambda: func(*args, **kwargs),
        )
        self._tasks[task_id] = task
        logger.info(f"Background task {task_id} submitted")

    async def cancel_task(self, task_id: str) -> bool:
        """Cancel a running task.

        Args:
            task_id: Task ID to cancel

        Returns:
            True if cancelled
        """
        if task_id in self._tasks:
            self._tasks[task_id].cancel()
            return True
        return False

    def get_task_status(self, task_id: str) -> Optional[str]:
        """Get status of a task.

        Args:
            task_id: Task ID

        Returns:
            Status string or None
        """
        if task_id not in self._tasks:
            return None

        task = self._tasks[task_id]
        if task.done():
            return "done"
        return "running"

    def shutdown(self) -> None:
        """Shutdown the worker pool."""
        self._executor.shutdown(wait=True)
        logger.info("AsyncWorker shutdown")


# Global worker instance
_worker: Optional[AsyncWorker] = None


def get_async_worker() -> AsyncWorker:
    """Get the async worker singleton."""
    global _worker
    if _worker is None:
        _worker = AsyncWorker()
    return _worker