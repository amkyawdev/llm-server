"""Batch Processor - Handles batch processing of requests."""

from typing import List, Callable, Any, Dict, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import asyncio
from loguru import logger


class BatchProcessor:
    """Processes requests in batches."""

    def __init__(self, batch_size: int = 8, max_workers: int = 4):
        self.batch_size = batch_size
        self.max_workers = max_workers
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        logger.info(f"BatchProcessor initialized (batch_size={batch_size})")

    async def process_batch(
        self,
        items: List[Any],
        process_func: Callable,
        **kwargs,
    ) -> List[Any]:
        """Process a batch of items.

        Args:
            items: List of items to process
            process_func: Function to process each item
            **kwargs: Additional arguments for process_func

        Returns:
            List of results
        """
        results = []
        
        # Process in chunks
        for i in range(0, len(items), self.batch_size):
            batch = items[i:i + self.batch_size]
            
            # Submit batch for parallel processing
            futures = []
            for item in batch:
                future = self._executor.submit(process_func, item, **kwargs)
                futures.append(future)
            
            # Collect results
            for future in as_completed(futures):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    logger.error(f"Batch processing error: {e}")
                    results.append({"error": str(e)})

        return results

    async def process_batch_async(
        self,
        items: List[Any],
        process_func: Callable,
    ) -> List[Any]:
        """Process a batch asynchronously.

        Args:
            items: List of items to process
            process_func: Async function to process each item

        Returns:
            List of results
        """
        tasks = [process_func(item) for item in items]
        return await asyncio.gather(*tasks, return_exceptions=True)

    def process_streaming(
        self,
        items: List[Any],
        process_func: Callable,
    ):
        """Process items with streaming results.

        Args:
            items: List of items to process
            process_func: Function that yields results

        Yields:
            Processing results
        """
        for item in items:
            try:
                result = process_func(item)
                if hasattr(result, "__iter__"):
                    for r in result:
                        yield r
                else:
                    yield result
            except Exception as e:
                logger.error(f"Streaming processing error: {e}")
                yield {"error": str(e)}

    def shutdown(self) -> None:
        """Shutdown the batch processor."""
        self._executor.shutdown(wait=True)
        logger.info("BatchProcessor shutdown")


# Global batch processor instance
_batch_processor: Optional[BatchProcessor] = None


def get_batch_processor() -> BatchProcessor:
    """Get the batch processor singleton."""
    global _batch_processor
    if _batch_processor is None:
        _batch_processor = BatchProcessor()
    return _batch_processor