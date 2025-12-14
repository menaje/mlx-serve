"""Batch processor for continuous batching of inference requests."""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Generic, TypeVar

from mlx_serve.config import settings

logger = logging.getLogger(__name__)

T = TypeVar("T")
R = TypeVar("R")


@dataclass
class BatchRequest(Generic[T]):
    """A request waiting to be batched."""

    data: T
    future: asyncio.Future = field(default_factory=asyncio.Future)
    timestamp: float = field(default_factory=time.time)


class BatchProcessor(Generic[T, R]):
    """
    Continuous batching processor for inference requests.

    Collects requests into batches and processes them together for
    improved throughput on GPU.
    """

    def __init__(
        self,
        process_fn: Callable[[list[T]], list[R]],
        max_batch_size: int | None = None,
        max_wait_ms: int | None = None,
    ):
        """
        Initialize the batch processor.

        Args:
            process_fn: Function to process a batch of requests.
                       Takes list of inputs, returns list of outputs.
            max_batch_size: Maximum batch size. Defaults to settings.batch_max_size.
            max_wait_ms: Maximum wait time in ms. Defaults to settings.batch_max_wait_ms.
        """
        self.process_fn = process_fn
        self.max_batch_size = max_batch_size or settings.batch_max_size
        self.max_wait_ms = max_wait_ms or settings.batch_max_wait_ms

        self._queue: asyncio.Queue[BatchRequest[T]] = asyncio.Queue()
        self._running = False
        self._task: asyncio.Task | None = None

    async def start(self) -> None:
        """Start the batch processing loop."""
        if self._running:
            return

        self._running = True
        self._task = asyncio.create_task(self._process_loop())
        logger.info(
            f"BatchProcessor started (max_batch={self.max_batch_size}, "
            f"max_wait={self.max_wait_ms}ms)"
        )

    async def stop(self) -> None:
        """Stop the batch processing loop."""
        if not self._running:
            return

        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("BatchProcessor stopped")

    async def submit(self, data: T) -> R:
        """
        Submit a request for batch processing.

        Args:
            data: Input data for the request.

        Returns:
            Processed result.
        """
        if not self._running:
            await self.start()

        request: BatchRequest[T] = BatchRequest(data=data)
        await self._queue.put(request)
        return await request.future

    async def _process_loop(self) -> None:
        """Main loop for collecting and processing batches."""
        while self._running:
            try:
                batch = await self._collect_batch()
                if batch:
                    await self._process_batch(batch)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Batch processing error: {e}")
                await asyncio.sleep(0.1)

    async def _collect_batch(self) -> list[BatchRequest[T]]:
        """Collect requests into a batch."""
        batch: list[BatchRequest[T]] = []

        try:
            # Wait for first request
            first = await asyncio.wait_for(
                self._queue.get(),
                timeout=1.0,  # Check every second if still running
            )
            batch.append(first)
        except asyncio.TimeoutError:
            return []

        # Collect more requests up to max_batch_size or max_wait_ms
        deadline = time.time() + (self.max_wait_ms / 1000.0)

        while len(batch) < self.max_batch_size:
            remaining = deadline - time.time()
            if remaining <= 0:
                break

            try:
                request = await asyncio.wait_for(
                    self._queue.get(),
                    timeout=remaining,
                )
                batch.append(request)
            except asyncio.TimeoutError:
                break

        return batch

    async def _process_batch(self, batch: list[BatchRequest[T]]) -> None:
        """Process a batch of requests."""
        if not batch:
            return

        inputs = [req.data for req in batch]

        try:
            # Run processing in executor to avoid blocking event loop
            loop = asyncio.get_event_loop()
            results = await loop.run_in_executor(None, self.process_fn, inputs)

            # Distribute results
            for request, result in zip(batch, results):
                if not request.future.done():
                    request.future.set_result(result)

            logger.debug(f"Processed batch of {len(batch)} requests")

        except Exception as e:
            # Propagate error to all pending requests
            for request in batch:
                if not request.future.done():
                    request.future.set_exception(e)
            logger.error(f"Batch processing failed: {e}")


class EmbeddingBatchProcessor:
    """Batch processor specifically for embedding generation."""

    def __init__(self, model: Any, tokenizer: Any):
        """
        Initialize the embedding batch processor.

        Args:
            model: The embedding model.
            tokenizer: The tokenizer.
        """
        self.model = model
        self.tokenizer = tokenizer
        self._processor = BatchProcessor(
            process_fn=self._generate_embeddings,
        )

    def _generate_embeddings(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a batch of texts."""
        from mlx_embeddings import generate

        result = generate(self.model, self.tokenizer, texts)
        return result.text_embeds.tolist()

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """
        Generate embeddings with batch optimization.

        If multiple requests arrive simultaneously, they will be
        batched together for efficient GPU utilization.

        Args:
            texts: List of texts to embed.

        Returns:
            List of embedding vectors.
        """
        # For single-request batching, submit each text separately
        # The processor will batch concurrent requests
        if len(texts) == 1:
            return [await self._processor.submit(texts[0])]

        # For multi-text requests, process directly as a batch
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self._generate_embeddings, texts
        )

    async def start(self) -> None:
        """Start the batch processor."""
        await self._processor.start()

    async def stop(self) -> None:
        """Stop the batch processor."""
        await self._processor.stop()


class RerankBatchProcessor:
    """Batch processor for reranking operations."""

    def __init__(self, model: Any, tokenizer: Any):
        """
        Initialize the rerank batch processor.

        Args:
            model: The reranker model.
            tokenizer: The tokenizer.
        """
        self.model = model
        self.tokenizer = tokenizer

    def compute_scores(
        self,
        query: str,
        documents: list[str],
        instruction: str | None = None,
    ) -> list[float]:
        """
        Compute relevance scores for query-document pairs.

        Uses batch processing for efficiency.

        Args:
            query: The search query.
            documents: List of documents to score.
            instruction: Optional instruction for the reranker.

        Returns:
            List of relevance scores.
        """
        import mlx.core as mx
        import mlx.nn as nn

        if instruction is None:
            instruction = "Given a web search query, retrieve relevant passages that answer the query"

        scores = []

        # Batch tokenization
        prompts = [
            f"<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}"
            for doc in documents
        ]

        # Get token IDs for "yes" and "no"
        token_true_id = self.tokenizer.convert_tokens_to_ids("yes")
        token_false_id = self.tokenizer.convert_tokens_to_ids("no")

        # Process each prompt (sequential for now, can be parallelized)
        for prompt in prompts:
            # Tokenize
            prefix_tokens = [self.tokenizer.bos_token_id] if self.tokenizer.bos_token_id else []
            suffix_tokens = [self.tokenizer.eos_token_id] if self.tokenizer.eos_token_id else []

            input_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
            input_ids = prefix_tokens + input_ids + suffix_tokens

            tokens = mx.array([input_ids])

            # Get model output
            outputs = self.model(tokens)
            logits = outputs.logits if hasattr(outputs, "logits") else outputs

            # Get last token logits
            last_logits = logits[0, -1, :]

            # Compute probability
            true_score = last_logits[token_true_id]
            false_score = last_logits[token_false_id]
            probs = mx.softmax(mx.stack([false_score, true_score]))

            scores.append(float(probs[1]))

        return scores

    async def rerank(
        self,
        query: str,
        documents: list[str],
        instruction: str | None = None,
    ) -> list[float]:
        """
        Rerank documents with batch optimization.

        Args:
            query: The search query.
            documents: List of documents to rerank.
            instruction: Optional instruction for the reranker.

        Returns:
            List of relevance scores.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self.compute_scores,
            query,
            documents,
            instruction,
        )
