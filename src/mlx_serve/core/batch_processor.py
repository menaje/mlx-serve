"""Batch processor for continuous batching of inference requests."""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Generic, TypeVar

from mlx_serve.config import settings
from mlx_serve.core.inference_control import InferenceOverloadedError

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
        max_queue_size: int | None = None,
        execution_lock: asyncio.Lock | None = None,
        request_cost_fn: Callable[[T], int] | None = None,
        max_batch_cost: int | None = None,
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
        self.max_queue_size = (
            settings.inference_max_queue_per_model
            if max_queue_size is None
            else max_queue_size
        )
        queue_maxsize = 0 if self.max_queue_size is None else self.max_queue_size

        self._queue: asyncio.Queue[BatchRequest[T]] = asyncio.Queue(maxsize=queue_maxsize)
        self._running = False
        self._task: asyncio.Task | None = None
        self._start_lock = asyncio.Lock()
        self._execution_lock = execution_lock
        self._request_cost_fn = request_cost_fn or (lambda _data: 1)
        self._max_batch_cost = max_batch_cost
        self._carry_over: BatchRequest[T] | None = None

    async def start(self) -> None:
        """Start the batch processing loop."""
        async with self._start_lock:
            if self._running:
                return

            self._running = True
            self._task = asyncio.create_task(self._process_loop())
            logger.info(
                f"BatchProcessor started (max_batch={self.max_batch_size}, "
                f"max_wait={self.max_wait_ms}ms, "
                f"max_batch_cost={self._max_batch_cost}, "
                f"max_queue={'unbounded' if self.max_queue_size is None else self.max_queue_size})"
            )

    async def stop(self) -> None:
        """Stop the batch processing loop."""
        task = self._task
        if not self._running and (task is None or task.done()):
            return

        self._running = False
        if task and not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        self._task = None
        logger.info("BatchProcessor stopped")

    def stop_nowait(self) -> None:
        """Request loop shutdown from sync cleanup paths."""
        task = self._task
        if not self._running and (task is None or task.done()):
            return

        self._running = False
        if task and not task.done():
            loop = task.get_loop()
            if loop.is_running():
                loop.call_soon_threadsafe(task.cancel)
            else:
                task.cancel()

    def close_nowait(self) -> None:
        """Stop the processor and release queued work/references best-effort."""
        self.stop_nowait()
        if self._carry_over is not None and not self._carry_over.future.done():
            self._carry_over.future.cancel()
        self._carry_over = None
        while True:
            try:
                request = self._queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            if not request.future.done():
                request.future.cancel()
        self.process_fn = lambda _inputs: []

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
        try:
            self._queue.put_nowait(request)
        except asyncio.QueueFull as exc:
            raise InferenceOverloadedError("Batch queue is full") from exc

        # 무한 대기 방지: 배치 추론이 메모리 압박 등으로 멈추면 future가 영영 resolve되지
        # 않아 호출자가 hang하고 (max_concurrency_per_model 슬롯을 영구 점유해 후속 요청과
        # /v1/models까지 막힌다). queue_timeout으로 상한을 두고, 초과 시 future를 취소하고
        # overloaded 에러를 반환한다.
        timeout = settings.inference_queue_timeout_seconds
        if timeout is None:
            return await request.future
        try:
            return await asyncio.wait_for(request.future, timeout=timeout)
        except asyncio.TimeoutError as exc:
            if not request.future.done():
                request.future.cancel()
            raise InferenceOverloadedError(
                f"Batch processing timed out after {timeout:.1f}s "
                f"(inference stalled, likely memory pressure)"
            ) from exc

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
        total_cost = 0

        if self._carry_over is not None:
            first = self._carry_over
            self._carry_over = None
            batch.append(first)
        else:
            try:
                # Wait for first request
                first = await asyncio.wait_for(
                    self._queue.get(),
                    timeout=1.0,  # Check every second if still running
                )
                batch.append(first)
            except asyncio.TimeoutError:
                return []

        total_cost = self._request_cost(first.data)

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
                request_cost = self._request_cost(request.data)
                if (
                    self._max_batch_cost is not None
                    and batch
                    and (total_cost + request_cost) > self._max_batch_cost
                ):
                    self._carry_over = request
                    break
                batch.append(request)
                total_cost += request_cost
            except asyncio.TimeoutError:
                break

        return batch

    def _request_cost(self, data: T) -> int:
        """Return the batching cost of one request, clamped to a sane minimum."""
        try:
            cost = int(self._request_cost_fn(data))
        except Exception:
            cost = 1
        return max(1, cost)

    async def _process_batch(self, batch: list[BatchRequest[T]]) -> None:
        """Process a batch of requests."""
        if not batch:
            return

        if self._execution_lock is not None:
            async with self._execution_lock:
                await self._run_batch(batch)
            return

        await self._run_batch(batch)

    async def _run_batch(self, batch: list[BatchRequest[T]]) -> None:
        """Execute a single collected batch."""
        if not batch:
            return

        inputs = [req.data for req in batch]

        try:
            # Run processing in executor to avoid blocking event loop
            loop = asyncio.get_running_loop()
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

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        execution_lock: asyncio.Lock | None = None,
    ):
        """
        Initialize the embedding batch processor.

        Args:
            model: The embedding model.
            tokenizer: The tokenizer.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.max_batch_texts = settings.embedding_batch_max_texts
        self._processor = BatchProcessor(
            process_fn=self._generate_embeddings,
            execution_lock=execution_lock,
            request_cost_fn=len,
            max_batch_cost=self.max_batch_texts,
        )

    def _generate_embeddings(
        self,
        batched_texts: list[list[str]],
    ) -> list[list[list[float]]]:
        """Generate embeddings for multiple requests in one model call."""
        import mlx.core as mx

        flat_texts = [text for texts in batched_texts for text in texts]

        tok = getattr(self.tokenizer, "_tokenizer", self.tokenizer)
        inputs = tok(
            flat_texts,
            return_tensors="np",
            padding=True,
            truncation=True,
            max_length=512,
        )

        input_ids = mx.array(inputs["input_ids"])
        attention_mask = mx.array(inputs["attention_mask"])

        result = self.model(input_ids, attention_mask=attention_mask)
        embeddings = result.text_embeds.tolist()

        outputs: list[list[list[float]]] = []
        offset = 0
        for texts in batched_texts:
            next_offset = offset + len(texts)
            outputs.append(embeddings[offset:next_offset])
            offset = next_offset

        return outputs

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
        if len(texts) <= self.max_batch_texts:
            return await self._processor.submit(texts)

        outputs: list[list[float]] = []
        for start in range(0, len(texts), self.max_batch_texts):
            chunk = texts[start : start + self.max_batch_texts]
            outputs.extend(await self._processor.submit(chunk))
        return outputs

    async def start(self) -> None:
        """Start the batch processor."""
        await self._processor.start()

    async def stop(self) -> None:
        """Stop the batch processor."""
        await self._processor.stop()

    def stop_nowait(self) -> None:
        """Request the batch processor to stop without awaiting cancellation."""
        self._processor.stop_nowait()

    def close_nowait(self) -> None:
        """Stop processing and break model/tokenizer references."""
        self._processor.close_nowait()
        self.model = None
        self.tokenizer = None


class RerankBatchProcessor:
    """Batch processor for reranking operations."""

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        max_batch_documents: int | None = None,
        max_batch_tokens: int | None = None,
    ):
        """
        Initialize the rerank batch processor.

        Args:
            model: The reranker model.
            tokenizer: The tokenizer.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.max_batch_documents = (
            settings.rerank_batch_max_documents
            if max_batch_documents is None
            else max_batch_documents
        )
        self.max_batch_tokens = (
            settings.rerank_batch_max_tokens
            if max_batch_tokens is None
            else max_batch_tokens
        )
        self._default_instruction = (
            "Given a web search query, retrieve relevant passages "
            "that answer the query"
        )
        self._prefix_tokens = (
            [self.tokenizer.bos_token_id]
            if getattr(self.tokenizer, "bos_token_id", None) is not None
            else []
        )
        self._suffix_tokens = (
            [self.tokenizer.eos_token_id]
            if getattr(self.tokenizer, "eos_token_id", None) is not None
            else []
        )
        self._pad_token_id = self._resolve_pad_token_id()

    def _resolve_pad_token_id(self) -> int:
        """Pick a stable pad token id for right-padded micro-batches."""
        for candidate in (
            getattr(self.tokenizer, "pad_token_id", None),
            getattr(self.tokenizer, "eos_token_id", None),
            0,
        ):
            if candidate is not None:
                return int(candidate)
        return 0

    def _encode_prompt(
        self,
        query: str,
        document: str,
        instruction: str,
    ) -> list[int]:
        """Encode one rerank prompt with explicit BOS/EOS handling."""
        prompt = f"<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {document}"
        input_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        return self._prefix_tokens + input_ids + self._suffix_tokens

    def _split_micro_batches(
        self,
        encoded_prompts: list[list[int]],
    ) -> list[list[list[int]]]:
        """Split encoded prompts into bounded micro-batches."""
        if not encoded_prompts:
            return []

        batches: list[list[list[int]]] = []
        current_batch: list[list[int]] = []
        current_max_len = 0

        for encoded_prompt in encoded_prompts:
            prompt_len = max(1, len(encoded_prompt))
            next_batch_size = len(current_batch) + 1
            next_max_len = max(current_max_len, prompt_len)

            exceeds_doc_limit = next_batch_size > self.max_batch_documents
            exceeds_token_limit = (
                bool(current_batch)
                and (next_batch_size * next_max_len) > self.max_batch_tokens
            )

            if exceeds_doc_limit or exceeds_token_limit:
                batches.append(current_batch)
                current_batch = [encoded_prompt]
                current_max_len = prompt_len
                continue

            current_batch.append(encoded_prompt)
            current_max_len = next_max_len

        if current_batch:
            batches.append(current_batch)

        return batches

    def _score_micro_batch(self, encoded_prompts: list[list[int]]) -> list[float]:
        """Run one padded rerank micro-batch in a single model call."""
        import mlx.core as mx

        token_true_id = self.tokenizer.convert_tokens_to_ids("yes")
        token_false_id = self.tokenizer.convert_tokens_to_ids("no")

        lengths = [len(prompt) for prompt in encoded_prompts]
        max_len = max(lengths)
        padded = [
            prompt + [self._pad_token_id] * (max_len - len(prompt))
            for prompt in encoded_prompts
        ]
        tokens = mx.array(padded)

        outputs = self.model(tokens)
        logits = outputs.logits if hasattr(outputs, "logits") else outputs

        # Extract only the two scalar scores per row before releasing the full logits tensor.
        # Slicing row-by-row keeps the full [batch, seq_len, vocab_size] logits alive
        # until the loop ends; collecting scalar pairs first lets us drop it early.
        row_pairs: list[tuple[mx.array, mx.array]] = [
            (logits[i, prompt_len - 1, token_true_id], logits[i, prompt_len - 1, token_false_id])
            for i, prompt_len in enumerate(lengths)
        ]
        del tokens, outputs, logits

        scores: list[float] = []
        for true_score, false_score in row_pairs:
            probs = mx.softmax(mx.stack([false_score, true_score]))
            scores.append(float(probs[1]))

        return scores

    def compute_scores(
        self,
        query: str,
        documents: list[str],
        instruction: str | None = None,
    ) -> tuple[list[float], int]:
        """
        Compute relevance scores for query-document pairs.

        Uses request-scoped micro-batching to bound memory usage.

        Args:
            query: The search query.
            documents: List of documents to score.
            instruction: Optional instruction for the reranker.

        Returns:
            Tuple of relevance scores and total token count.
        """
        if instruction is None:
            instruction = self._default_instruction

        encoded_prompts = [
            self._encode_prompt(query, document, instruction)
            for document in documents
        ]
        total_tokens = sum(len(prompt) for prompt in encoded_prompts)

        scores: list[float] = []
        for micro_batch in self._split_micro_batches(encoded_prompts):
            scores.extend(self._score_micro_batch(micro_batch))

        return scores, total_tokens

    async def rerank(
        self,
        query: str,
        documents: list[str],
        instruction: str | None = None,
    ) -> tuple[list[float], int]:
        """
        Rerank documents with batch optimization.

        Args:
            query: The search query.
            documents: List of documents to rerank.
            instruction: Optional instruction for the reranker.

        Returns:
            Tuple of relevance scores and total token count.
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            self.compute_scores,
            query,
            documents,
            instruction,
        )
