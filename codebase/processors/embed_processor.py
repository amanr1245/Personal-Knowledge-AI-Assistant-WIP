"""
Parallel Embedding Generation Module

Provides thread-safe parallel embedding generation using OpenAI API.
Each worker uses its own OpenAI client instance for thread safety.
"""

import os
import concurrent.futures
from typing import List, Callable, Optional
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

MAX_WORKERS = int(os.getenv("EMBED_MAX_WORKERS", "8"))
BATCH_SIZE = int(os.getenv("EMBED_BATCH_SIZE", "100"))  # OpenAI batch size per request


class EmbedProcessor:
    """
    Thread-safe parallel embedding processor.

    Each worker thread creates its own OpenAI client instance
    to ensure thread safety.
    """

    def __init__(
        self,
        max_workers: int = None,
        batch_size: int = None,
        model: str = "text-embedding-3-small"
    ):
        """
        Initialize the embedding processor.

        Args:
            max_workers: Maximum parallel workers (default from env or 8)
            batch_size: Texts per API request (default from env or 100)
            model: OpenAI embedding model to use
        """
        self.max_workers = max_workers or MAX_WORKERS
        self.batch_size = batch_size or BATCH_SIZE
        self.model = model
        self.api_key = os.getenv("OPENAI_API_KEY")

    def _create_client(self) -> OpenAI:
        """Create a new OpenAI client for thread-local use."""
        return OpenAI(api_key=self.api_key)

    def _embed_batch(self, batch_info: tuple) -> tuple:
        """
        Embed a single batch of texts.

        Creates its own client instance for thread safety.

        Args:
            batch_info: (batch_index, texts) tuple

        Returns:
            (batch_index, embeddings) tuple
        """
        import sys
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        import cache_manager
        from retry_manager import with_retry

        batch_idx, texts = batch_info

        # Check cache for each text
        embeddings = []
        uncached_texts = []
        uncached_indices = []

        for i, text in enumerate(texts):
            cached = cache_manager.get_cached_embedding(text)
            if cached is not None:
                embeddings.append((i, cached))
            else:
                uncached_texts.append(text)
                uncached_indices.append(i)

        # Embed uncached texts
        if uncached_texts:
            client = self._create_client()

            response = with_retry(
                client.embeddings.create,
                model=self.model,
                input=uncached_texts
            )

            new_embeddings = [item.embedding for item in response.data]

            # Cache and add to results
            for idx, (text, embedding) in enumerate(zip(uncached_texts, new_embeddings)):
                cache_manager.cache_embedding(text, embedding)
                embeddings.append((uncached_indices[idx], embedding))

        # Sort by original index within batch
        embeddings.sort(key=lambda x: x[0])
        return (batch_idx, [emb for _, emb in embeddings])

    def embed_batch_parallel(
        self,
        texts: List[str],
        progress_callback: Callable[[int, int], None] = None
    ) -> List[List[float]]:
        """
        Embed texts in parallel batches.

        Args:
            texts: List of texts to embed
            progress_callback: Optional callback(completed, total) for progress

        Returns:
            List of embedding vectors in same order as input
        """
        if not texts:
            return []

        # Split into batches
        batches = [
            (i, texts[i:i + self.batch_size])
            for i in range(0, len(texts), self.batch_size)
        ]
        batches = [(i, batch) for i, (_, batch) in enumerate(batches)]

        total_batches = len(batches)
        completed = 0

        print(f"Embedding {len(texts)} texts in {total_batches} batches with {self.max_workers} workers...")

        results = {}

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_batch = {
                executor.submit(self._embed_batch, (i, batch)): i
                for i, batch in batches
            }

            for future in concurrent.futures.as_completed(future_to_batch):
                batch_idx = future_to_batch[future]
                completed += 1

                try:
                    idx, embeddings = future.result()
                    results[idx] = embeddings
                    status = f"{len(embeddings)} embeddings"
                except Exception as e:
                    print(f"Embedding error for batch {batch_idx}: {e}")
                    # Return empty embeddings for failed batch
                    results[batch_idx] = [[0.0] * 1536] * len(batches[batch_idx][1])
                    status = "ERROR"

                # Progress update
                if progress_callback:
                    progress_callback(completed, total_batches)
                else:
                    print(f"  [{completed}/{total_batches}] Batch {batch_idx}: {status}")

        # Flatten in order
        all_embeddings = []
        for i in range(len(batches)):
            all_embeddings.extend(results[i])

        return all_embeddings


# Module-level convenience function
def embed_texts_parallel(
    texts: List[str],
    max_workers: int = None,
    batch_size: int = None,
    model: str = "text-embedding-3-small"
) -> List[List[float]]:
    """
    Embed multiple texts in parallel.

    Args:
        texts: List of texts to embed
        max_workers: Max parallel workers
        batch_size: Texts per API request
        model: OpenAI embedding model

    Returns:
        List of embedding vectors
    """
    processor = EmbedProcessor(max_workers, batch_size, model)
    return processor.embed_batch_parallel(texts)


if __name__ == "__main__":
    print(f"EmbedProcessor configured with MAX_WORKERS={MAX_WORKERS}, BATCH_SIZE={BATCH_SIZE}")
    print("Usage: from processors import EmbedProcessor")
