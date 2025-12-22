"""
Reranker Module for RAG

Two-stage retrieval:
1. Vector search returns top N candidates
2. LLM reranks by relevance to question
3. Take top K after reranking

This improves answer quality by finding the most relevant chunks,
not just the most semantically similar ones.
"""

import os
import json
from typing import List, Tuple, Any
from dotenv import load_dotenv
from pydantic import BaseModel
from openai import OpenAI

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Configuration from .env with defaults
RERANK_CANDIDATES = int(os.getenv("RERANK_CANDIDATES", "20"))
RERANK_TOP_K = int(os.getenv("RERANK_TOP_K", "5"))


class RerankResponse(BaseModel):
    """Response schema for reranking."""
    indices: List[int]  # Reordered indices, most relevant first
    rejected: bool = False  # True if no chunks are relevant


RERANK_PROMPT = """Given a question and candidate text chunks, rerank by relevance.

QUESTION: {question}

CHUNKS:
{numbered_chunks}

Return the chunk indices in order of relevance to answering the question.
Most relevant first. Only include chunks that help answer the question.

If no chunks are relevant to the question, set rejected=true and return empty indices.

Respond ONLY with valid JSON matching this schema:
{{
    "indices": [3, 1, 7, ...],
    "rejected": false
}}

Rules:
- indices should contain the chunk numbers that are relevant, ordered by relevance
- Most relevant chunk first
- Only include chunks that actually help answer the question
- If none are relevant, use {{"indices": [], "rejected": true}}"""


def _validate_and_repair_indices(
    indices: List[int],
    num_chunks: int
) -> List[int]:
    """
    Validate and repair the returned indices.

    - Remove duplicates (keep first occurrence)
    - Remove out-of-range indices
    - Handle malformed responses gracefully

    Args:
        indices: Raw indices from LLM
        num_chunks: Total number of chunks

    Returns:
        Cleaned list of valid indices
    """
    seen = set()
    valid_indices = []

    for idx in indices:
        # Check if it's a valid integer
        if not isinstance(idx, int):
            try:
                idx = int(idx)
            except (ValueError, TypeError):
                continue

        # Check if in valid range (0-indexed)
        if idx < 0 or idx >= num_chunks:
            continue

        # Skip duplicates
        if idx in seen:
            continue

        seen.add(idx)
        valid_indices.append(idx)

    return valid_indices


def rerank_chunks(
    question: str,
    chunks: List[Tuple[Any, ...]],
    top_k: int = None
) -> Tuple[List[Tuple[Any, ...]], bool]:
    """
    Rerank chunks by relevance to the question using LLM.

    Args:
        question: The user's question
        chunks: List of chunk tuples (chunk_text, header, source, chunk_index, similarity)
        top_k: Number of top chunks to return (default: RERANK_TOP_K from .env)

    Returns:
        Tuple of (reranked_chunks, rejected_flag)
        - reranked_chunks: Top K chunks ordered by relevance
        - rejected_flag: True if LLM determined no chunks are relevant
    """
    if top_k is None:
        top_k = RERANK_TOP_K

    if not chunks:
        return [], True

    # Build numbered chunks for the prompt
    # Chunk tuple format: (chunk_text, header, source, chunk_index, similarity)
    numbered_chunks = ""
    for i, chunk_tuple in enumerate(chunks):
        chunk_text = chunk_tuple[0]
        header = chunk_tuple[1]

        if header:
            numbered_chunks += f"[{i}] {header}\n{chunk_text[:500]}{'...' if len(chunk_text) > 500 else ''}\n\n"
        else:
            numbered_chunks += f"[{i}] {chunk_text[:500]}{'...' if len(chunk_text) > 500 else ''}\n\n"

    prompt = RERANK_PROMPT.format(
        question=question,
        numbered_chunks=numbered_chunks
    )

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "You are a relevance ranking expert. Return only valid JSON."
                },
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=500
        )

        response_text = response.choices[0].message.content.strip()

        # Clean up response if wrapped in markdown
        if response_text.startswith("```"):
            response_text = response_text.split("```")[1]
            if response_text.startswith("json"):
                response_text = response_text[4:]
            response_text = response_text.strip()

        # Parse JSON response
        data = json.loads(response_text)
        rerank_response = RerankResponse(**data)

        # Check if rejected
        if rerank_response.rejected or not rerank_response.indices:
            return [], True

        # Validate and repair indices
        valid_indices = _validate_and_repair_indices(
            rerank_response.indices,
            len(chunks)
        )

        if not valid_indices:
            return [], True

        # Reorder chunks by the validated indices
        reranked = [chunks[i] for i in valid_indices[:top_k]]

        return reranked, False

    except Exception as e:
        print(f"Reranking error: {e}")
        # Fallback: return original chunks in order
        return chunks[:top_k], False


def rerank_with_fallback(
    question: str,
    chunks: List[Tuple[Any, ...]],
    top_k: int = None
) -> List[Tuple[Any, ...]]:
    """
    Rerank chunks with automatic fallback to original order on failure.

    Args:
        question: The user's question
        chunks: List of chunk tuples
        top_k: Number of top chunks to return

    Returns:
        Reranked chunks (or original top_k if reranking fails/rejects)
    """
    if top_k is None:
        top_k = RERANK_TOP_K

    reranked, rejected = rerank_chunks(question, chunks, top_k)

    if rejected or not reranked:
        # Fallback to original vector similarity order
        return chunks[:top_k]

    return reranked


# CLI for testing
if __name__ == "__main__":
    # Test with sample data
    test_chunks = [
        ("Paging is a memory management scheme.", "Memory Paging Basics", "test.pdf", 0, 0.85),
        ("The CPU executes instructions in a pipeline.", "CPU Pipeline", "test.pdf", 1, 0.80),
        ("Virtual memory uses page tables for address translation.", "Page Tables", "test.pdf", 2, 0.78),
        ("Network protocols define communication standards.", "Networking", "test.pdf", 3, 0.75),
        ("Page faults occur when a page is not in memory.", "Page Faults", "test.pdf", 4, 0.73),
    ]

    question = "How does paging work in operating systems?"

    print(f"Question: {question}")
    print(f"\nOriginal chunks (by vector similarity):")
    for i, (text, header, _, _, sim) in enumerate(test_chunks):
        print(f"  [{i}] {header}: {text[:50]}... (sim: {sim})")

    reranked, rejected = rerank_chunks(question, test_chunks, top_k=3)

    print(f"\nReranked chunks (rejected={rejected}):")
    for i, (text, header, _, _, sim) in enumerate(reranked):
        print(f"  [{i}] {header}: {text[:50]}... (orig sim: {sim})")
