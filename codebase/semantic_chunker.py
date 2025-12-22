"""
Semantic Chunker Module

AI-powered semantic chunking that uses spaCy for sentence tokenization
and OpenAI for intelligent boundary detection.

Replaces regex-based chunking with semantically-aware chunks that:
- Preserve complete concepts and explanations
- Keep code blocks together
- Split at topic changes
- Include hyper-specific headers for each chunk
"""

import os
import json
from typing import List, Tuple
from dotenv import load_dotenv
import spacy
from pydantic import BaseModel
from openai import OpenAI

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Load spaCy model for sentence tokenization
# Run: python -m spacy download en_core_web_sm
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Downloading spaCy model...")
    from spacy.cli import download
    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")


# Pydantic models for structured LLM response
class ChunkBoundary(BaseModel):
    start_sentence: int  # 0-indexed
    end_sentence: int    # 0-indexed, inclusive
    header: str          # Semantic summary of chunk content


class ChunkResponse(BaseModel):
    chunks: List[ChunkBoundary]


# Prompt template for semantic boundary detection
CHUNKING_PROMPT = """Analyze this text and determine optimal semantic chunk boundaries.

Target: ~500 words per chunk (range: 375-625, soft max 750)

For each chunk provide:
1. start_sentence and end_sentence indices (0-based, inclusive)
2. A hyper-specific header rich with entity names and concepts

Split aggressively at:
- Topic changes
- New concepts or derivations
- Reference frame shifts

Keep together:
- Headings with their following content
- Code blocks
- Bracketed content (images, equations)

SENTENCES (indexed):
{indexed_sentences}

Total sentences: {total_sentences}

Respond ONLY with valid JSON matching this schema:
{{
    "chunks": [
        {{"start_sentence": 0, "end_sentence": 5, "header": "specific header here"}},
        ...
    ]
}}

Ensure all sentences are covered exactly once with no gaps or overlaps."""


def _tokenize_sentences(text: str) -> List[str]:
    """
    Split text into sentences using spaCy.

    Args:
        text: Input text to tokenize

    Returns:
        List of sentence strings
    """
    doc = nlp(text)
    sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
    return sentences


def _count_words(text: str) -> int:
    """Count words in text."""
    return len(text.split())


def _group_sentences_into_blocks(sentences: List[str], target_words: int = 2000) -> List[Tuple[int, int]]:
    """
    Group sentences into blocks of approximately target_words.

    Args:
        sentences: List of sentences
        target_words: Target word count per block (default 2000)

    Returns:
        List of (start_idx, end_idx) tuples for each block
    """
    blocks = []
    current_start = 0
    current_word_count = 0

    for i, sentence in enumerate(sentences):
        sentence_words = _count_words(sentence)

        # If adding this sentence exceeds target and we have content, create block
        if current_word_count + sentence_words > target_words and current_word_count > 0:
            blocks.append((current_start, i - 1))
            current_start = i
            current_word_count = sentence_words
        else:
            current_word_count += sentence_words

    # Don't forget the last block
    if current_start < len(sentences):
        blocks.append((current_start, len(sentences) - 1))

    return blocks


def _get_semantic_boundaries(sentences: List[str], start_offset: int = 0) -> List[ChunkBoundary]:
    """
    Send sentences to LLM for semantic boundary detection.

    Args:
        sentences: List of sentences to analyze
        start_offset: Offset to add to sentence indices (for block processing)

    Returns:
        List of ChunkBoundary objects with adjusted indices
    """
    # Create indexed sentence list for the prompt
    indexed_sentences = "\n".join(
        f"[{i}] {sent}" for i, sent in enumerate(sentences)
    )

    prompt = CHUNKING_PROMPT.format(
        indexed_sentences=indexed_sentences,
        total_sentences=len(sentences)
    )

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "You are a text analysis expert. Return only valid JSON, no markdown."
                },
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=4096
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
        chunk_response = ChunkResponse(**data)

        # Adjust indices by offset
        adjusted_chunks = []
        for chunk in chunk_response.chunks:
            adjusted_chunks.append(ChunkBoundary(
                start_sentence=chunk.start_sentence + start_offset,
                end_sentence=chunk.end_sentence + start_offset,
                header=chunk.header
            ))

        return adjusted_chunks

    except Exception as e:
        print(f"Error getting semantic boundaries: {e}")
        # Fallback: return single chunk for entire block
        return [ChunkBoundary(
            start_sentence=start_offset,
            end_sentence=start_offset + len(sentences) - 1,
            header="Content section"
        )]


def chunk_semantically(text: str, target_words: int = 500) -> List[Tuple[str, str]]:
    """
    Chunk text using AI-powered semantic boundary detection.

    Args:
        text: Input text to chunk
        target_words: Target words per chunk (default 500, range 375-625)

    Returns:
        List of (header, chunk_text) tuples
    """
    print("Tokenizing text into sentences...")
    sentences = _tokenize_sentences(text)
    total_sentences = len(sentences)
    print(f"Found {total_sentences} sentences")

    if total_sentences == 0:
        return []

    # Group sentences into ~2000 word blocks for LLM processing
    print("Grouping sentences into blocks...")
    blocks = _group_sentences_into_blocks(sentences, target_words=2000)
    print(f"Created {len(blocks)} blocks for processing")

    all_boundaries: List[ChunkBoundary] = []

    # Process each block
    for block_idx, (start_idx, end_idx) in enumerate(blocks):
        block_sentences = sentences[start_idx:end_idx + 1]
        print(f"Processing block {block_idx + 1}/{len(blocks)} ({len(block_sentences)} sentences)...")

        # Get semantic boundaries for this block
        boundaries = _get_semantic_boundaries(block_sentences, start_offset=start_idx)
        all_boundaries.extend(boundaries)

    # Handle carry-over: merge any overlapping or adjacent small chunks
    all_boundaries = _merge_adjacent_small_chunks(all_boundaries, sentences, min_words=200)

    # Convert boundaries to (header, text) tuples
    result = []
    for boundary in all_boundaries:
        chunk_sentences = sentences[boundary.start_sentence:boundary.end_sentence + 1]
        chunk_text = " ".join(chunk_sentences)
        result.append((boundary.header, chunk_text))

    print(f"Created {len(result)} semantic chunks")
    return result


def _merge_adjacent_small_chunks(
    boundaries: List[ChunkBoundary],
    sentences: List[str],
    min_words: int = 200
) -> List[ChunkBoundary]:
    """
    Merge adjacent chunks that are too small.

    Args:
        boundaries: List of chunk boundaries
        sentences: All sentences
        min_words: Minimum words per chunk

    Returns:
        Merged list of boundaries
    """
    if len(boundaries) <= 1:
        return boundaries

    merged = []
    current = boundaries[0]

    for next_boundary in boundaries[1:]:
        current_text = " ".join(sentences[current.start_sentence:current.end_sentence + 1])
        current_words = _count_words(current_text)

        if current_words < min_words:
            # Merge with next chunk
            current = ChunkBoundary(
                start_sentence=current.start_sentence,
                end_sentence=next_boundary.end_sentence,
                header=f"{current.header}; {next_boundary.header}"
            )
        else:
            merged.append(current)
            current = next_boundary

    # Don't forget the last chunk
    merged.append(current)

    return merged


def chunk_semantically_from_pdf(pdf_path: str, target_words: int = 500) -> List[Tuple[str, str]]:
    """
    Extract text from PDF and chunk semantically.

    Args:
        pdf_path: Path to PDF file
        target_words: Target words per chunk

    Returns:
        List of (header, chunk_text) tuples
    """
    import pdfplumber

    print(f"Extracting text from: {pdf_path}")
    with pdfplumber.open(pdf_path) as pdf:
        text = " ".join(p.extract_text() or "" for p in pdf.pages)

    return chunk_semantically(text, target_words)


# CLI interface
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        # Use default from .env
        pdf_path = os.getenv("PDF_PATH")
        if not pdf_path:
            print("Usage: python semantic_chunker.py <pdf_path> [target_words]")
            sys.exit(1)
    else:
        pdf_path = sys.argv[1]

    target_words = int(sys.argv[2]) if len(sys.argv) > 2 else 500

    chunks = chunk_semantically_from_pdf(pdf_path, target_words)

    print("\n" + "=" * 60)
    print(f"SEMANTIC CHUNKS ({len(chunks)} total)")
    print("=" * 60)

    for i, (header, text) in enumerate(chunks):
        word_count = _count_words(text)
        print(f"\n--- Chunk {i + 1} ({word_count} words) ---")
        print(f"HEADER: {header}")
        print(f"TEXT: {text[:300]}..." if len(text) > 300 else f"TEXT: {text}")
