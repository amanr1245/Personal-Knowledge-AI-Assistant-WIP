import os
import pdfplumber
import re
from dotenv import load_dotenv
from ocr_processor import process_pdf_with_ocr_combined, OcrStrategy
import cache_manager

load_dotenv()


def chunk_pdf(path, max_len=500, overlap=100, use_ocr=False, ocr_strategy=OcrStrategy.AUTO, ocr_threshold=50):
    """
    Chunk a PDF into text segments.

    Args:
        path: Path to the PDF file
        max_len: Maximum chunk length in characters (default 500)
        overlap: Character overlap between chunks (default 100)
        use_ocr: Whether to use OCR processing (default False)
        ocr_strategy: OCR strategy when use_ocr=True (default AUTO)
        ocr_threshold: Word count threshold for AUTO strategy (default 50)

    Returns:
        List of text chunks
    """
    if use_ocr:
        # Use OCR processor for text extraction
        text = process_pdf_with_ocr_combined(path, ocr_strategy, ocr_threshold)
    else:
        # Use pdfplumber for standard text extraction
        with pdfplumber.open(path) as pdf:
            text = " ".join(p.extract_text() or "" for p in pdf.pages)

    # Clean up whitespace
    text = re.sub(r"\s+", " ", text).strip()

    # Check cache for chunks (by document text hash + params)
    cache_content = f"{text}|{max_len}|{overlap}"
    cached_chunks = cache_manager.get_cached_chunks(cache_content)
    if cached_chunks is not None:
        print(f"Cache hit: {len(cached_chunks)} chunks from cache")
        return cached_chunks

    # Split into sentences (break on . ! ?)
    sentences = re.split(r'(?<=[.!?])\s+', text)

    chunks = []
    current_chunk = ""

    for sentence in sentences:
        # If adding this sentence exceeds max_len, save current chunk
        if len(current_chunk) + len(sentence) > max_len and current_chunk:
            chunks.append(current_chunk.strip())
            # Start new chunk with overlap from previous
            overlap_text = current_chunk[-overlap:] if len(current_chunk) > overlap else current_chunk
            current_chunk = overlap_text + " " + sentence
        else:
            current_chunk += " " + sentence

    # Don't forget the last chunk
    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    # Cache the result
    cache_manager.cache_chunks(cache_content, chunks)

    return chunks


if __name__ == "__main__":
    import sys

    pdf_path = os.getenv("PDF_PATH")
    use_ocr = "--ocr" in sys.argv

    if use_ocr:
        print("Using OCR processing...")
        chunks = chunk_pdf(pdf_path, use_ocr=True)
    else:
        chunks = chunk_pdf(pdf_path)

    print(f"{len(chunks)} chunks generated")
    for i, c in enumerate(chunks):
        print(f"Chunk {i}:")
        print(c)
        print("-" * 80)
