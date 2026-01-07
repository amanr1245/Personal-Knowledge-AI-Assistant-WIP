import os
import re
import hashlib
from dotenv import load_dotenv
from ocr_processor import process_pdf_with_ocr_combined, OcrStrategy
from pdf_extractor import extract_text_from_pdf
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
        # Use PyMuPDF for standard text extraction
        text = extract_text_from_pdf(path)

    # Clean up whitespace
    text = re.sub(r"\s+", " ", text).strip()

    # Check cache for chunks (by document text hash + params)
    text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
    if use_ocr:
        cache_content = f"{text_hash}|{max_len}|{overlap}|ocr|{ocr_strategy}|{ocr_threshold}"
    else:
        cache_content = f"{text_hash}|{max_len}|{overlap}|no_ocr"
    cached_chunks = cache_manager.get_cached_chunks(cache_content)
    if cached_chunks is not None:
        print(f"Cache hit: {len(cached_chunks)} chunks from cache")
        return cached_chunks

    # Split into sentences (break on . ! ?)
    sentences = re.split(r'(?<=[.!?])\s+', text)

    chunks = []
    current_chunk = ""

    for sentence in sentences:
        # Handle overly long sentences by splitting into smaller segments
        if len(sentence) > max_len:
            words = sentence.split()
            segment = ""

            for word in words:
                # Handle extremely long words by splitting them
                if len(word) > max_len:
                    # Process current segment first if exists
                    if segment:
                        if len(current_chunk) + len(segment) + 1 > max_len and current_chunk:
                            chunks.append(current_chunk.strip())
                            overlap_text = current_chunk[-overlap:] if len(current_chunk) > overlap else current_chunk
                            available_space = max_len - len(segment) - 1
                            if available_space <= 0:
                                current_chunk = segment[:max_len].strip()
                            elif len(overlap_text) > available_space:
                                overlap_text = overlap_text[-available_space:]
                                current_chunk = (overlap_text + " " + segment).strip()
                            else:
                                current_chunk = (overlap_text + " " + segment).strip()
                        else:
                            current_chunk = (current_chunk + " " + segment).strip()
                        segment = ""

                    # Split the long word into max_len chunks
                    for i in range(0, len(word), max_len):
                        word_chunk = word[i:i+max_len]
                        if len(current_chunk) + len(word_chunk) + 1 > max_len and current_chunk:
                            chunks.append(current_chunk.strip())
                            overlap_text = current_chunk[-overlap:] if len(current_chunk) > overlap else current_chunk
                            available_space = max_len - len(word_chunk) - 1
                            if available_space <= 0:
                                current_chunk = word_chunk.strip()
                            elif len(overlap_text) > available_space:
                                overlap_text = overlap_text[-available_space:]
                                current_chunk = (overlap_text + " " + word_chunk).strip()
                            else:
                                current_chunk = (overlap_text + " " + word_chunk).strip()
                        else:
                            current_chunk = (current_chunk + " " + word_chunk).strip()
                    continue

                # Check if adding this word would exceed max_len
                test_segment = (segment + " " + word).strip() if segment else word

                if len(test_segment) > max_len and segment:
                    # Process the current segment as a complete unit
                    if len(current_chunk) + len(segment) + 1 > max_len and current_chunk:
                        chunks.append(current_chunk.strip())
                        overlap_text = current_chunk[-overlap:] if len(current_chunk) > overlap else current_chunk

                        # Ensure overlap + segment doesn't exceed max_len
                        available_space = max_len - len(segment) - 1
                        if available_space <= 0:
                            # Segment itself exceeds or equals max_len, use no overlap
                            current_chunk = segment[:max_len].strip()
                        elif len(overlap_text) > available_space:
                            # Trim overlap from the start to fit
                            overlap_text = overlap_text[-available_space:]
                            current_chunk = (overlap_text + " " + segment).strip()
                        else:
                            current_chunk = (overlap_text + " " + segment).strip()
                    else:
                        current_chunk = (current_chunk + " " + segment).strip()

                    segment = word
                else:
                    segment = test_segment

            # Process remaining segment as the sentence
            sentence = segment

        # Normal sentence processing
        if len(current_chunk) + len(sentence) + 1 > max_len and current_chunk:
            chunks.append(current_chunk.strip())
            # Start new chunk with overlap from previous
            overlap_text = current_chunk[-overlap:] if len(current_chunk) > overlap else current_chunk

            # Ensure overlap + sentence doesn't exceed max_len
            available_space = max_len - len(sentence) - 1
            if available_space <= 0:
                # Sentence itself exceeds or equals max_len, use no overlap
                current_chunk = sentence[:max_len].strip()
            elif len(overlap_text) > available_space:
                # Trim overlap from the start to fit
                overlap_text = overlap_text[-available_space:]
                current_chunk = (overlap_text + " " + sentence).strip()
            else:
                current_chunk = (overlap_text + " " + sentence).strip()
        else:
            current_chunk = (current_chunk + " " + sentence).strip()

    # Don't forget the last chunk
    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    # Cache the result
    cache_manager.cache_chunks(cache_content, chunks)

    return chunks


if __name__ == "__main__":
    import sys

    pdf_path = os.getenv("PDF_PATH")
    if not pdf_path:
        print("ERROR: PDF_PATH environment variable is required")
        print("Usage: Set PDF_PATH in .env or run with: PDF_PATH=/path/to/file.pdf python chunk_pdf.py")
        sys.exit(1)

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
