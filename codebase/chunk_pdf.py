import os
import pdfplumber
import re
from dotenv import load_dotenv

load_dotenv()


def chunk_pdf(path, max_len=500, overlap=100):
    with pdfplumber.open(path) as pdf:
        text = " ".join(p.extract_text() or "" for p in pdf.pages)

    # Clean up whitespace
    text = re.sub(r"\s+", " ", text).strip()

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

    return chunks


if __name__ == "__main__":
    pdf_path = os.getenv("PDF_PATH")
    chunks = chunk_pdf(pdf_path)
    print(f"{len(chunks)} chunks generated")
    for i, c in enumerate(chunks):
        print(f"Chunk {i}:")
        print(c)
        print("-" * 80)
