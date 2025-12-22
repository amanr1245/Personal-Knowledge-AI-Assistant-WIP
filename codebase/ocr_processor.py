"""
OCR Processor Module for RAG Project

Uses PyMuPDF (fitz) for PDF handling and OpenAI GPT-4 Vision for image-to-text.

Strategies:
- AUTO: Chooses STRICT or RELAXED based on word count per page
- STRICT: Renders full page at 150 DPI, extracts text via Vision API
- RELAXED: Uses pdfplumber text + extracts embedded images via Vision API
"""

import os
import base64
from enum import Enum
from typing import List, Tuple
from dotenv import load_dotenv
import fitz  # PyMuPDF
import pdfplumber
from openai import OpenAI

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


class OcrStrategy(Enum):
    """OCR processing strategy for PDF pages."""
    AUTO = "auto"
    STRICT = "strict"
    RELAXED = "relaxed"


def _image_to_base64(image_bytes: bytes) -> str:
    """Convert image bytes to base64 string."""
    return base64.b64encode(image_bytes).decode("utf-8")


def _extract_text_with_vision(image_base64: str, prompt: str) -> str:
    """
    Send an image to OpenAI Vision API and extract text.
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{image_base64}",
                                "detail": "high"
                            }
                        }
                    ]
                }
            ],
            max_tokens=4096
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Vision API error: {e}")
        return ""


def _render_page_to_image(page: fitz.Page, dpi: int = 150) -> bytes:
    """Render a PDF page to PNG image bytes at specified DPI."""
    pixmap = page.get_pixmap(dpi=dpi)
    return pixmap.tobytes("png")


def _extract_embedded_images(page: fitz.Page) -> List[Tuple[bytes, int]]:
    """
    Extract embedded images from a PDF page.
    Returns list of (image_bytes, image_index) tuples.
    """
    images = []
    image_list = page.get_images(full=True)

    for img_index, img_info in enumerate(image_list):
        xref = img_info[0]
        try:
            base_image = page.parent.extract_image(xref)
            image_bytes = base_image["image"]
            images.append((image_bytes, img_index))
        except Exception as e:
            print(f"Error extracting image {img_index}: {e}")
            continue

    return images


def resolve_ocr_strategy(page_text: str, threshold: int = 50) -> OcrStrategy:
    """Determine OCR strategy based on word count."""
    word_count = len(page_text.split())
    if word_count < threshold:
        return OcrStrategy.STRICT
    return OcrStrategy.RELAXED


def _process_page_strict(pdf_doc: fitz.Document, page_num: int) -> str:
    """
    STRICT mode: Render full page at 150 DPI and extract text via Vision API.
    """
    page = pdf_doc[page_num]
    image_bytes = _render_page_to_image(page, dpi=150)
    image_base64 = _image_to_base64(image_bytes)

    prompt = "Extract all text from this document page exactly as written. Preserve formatting and structure."
    return _extract_text_with_vision(image_base64, prompt)


def _process_page_relaxed(pdf_path: str, pdf_doc: fitz.Document, page_num: int) -> str:
    """
    RELAXED mode: Keep pdfplumber text, extract embedded images via Vision API.
    Image text is inserted as [image: extracted text here].
    """
    # Extract text using pdfplumber
    with pdfplumber.open(pdf_path) as pdf:
        if page_num < len(pdf.pages):
            page_text = pdf.pages[page_num].extract_text() or ""
        else:
            page_text = ""

    # Extract embedded images using PyMuPDF
    page = pdf_doc[page_num]
    embedded_images = _extract_embedded_images(page)

    if not embedded_images:
        return page_text

    # Process each embedded image with Vision API
    image_texts = []
    for image_bytes, img_index in embedded_images:
        image_base64 = _image_to_base64(image_bytes)
        prompt = (
            "Extract any text from this image. If it's a diagram or figure, "
            "describe its content and any labels. Be concise."
        )
        image_text = _extract_text_with_vision(image_base64, prompt)
        if image_text:
            image_texts.append(f"[image: {image_text}]")

    # Combine text with image descriptions
    if image_texts:
        return page_text + "\n\n" + "\n\n".join(image_texts)
    return page_text


def process_pdf_with_ocr(
    pdf_path: str,
    strategy: OcrStrategy = OcrStrategy.AUTO,
    threshold: int = 50
) -> List[str]:
    """
    Process a PDF file with OCR capabilities.

    Args:
        pdf_path: Path to the PDF file
        strategy: OCR strategy (AUTO, STRICT, or RELAXED)
        threshold: Word count threshold for AUTO strategy (default 50)

    Returns:
        List of extracted text strings, one per page
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    pdf_doc = fitz.open(pdf_path)
    num_pages = len(pdf_doc)

    print(f"Processing PDF: {pdf_path}")
    print(f"Total pages: {num_pages}, Strategy: {strategy.value}")

    extracted_pages = []

    for page_num in range(num_pages):
        print(f"Processing page {page_num + 1}/{num_pages}...", end=" ")

        if strategy == OcrStrategy.STRICT:
            page_text = _process_page_strict(pdf_doc, page_num)
            mode_used = "STRICT"

        elif strategy == OcrStrategy.RELAXED:
            page_text = _process_page_relaxed(pdf_path, pdf_doc, page_num)
            mode_used = "RELAXED"

        else:  # AUTO strategy
            # Get pdfplumber text to check word count
            with pdfplumber.open(pdf_path) as pdf:
                if page_num < len(pdf.pages):
                    initial_text = pdf.pages[page_num].extract_text() or ""
                else:
                    initial_text = ""

            resolved_strategy = resolve_ocr_strategy(initial_text, threshold)

            if resolved_strategy == OcrStrategy.STRICT:
                page_text = _process_page_strict(pdf_doc, page_num)
                mode_used = "STRICT"
            else:
                page_text = _process_page_relaxed(pdf_path, pdf_doc, page_num)
                mode_used = "RELAXED"

        extracted_pages.append(page_text)
        print(f"[{mode_used}] {len(page_text)} chars")

    pdf_doc.close()
    print(f"Processing complete.")
    return extracted_pages


def process_pdf_with_ocr_combined(
    pdf_path: str,
    strategy: OcrStrategy = OcrStrategy.AUTO,
    threshold: int = 50
) -> str:
    """Process a PDF and return all text combined into a single string."""
    pages = process_pdf_with_ocr(pdf_path, strategy, threshold)
    return "\n\n".join(pages)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python ocr_processor.py <pdf_path> [strategy] [threshold]")
        print("  strategy: auto (default), strict, relaxed")
        print("  threshold: word count threshold for auto (default 50)")
        sys.exit(1)

    pdf_path = sys.argv[1]
    strategy_str = sys.argv[2].lower() if len(sys.argv) > 2 else "auto"
    strategy_map = {"auto": OcrStrategy.AUTO, "strict": OcrStrategy.STRICT, "relaxed": OcrStrategy.RELAXED}
    strategy = strategy_map.get(strategy_str, OcrStrategy.AUTO)
    threshold = int(sys.argv[3]) if len(sys.argv) > 3 else 50

    pages = process_pdf_with_ocr(pdf_path, strategy, threshold)

    print("\n" + "=" * 60)
    for i, page_text in enumerate(pages):
        print(f"\n--- Page {i + 1} ---")
        print(page_text[:500] + "..." if len(page_text) > 500 else page_text)
