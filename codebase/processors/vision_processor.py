"""
Parallel Vision Processing Module

Provides thread-safe parallel processing of images using GPT-4 Vision API.
Each worker uses its own OpenAI client instance for thread safety.
"""

import os
import concurrent.futures
from typing import List, Tuple, Callable, Optional
from dataclasses import dataclass
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

MAX_WORKERS = int(os.getenv("VISION_MAX_WORKERS", "8"))


@dataclass
class VisionRequest:
    """Request for vision processing."""
    image_bytes: bytes
    prompt: str
    index: int  # For result ordering
    image_id: str = ""  # Optional identifier for logging


class VisionProcessor:
    """
    Thread-safe parallel vision processor.

    Each worker thread creates its own OpenAI client instance
    to ensure thread safety.
    """

    def __init__(self, max_workers: int = None):
        """
        Initialize the vision processor.

        Args:
            max_workers: Maximum parallel workers (default from env or 8)
        """
        self.max_workers = max_workers or MAX_WORKERS
        self.api_key = os.getenv("OPENAI_API_KEY")

    def _create_client(self) -> OpenAI:
        """Create a new OpenAI client for thread-local use."""
        return OpenAI(api_key=self.api_key)

    def _process_single(self, request: VisionRequest) -> str:
        """
        Process a single image with Vision API.

        Creates its own client instance for thread safety.
        """
        import base64
        import sys
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        import cache_manager
        from retry_manager import with_retry

        # Check cache first
        cached = cache_manager.get_cached_ocr(request.image_bytes)
        if cached is not None:
            return cached

        # Create thread-local client
        client = self._create_client()

        # Convert to base64
        image_base64 = base64.b64encode(request.image_bytes).decode('utf-8')

        try:
            response = with_retry(
                client.chat.completions.create,
                model="gpt-4o",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": request.prompt},
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
            text = response.choices[0].message.content.strip()

            # Cache the result
            cache_manager.cache_ocr(request.image_bytes, text)

            return text

        except Exception as e:
            print(f"Vision API error: {e}")
            return ""

    def process_batch(
        self,
        requests: List[VisionRequest],
        progress_callback: Callable[[int, int], None] = None
    ) -> List[Tuple[int, Optional[str]]]:
        """
        Process multiple images in parallel.

        Args:
            requests: List of VisionRequest objects
            progress_callback: Optional callback(completed, total) for progress

        Returns:
            List of (index, text) tuples sorted by original index
        """
        if not requests:
            return []

        results = []
        completed = 0
        total = len(requests)

        print(f"Processing {total} images with {self.max_workers} workers...")

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_request = {
                executor.submit(self._process_single, req): req
                for req in requests
            }

            for future in concurrent.futures.as_completed(future_to_request):
                req = future_to_request[future]
                completed += 1

                try:
                    text = future.result()
                    results.append((req.index, text))
                    status = "OK" if text else "EMPTY"
                except Exception as e:
                    print(f"Vision error for image {req.index}: {e}")
                    results.append((req.index, None))
                    status = "ERROR"

                # Progress update
                if progress_callback:
                    progress_callback(completed, total)
                else:
                    print(f"  [{completed}/{total}] Image {req.index}: {status}")

        # Sort by original index
        results.sort(key=lambda x: x[0])
        return results

    def process_images(
        self,
        images: List[Tuple[bytes, str]],
        prompt: str = "Extract all text from this image. Preserve formatting."
    ) -> List[str]:
        """
        Convenience method to process a list of images.

        Args:
            images: List of (image_bytes, image_id) tuples
            prompt: Prompt to use for all images

        Returns:
            List of extracted text strings in same order as input
        """
        requests = [
            VisionRequest(
                image_bytes=img_bytes,
                prompt=prompt,
                index=i,
                image_id=img_id
            )
            for i, (img_bytes, img_id) in enumerate(images)
        ]

        results = self.process_batch(requests)
        return [text or "" for _, text in results]


# Module-level convenience function
def process_images_parallel(
    images: List[Tuple[bytes, str]],
    prompt: str = "Extract all text from this image. Preserve formatting.",
    max_workers: int = None
) -> List[str]:
    """
    Process multiple images in parallel.

    Args:
        images: List of (image_bytes, image_id) tuples
        prompt: Prompt for Vision API
        max_workers: Max parallel workers

    Returns:
        List of extracted text strings
    """
    processor = VisionProcessor(max_workers)
    return processor.process_images(images, prompt)


if __name__ == "__main__":
    print(f"VisionProcessor configured with MAX_WORKERS={MAX_WORKERS}")
    print("Usage: from processors import VisionProcessor")
