"""
Processors Package

Provides parallel processing capabilities for compute-intensive operations:
- VisionProcessor: Parallel image processing with GPT-4 Vision
- EmbedProcessor: Parallel embedding generation

Usage:
    from processors import VisionProcessor, EmbedProcessor

    # Parallel vision processing
    processor = VisionProcessor(max_workers=8)
    results = processor.process_images(images, prompt)

    # Parallel embedding
    embed_processor = EmbedProcessor(max_workers=8, batch_size=100)
    embeddings = embed_processor.embed_batch_parallel(texts)
"""

from .vision_processor import (
    VisionProcessor,
    VisionRequest,
    process_images_parallel,
    MAX_WORKERS as VISION_MAX_WORKERS
)

from .embed_processor import (
    EmbedProcessor,
    embed_texts_parallel,
    MAX_WORKERS as EMBED_MAX_WORKERS,
    BATCH_SIZE as EMBED_BATCH_SIZE
)

__all__ = [
    # Vision processing
    'VisionProcessor',
    'VisionRequest',
    'process_images_parallel',
    'VISION_MAX_WORKERS',
    # Embedding processing
    'EmbedProcessor',
    'embed_texts_parallel',
    'EMBED_MAX_WORKERS',
    'EMBED_BATCH_SIZE',
]
