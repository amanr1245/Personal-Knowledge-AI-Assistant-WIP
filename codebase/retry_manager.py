"""
Retry Manager for RAG System

Provides exponential backoff retry logic for all external API calls:
- OpenAI API (embeddings, chat, vision)
- Database operations (PostgreSQL)

Handles transient failures gracefully with configurable retry parameters.
"""

import os
import time
import logging
from functools import wraps
from typing import Callable, Any, Tuple, Type
from dotenv import load_dotenv

# Optional imports - handle gracefully if not installed
try:
    import openai
    OPENAI_ERRORS: Tuple[Type[Exception], ...] = (
        openai.APIError,
        openai.APIConnectionError,
        openai.RateLimitError,
        openai.APITimeoutError,
    )
except ImportError:
    OPENAI_ERRORS = ()

try:
    import psycopg2
    DB_ERRORS: Tuple[Type[Exception], ...] = (
        psycopg2.OperationalError,
        psycopg2.InterfaceError,
    )
except ImportError:
    DB_ERRORS = ()

try:
    import requests
    REQUEST_ERRORS: Tuple[Type[Exception], ...] = (
        requests.RequestException,
        requests.ConnectionError,
        requests.Timeout,
    )
except ImportError:
    REQUEST_ERRORS = ()

load_dotenv()

# Configuration from .env with defaults
RETRY_MAX = int(os.getenv("RETRY_MAX", "5"))
RETRY_DELAY = float(os.getenv("RETRY_DELAY", "1.0"))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("retry_manager")


# All retryable errors
RETRYABLE_ERRORS = OPENAI_ERRORS + DB_ERRORS + REQUEST_ERRORS + (
    ConnectionError,
    TimeoutError,
    OSError,
)


class RetryManager:
    """
    Manages retry logic with exponential backoff for API calls.

    Usage:
        retry_mgr = RetryManager()
        result = retry_mgr.retry(api_function, arg1, arg2, kwarg1=value)

    Or as decorator:
        @RetryManager().decorator
        def my_api_function():
            ...
    """

    def __init__(
        self,
        max_retries: int = None,
        base_delay: float = None,
        max_delay: float = 60.0,
        backoff: float = 2.0,
        retryable_errors: Tuple[Type[Exception], ...] = None
    ):
        """
        Initialize RetryManager.

        Args:
            max_retries: Maximum number of retry attempts (default: RETRY_MAX from .env)
            base_delay: Initial delay between retries in seconds (default: RETRY_DELAY from .env)
            max_delay: Maximum delay between retries in seconds (default: 60.0)
            backoff: Multiplier for exponential backoff (default: 2.0)
            retryable_errors: Tuple of exception types to retry on
        """
        self.max_retries = max_retries if max_retries is not None else RETRY_MAX
        self.base_delay = base_delay if base_delay is not None else RETRY_DELAY
        self.max_delay = max_delay
        self.backoff = backoff
        self.retryable_errors = retryable_errors or RETRYABLE_ERRORS

    def retry(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute a function with retry logic.

        Args:
            func: Function to execute
            *args: Positional arguments for the function
            **kwargs: Keyword arguments for the function

        Returns:
            Result of the function call

        Raises:
            The last exception if all retries fail
        """
        last_exception = None
        delay = self.base_delay

        for attempt in range(self.max_retries + 1):
            try:
                return func(*args, **kwargs)
            except self.retryable_errors as e:
                last_exception = e

                if attempt < self.max_retries:
                    # Log retry attempt
                    logger.warning(
                        f"Attempt {attempt + 1}/{self.max_retries + 1} failed: "
                        f"{type(e).__name__}: {str(e)[:100]}. "
                        f"Retrying in {delay:.1f}s..."
                    )

                    # Wait before retry
                    time.sleep(delay)

                    # Exponential backoff with max cap
                    delay = min(delay * self.backoff, self.max_delay)
                else:
                    # Final attempt failed
                    logger.error(
                        f"All {self.max_retries + 1} attempts failed. "
                        f"Last error: {type(e).__name__}: {str(e)}"
                    )

        # Re-raise the last exception
        raise last_exception

    def decorator(self, func: Callable) -> Callable:
        """
        Decorator version of retry logic.

        Usage:
            @RetryManager().decorator
            def my_function():
                ...
        """
        @wraps(func)
        def wrapper(*args, **kwargs):
            return self.retry(func, *args, **kwargs)
        return wrapper


# Global retry manager instance with default settings
default_retry_manager = RetryManager()


def with_retry(func: Callable, *args, **kwargs) -> Any:
    """
    Convenience function to execute with default retry logic.

    Usage:
        result = with_retry(openai_client.embeddings.create, model="...", input="...")
    """
    return default_retry_manager.retry(func, *args, **kwargs)


def retry_decorator(
    max_retries: int = None,
    base_delay: float = None,
    max_delay: float = 60.0,
    backoff: float = 2.0
):
    """
    Decorator factory for retry logic with custom parameters.

    Usage:
        @retry_decorator(max_retries=3, base_delay=0.5)
        def my_function():
            ...
    """
    manager = RetryManager(
        max_retries=max_retries,
        base_delay=base_delay,
        max_delay=max_delay,
        backoff=backoff
    )
    return manager.decorator


# Specialized retry functions for different API types

def retry_openai(func: Callable, *args, **kwargs) -> Any:
    """Retry wrapper specifically for OpenAI API calls."""
    manager = RetryManager(retryable_errors=OPENAI_ERRORS or RETRYABLE_ERRORS)
    return manager.retry(func, *args, **kwargs)


def retry_database(func: Callable, *args, **kwargs) -> Any:
    """Retry wrapper specifically for database operations."""
    manager = RetryManager(retryable_errors=DB_ERRORS or RETRYABLE_ERRORS)
    return manager.retry(func, *args, **kwargs)


# CLI for testing
if __name__ == "__main__":
    print(f"Retry Manager Configuration:")
    print(f"  RETRY_MAX: {RETRY_MAX}")
    print(f"  RETRY_DELAY: {RETRY_DELAY}s")
    print(f"  Retryable errors: {len(RETRYABLE_ERRORS)} types")

    # Test with a failing function
    print("\nTesting retry logic with simulated failures...")

    call_count = 0

    def flaky_function():
        global call_count
        call_count += 1
        if call_count < 3:
            raise ConnectionError(f"Simulated failure #{call_count}")
        return "Success!"

    try:
        result = with_retry(flaky_function)
        print(f"Result: {result} (after {call_count} attempts)")
    except Exception as e:
        print(f"Failed: {e}")
