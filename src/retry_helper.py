"""
Retry Helper - Implements exponential backoff for API calls

Provides a decorator that automatically retries failed requests with increasing delays
"""
import time
import logging
from functools import wraps
from typing import Callable, Type, Tuple

logger = logging.getLogger(__name__)


def retry_with_backoff(
    max_retries: int = 3,
    initial_delay: float = 1.0,
    backoff_multiplier: float = 2.0,
    max_delay: float = 30.0,
    exceptions: Tuple[Type[Exception], ...] = (Exception,)
):
    """
    Decorator that retries a function with exponential backoff

    Args:
        max_retries: Maximum number of retry attempts
        initial_delay: Initial delay in seconds before first retry
        backoff_multiplier: Multiplier for delay after each retry
        max_delay: Maximum delay between retries in seconds
        exceptions: Tuple of exception types to catch and retry

    Returns:
        Decorated function that retries on failure

    Example:
        @retry_with_backoff(max_retries=3, initial_delay=1.0)
        def make_api_call():
            return requests.get(url)
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            delay = initial_delay
            last_exception = None

            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e

                    # Don't retry on the last attempt
                    if attempt == max_retries:
                        logger.error(
                            f"{func.__name__} failed after {max_retries} retries: {e}"
                        )
                        raise

                    # Log the retry
                    logger.warning(
                        f"{func.__name__} failed (attempt {attempt + 1}/{max_retries + 1}), "
                        f"retrying in {delay:.1f}s: {e}"
                    )

                    # Sleep with exponential backoff
                    time.sleep(delay)
                    delay = min(delay * backoff_multiplier, max_delay)

            # This should never be reached, but just in case
            if last_exception:
                raise last_exception

        return wrapper
    return decorator


class RetryableError(Exception):
    """Base exception for errors that should trigger a retry"""
    pass


class RateLimitError(RetryableError):
    """Raised when rate limit is exceeded (429 status code)"""
    pass


class ServerError(RetryableError):
    """Raised when server returns 5xx error"""
    pass


class NetworkError(RetryableError):
    """Raised when network connection fails"""
    pass
