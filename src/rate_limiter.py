"""
Rate Limiter - Prevents API throttling by limiting request frequency
"""
import time
import logging

logger = logging.getLogger(__name__)


class RateLimiter:
    """
    Simple rate limiter that enforces minimum time between operations

    Usage:
        limiter = RateLimiter(calls_per_second=2)

        for item in items:
            limiter.wait()  # Will sleep if needed
            make_api_call(item)
    """

    def __init__(self, calls_per_second: float = 2.0):
        """
        Initialize rate limiter

        Args:
            calls_per_second: Maximum number of calls allowed per second
        """
        if calls_per_second <= 0:
            raise ValueError("calls_per_second must be positive")

        self.min_interval = 1.0 / calls_per_second
        self.last_call = 0
        self.total_waits = 0
        self.total_wait_time = 0

        logger.debug(f"Rate limiter initialized: max {calls_per_second} calls/sec (min {self.min_interval:.3f}s between calls)")

    def wait(self):
        """
        Wait if necessary to maintain rate limit
        Returns immediately if enough time has passed since last call
        """
        now = time.time()
        elapsed = now - self.last_call

        if elapsed < self.min_interval:
            sleep_time = self.min_interval - elapsed
            time.sleep(sleep_time)
            self.total_waits += 1
            self.total_wait_time += sleep_time

        self.last_call = time.time()

    def reset(self):
        """Reset the rate limiter state"""
        self.last_call = 0
        self.total_waits = 0
        self.total_wait_time = 0

    def get_stats(self) -> dict:
        """Get statistics about rate limiting"""
        return {
            'total_waits': self.total_waits,
            'total_wait_time': self.total_wait_time,
            'avg_wait_time': self.total_wait_time / self.total_waits if self.total_waits > 0 else 0
        }
