"""
Utility functions for the StrapTo local server.

This module provides shared functionality used across different components
of the server, including:
- Data conversion and validation
- WebRTC helpers
- Async utilities
- Error handling
"""

import json
import asyncio
import logging
from typing import Any, Dict, Optional, TypeVar, Callable, Awaitable
from datetime import datetime, date
from functools import wraps
import uuid

logger = logging.getLogger(__name__)

T = TypeVar('T')

def generate_client_id() -> str:
    """Generate a unique client identifier."""
    return str(uuid.uuid4())

def sanitize_json(data: Any) -> Dict[str, Any]:
    """
    Ensure data is JSON-serializable and handle common edge cases.
    
    Args:
        data: Input data to sanitize
        
    Returns:
        Dict containing JSON-safe data
        
    Raises:
        ValueError: If data cannot be converted to JSON-safe format
    """
    def _sanitize(obj: Any) -> Any:
        if isinstance(obj, (str, int, float, bool, type(None))):
            return obj
        elif isinstance(obj, (datetime, date)):
            return obj.isoformat()
        elif isinstance(obj, bytes):
            return obj.decode('utf-8', errors='replace')
        elif isinstance(obj, (list, tuple)):
            return [_sanitize(item) for item in obj]
        elif isinstance(obj, dict):
            return {str(k): _sanitize(v) for k, v in obj.items()}
        else:
            # Try to convert object to string representation
            try:
                return str(obj)
            except Exception as e:
                logger.warning(f"Could not sanitize object of type {type(obj)}: {e}")
                return None

    try:
        if not isinstance(data, dict):
            raise ValueError("Input must be a dictionary")
        sanitized = _sanitize(data)
        # Verify JSON serialization works
        json.dumps(sanitized)
        return sanitized
    except Exception as e:
        raise ValueError(f"Could not convert data to JSON-safe format: {e}")

def retry_async(
    max_attempts: int = 3,
    delay: float = 1.0,
    exponential_backoff: bool = True,
    exceptions: tuple = (Exception,)
) -> Callable[[Callable[..., Awaitable[T]]], Callable[..., Awaitable[T]]]:
    """
    Decorator for retrying async functions with optional exponential backoff.
    
    Args:
        max_attempts: Maximum number of retry attempts
        delay: Base delay between retries in seconds
        exponential_backoff: Whether to use exponential backoff
        exceptions: Tuple of exceptions to catch and retry
        
    Returns:
        Decorated async function with retry logic
    """
    def decorator(func: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> T:
            last_exception: Optional[Exception] = None
            
            for attempt in range(max_attempts):
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    
                    if attempt == max_attempts - 1:
                        raise
                        
                    wait_time = delay * (2 ** attempt if exponential_backoff else 1)
                    logger.warning(
                        f"Attempt {attempt + 1}/{max_attempts} failed: {e}. "
                        f"Retrying in {wait_time:.1f}s..."
                    )
                    
                    await asyncio.sleep(wait_time)
            
            assert last_exception is not None  # for type checker
            raise last_exception
            
        return wrapper
    return decorator

async def cancel_tasks(tasks: set[asyncio.Task]) -> None:
    """
    Safely cancel a set of asyncio tasks.
    
    Args:
        tasks: Set of tasks to cancel
    """
    if not tasks:
        return
        
    for task in tasks:
        if not task.done():
            task.cancel()
            
    await asyncio.gather(*tasks, return_exceptions=True)
    tasks.clear()

def format_error(e: Exception) -> Dict[str, str]:
    """
    Format exception into a consistent error response structure.
    
    Args:
        e: Exception to format
        
    Returns:
        Dict containing error details
    """
    return {
        "error": str(e),
        "type": e.__class__.__name__,
        "timestamp": datetime.utcnow().isoformat()
    }

class RateLimiter:
    """Simple rate limiter using token bucket algorithm."""
    
    def __init__(self, rate: float, burst: int = 1):
        """
        Initialize rate limiter.
        
        Args:
            rate: Tokens per second
            burst: Maximum burst size
        """
        self.rate = rate
        self.burst = burst
        self.tokens = burst
        self.last_update = asyncio.get_event_loop().time()
        self._lock = asyncio.Lock()
        
    async def acquire(self) -> bool:
        """
        Attempt to acquire a token.
        
        Returns:
            bool: True if token was acquired, False if rate limit exceeded
        """
        async with self._lock:
            now = asyncio.get_event_loop().time()
            time_passed = now - self.last_update
            self.tokens = min(
                self.burst,
                self.tokens + time_passed * self.rate
            )
            
            if self.tokens >= 1:
                self.tokens -= 1
                self.last_update = now
                return True
            
            return False