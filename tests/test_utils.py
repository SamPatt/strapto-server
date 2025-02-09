"""
Tests for StrapTo server utility functions.

This module contains tests for both synchronous and asynchronous utilities,
including data handling, retries, rate limiting, and error formatting.
"""

import pytest
import asyncio
import json
from datetime import datetime, date
from unittest.mock import Mock, patch

from strapto_server.utils import (
    generate_client_id,
    sanitize_json,
    retry_async,
    cancel_tasks,
    format_error,
    RateLimiter
)

# Test client ID generation
def test_generate_client_id():
    """Test that client IDs are unique and properly formatted."""
    id1 = generate_client_id()
    id2 = generate_client_id()
    
    assert isinstance(id1, str)
    assert len(id1) > 0
    assert id1 != id2  # IDs should be unique

# Test JSON sanitization
def test_sanitize_json_basic_types():
    """Test sanitization of basic Python types."""
    data = {
        "string": "test",
        "integer": 42,
        "float": 3.14,
        "boolean": True,
        "none": None,
        "list": [1, 2, 3],
        "nested": {"key": "value"}
    }
    
    sanitized = sanitize_json(data)
    # Should be able to JSON serialize
    assert json.dumps(sanitized)
    assert sanitized == data  # Basic types should remain unchanged

def test_sanitize_json_datetime():
    """Test sanitization of datetime objects."""
    now = datetime.now()
    today = date.today()
    data = {
        "datetime": now,
        "date": today
    }
    
    sanitized = sanitize_json(data)
    assert isinstance(sanitized["datetime"], str)
    assert isinstance(sanitized["date"], str)
    assert sanitized["datetime"] == now.isoformat()
    assert sanitized["date"] == today.isoformat()

def test_sanitize_json_bytes():
    """Test sanitization of bytes objects."""
    data = {
        "bytes": b"test string",
        "nested": {"bytes": b"nested bytes"}
    }
    
    sanitized = sanitize_json(data)
    assert isinstance(sanitized["bytes"], str)
    assert sanitized["bytes"] == "test string"
    assert isinstance(sanitized["nested"]["bytes"], str)

def test_sanitize_json_custom_object():
    """Test sanitization of custom objects."""
    class CustomClass:
        def __str__(self):
            return "custom_string"
    
    data = {
        "custom": CustomClass()
    }
    
    sanitized = sanitize_json(data)
    assert sanitized["custom"] == "custom_string"

def test_sanitize_json_invalid():
    """Test handling of objects that can't be sanitized."""
    class BadClass:
        def __str__(self):
            raise Exception("Can't convert to string")
    
    data = {
        "bad": BadClass()
    }
    
    sanitized = sanitize_json(data)
    assert sanitized["bad"] is None

# Test async retry decorator
@pytest.mark.asyncio
async def test_retry_async_success():
    """Test successful execution without retries."""
    mock_func = Mock()
    mock_func.return_value = asyncio.Future()
    mock_func.return_value.set_result("success")
    
    @retry_async(max_attempts=3)
    async def test_func():
        return await mock_func()
    
    result = await test_func()
    assert result == "success"
    assert mock_func.call_count == 1

@pytest.mark.asyncio
async def test_retry_async_with_retries():
    """Test retrying on failure."""
    attempts = 0
    
    @retry_async(max_attempts=3, delay=0.1)
    async def test_func():
        nonlocal attempts
        attempts += 1
        if attempts < 2:
            raise ValueError("Temporary error")
        return "success"
    
    result = await test_func()
    assert result == "success"
    assert attempts == 2

@pytest.mark.asyncio
async def test_retry_async_max_attempts():
    """Test that max attempts is respected."""
    @retry_async(max_attempts=3, delay=0.1)
    async def test_func():
        raise ValueError("Persistent error")
    
    with pytest.raises(ValueError):
        await test_func()

# Test task cancellation
@pytest.mark.asyncio
async def test_cancel_tasks():
    """Test cancellation of asyncio tasks."""
    async def dummy_task():
        try:
            await asyncio.sleep(10)
        except asyncio.CancelledError:
            pass
    
    # Create some tasks
    tasks = {
        asyncio.create_task(dummy_task()),
        asyncio.create_task(dummy_task())
    }
    
    # Cancel them
    await cancel_tasks(tasks)
    
    # Verify all tasks are cancelled and set is empty
    assert len(tasks) == 0
    
@pytest.mark.asyncio
async def test_cancel_tasks_empty():
    """Test cancellation with empty task set."""
    tasks = set()
    await cancel_tasks(tasks)  # Should not raise
    assert len(tasks) == 0

# Test error formatting
def test_format_error():
    """Test error formatting."""
    try:
        raise ValueError("Test error")
    except Exception as e:
        error_dict = format_error(e)
        
        assert isinstance(error_dict, dict)
        assert error_dict["error"] == "Test error"
        assert error_dict["type"] == "ValueError"
        assert "timestamp" in error_dict
        # Verify timestamp is ISO format
        datetime.fromisoformat(error_dict["timestamp"])

# Test rate limiter
@pytest.mark.asyncio
async def test_rate_limiter_basic():
    """Test basic rate limiting functionality."""
    limiter = RateLimiter(rate=10, burst=1)
    
    # First attempt should succeed
    assert await limiter.acquire()
    
    # Immediate second attempt should fail
    assert not await limiter.acquire()

@pytest.mark.asyncio
async def test_rate_limiter_recovery():
    """Test rate limiter token recovery."""
    limiter = RateLimiter(rate=10, burst=1)
    
    # Use token
    assert await limiter.acquire()
    
    # Wait for token recovery
    await asyncio.sleep(0.2)  # Wait for 2 tokens worth of time
    
    # Should be able to acquire again
    assert await limiter.acquire()

@pytest.mark.asyncio
async def test_rate_limiter_burst():
    """Test burst functionality."""
    limiter = RateLimiter(rate=10, burst=2)
    
    # Should be able to use burst amount immediately
    assert await limiter.acquire()
    assert await limiter.acquire()
    
    # But not more than burst
    assert not await limiter.acquire()

@pytest.mark.asyncio
async def test_rate_limiter_concurrent():
    """Test rate limiter under concurrent access."""
    limiter = RateLimiter(rate=10, burst=1)
    
    async def compete():
        return await limiter.acquire()
    
    # Try to acquire concurrently
    results = await asyncio.gather(*(
        compete() for _ in range(5)
    ))
    
    # Only one should succeed
    assert sum(results) == 1