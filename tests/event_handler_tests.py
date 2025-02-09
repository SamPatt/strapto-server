"""
Tests for the event handler module.
"""

import asyncio
import pytest
from strapto_server.event_handler import Event, EventEmitter

@pytest.fixture
def event_emitter():
    """Fixture to provide a fresh EventEmitter for each test."""
    return EventEmitter()

@pytest.mark.asyncio
async def test_basic_event_emission(event_emitter):
    """Test basic event emission and handling."""
    received_events = []
    
    async def test_handler(event):
        received_events.append(event)
    
    # Register handler and emit event
    event_emitter.add_listener("test", test_handler)
    test_event = Event(type="test", data="test data")
    await event_emitter.emit(test_event)
    
    assert len(received_events) == 1
    assert received_events[0].type == "test"
    assert received_events[0].data == "test data"

@pytest.mark.asyncio
async def test_multiple_handlers(event_emitter):
    """Test multiple handlers for the same event type."""
    handler1_called = False
    handler2_called = False
    
    async def handler1(event):
        nonlocal handler1_called
        handler1_called = True
    
    async def handler2(event):
        nonlocal handler2_called
        handler2_called = True
    
    # Register handlers and emit event
    event_emitter.add_listener("test", handler1)
    event_emitter.add_listener("test", handler2)
    await event_emitter.emit(Event(type="test", data="test"))
    
    assert handler1_called
    assert handler2_called

@pytest.mark.asyncio
async def test_remove_listener(event_emitter):
    """Test removing an event listener."""
    called = False
    
    async def test_handler(event):
        nonlocal called
        called = True
    
    # Register handler, then remove it
    event_emitter.add_listener("test", test_handler)
    event_emitter.remove_listener("test", test_handler)
    await event_emitter.emit(Event(type="test", data="test"))
    
    assert not called

@pytest.mark.asyncio
async def test_error_handling(event_emitter):
    """Test error handling in event handlers."""
    error_events = []
    
    async def failing_handler(event):
        raise ValueError("Test error")
    
    async def error_handler(error_event):
        error_events.append(error_event)
    
    # Register handlers
    event_emitter.add_listener("test", failing_handler)
    event_emitter.add_error_handler(error_handler)
    
    # Emit event that will cause an error
    await event_emitter.emit(Event(type="test", data="test"))
    
    assert len(error_events) == 1
    assert error_events[0].type == "error"
    assert "Test error" in error_events[0].data["error"]

@pytest.mark.asyncio
async def test_event_json_serialization():
    """Test Event class JSON serialization."""
    event = Event(type="test", data={"message": "hello"}, metadata={"source": "test"})
    json_str = event.to_json()
    
    # Test that all fields are present in JSON
    assert "type" in json_str
    assert "data" in json_str
    assert "timestamp" in json_str
    assert "metadata" in json_str

@pytest.mark.asyncio
async def test_emitter_stopped_state(event_emitter):
    """Test that stopped emitter raises RuntimeError."""
    event_emitter.stop()
    
    with pytest.raises(RuntimeError):
        await event_emitter.emit(Event(type="test", data="test"))

@pytest.mark.asyncio
async def test_concurrent_event_handling(event_emitter):
    """Test concurrent event handling."""
    results = []
    
    async def slow_handler(event):
        await asyncio.sleep(0.1)
        results.append(1)
    
    async def fast_handler(event):
        results.append(2)
    
    # Register both handlers
    event_emitter.add_listener("test", slow_handler)
    event_emitter.add_listener("test", fast_handler)
    
    # Emit event
    await event_emitter.emit(Event(type="test", data="test"))
    
    assert len(results) == 2
    # Fast handler should finish first
    assert results[0] == 2
    assert results[1] == 1