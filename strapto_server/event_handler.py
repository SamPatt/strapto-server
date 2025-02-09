"""
Event Handler for the StrapTo Local Server.

This module implements an asynchronous event emitter/listener system for processing
messages between different components of the application. It handles:
- Registration of event listeners
- Event emission and processing
- Error handling and logging for event processing
"""

import asyncio
import logging
from typing import Any, Awaitable, Callable, Dict, List, Optional, Set
from dataclasses import dataclass, field
import json
import time

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class Event:
    """
    Represents a single event in the system.
    """
    type: str
    data: Any
    timestamp: float = field(default_factory=time.time)
    metadata: Optional[Dict[str, Any]] = None

    def to_json(self) -> str:
        """Convert event to JSON string."""
        return json.dumps({
            "type": self.type,
            "data": self.data,
            "timestamp": self.timestamp,
            "metadata": self.metadata
        })

# Type hint for event handlers
EventHandler = Callable[[Event], Awaitable[None]]

class EventEmitter:
    """
    Asynchronous event emitter that manages event listeners and event emission.
    """
    
    def __init__(self):
        """Initialize the event emitter with empty listener maps."""
        self._listeners: Dict[str, Set[EventHandler]] = {}
        self._error_handlers: Set[EventHandler] = set()
        self._running = True
        
    def add_listener(self, event_type: str, handler: EventHandler) -> None:
        """
        Register a handler for a specific event type.
        
        Args:
            event_type: Type of event to listen for
            handler: Async function to handle the event
        """
        if event_type not in self._listeners:
            self._listeners[event_type] = set()
        self._listeners[event_type].add(handler)
        logger.debug(f"Added listener for event type: {event_type}")

    def remove_listener(self, event_type: str, handler: EventHandler) -> None:
        """
        Remove a handler for a specific event type.
        
        Args:
            event_type: Type of event
            handler: Handler to remove
        """
        if event_type in self._listeners:
            self._listeners[event_type].discard(handler)
            if not self._listeners[event_type]:
                del self._listeners[event_type]
            logger.debug(f"Removed listener for event type: {event_type}")

    def add_error_handler(self, handler: EventHandler) -> None:
        """
        Add a handler for error events.
        
        Args:
            handler: Async function to handle errors
        """
        self._error_handlers.add(handler)

    def remove_error_handler(self, handler: EventHandler) -> None:
        """
        Remove an error handler.
        
        Args:
            handler: Handler to remove
        """
        self._error_handlers.discard(handler)

    async def emit(self, event: Event) -> None:
        """
        Emit an event to all registered handlers.
        
        Args:
            event: Event to emit
        """
        if not self._running:
            raise RuntimeError("EventEmitter is not running")

        handlers = self._listeners.get(event.type, set())
        if not handlers:
            logger.debug(f"No handlers for event type: {event.type}")
            return

        tasks = []
        for handler in handlers:
            task = asyncio.create_task(self._safe_handle(handler, event))
            tasks.append(task)

        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _safe_handle(self, handler: EventHandler, event: Event) -> None:
        """
        Safely execute a handler with error handling.
        
        Args:
            handler: Event handler to execute
            event: Event to handle
        """
        try:
            await handler(event)
        except Exception as e:
            error_event = Event(
                type="error",
                data={
                    "original_event": event,
                    "error": str(e),
                    "handler": handler.__name__
                }
            )
            logger.error(f"Error in event handler: {e}")
            
            # Notify error handlers
            for error_handler in self._error_handlers:
                try:
                    await error_handler(error_event)
                except Exception as e:
                    logger.error(f"Error in error handler: {e}")

    def stop(self) -> None:
        """Stop the event emitter."""
        self._running = False
        logger.info("EventEmitter stopped")

# Example usage and testing
async def example_usage():
    """Demonstrate how to use the EventEmitter."""
    emitter = EventEmitter()
    
    # Example event handler
    async def handle_message(event: Event):
        print(f"Received message: {event.data}")
    
    # Example error handler
    async def handle_error(event: Event):
        print(f"Error occurred: {event.data['error']}")
    
    # Register handlers
    emitter.add_listener("message", handle_message)
    emitter.add_error_handler(handle_error)
    
    # Emit an event
    event = Event(type="message", data="Hello, World!")
    await emitter.emit(event)
    
    # Clean up
    emitter.stop()

if __name__ == "__main__":
    asyncio.run(example_usage())