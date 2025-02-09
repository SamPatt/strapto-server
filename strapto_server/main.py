#!/usr/bin/env python3

"""
StrapTo Local Server Entry Point

This module serves as the main entry point for the StrapTo local server. It coordinates
the lifecycle of server components and manages the main server loop. The server can be
started directly by running this file or by importing and using the StrapToServer class.

Components:
- ServerConfig: Configuration management and environment settings
- EventEmitter: Event-based communication between components
- WebRTCManager: WebRTC connection handling and data channels
- GenericModelInterface: Default implementation of model I/O interface
"""

import asyncio
import logging
import signal
from typing import Optional, Set

from .config import get_config, ServerConfig
from .webrtc_manager import WebRTCManager
from .model_interface import GenericModelInterface
from .event_handler import EventEmitter

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class StrapToServer:
    """
    Main server class that orchestrates all StrapTo components.

    This class manages the lifecycle of server components, handles graceful shutdown,
    and maintains the main server loop using asyncio.
    """
    def __init__(self):
        """Initialize server components with proper configuration."""
        self.config: ServerConfig = get_config()
        self.event_emitter = EventEmitter()
        self.webrtc_manager = WebRTCManager(config=self.config, event_emitter=self.event_emitter)
        self.model_interface = GenericModelInterface(self.event_emitter)
        self.running = False
        self.tasks: Set[asyncio.Task] = set()

    async def connect(self):
        """Connect all components."""
        await self.webrtc_manager.connect(self.config)
        await self.model_interface.connect(self.config)

    async def start(self):
        """
        Start the server and all its components.

        This method:
          1. Establishes signal handlers for graceful shutdown.
          2. Connects components (WebRTC manager and model interface).
          3. Runs the main server loop.
          4. Handles errors and cleanup.
        """
        logger.info("Starting StrapTo local server...")
        self.running = True

        # Setup signal handlers for graceful shutdown.
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(
                sig,
                lambda s=sig: asyncio.create_task(self.shutdown(s))
            )

        try:
            await self.connect()
            while self.running:
                await asyncio.sleep(1)  # Main loop tick.
        except Exception as e:
            logger.error(f"Critical error in server: {e}")
            raise
        finally:
            await self.shutdown()

    async def shutdown(self, sig: Optional[signal.Signals] = None):
        """
        Gracefully shutdown all server components.

        Args:
            sig: Optional signal that triggered the shutdown.
        """
        if sig:
            logger.info(f"Received shutdown signal: {sig.name}")

        if not self.running:
            logger.info("Server already shutting down...")
            return

        self.running = False
        logger.info("Initiating StrapTo server shutdown...")

        # Cancel all running tasks.
        for task in self.tasks:
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        # Disconnect components in reverse order.
        try:
            await self.model_interface.disconnect()
        except Exception as e:
            logger.error(f"Error disconnecting model interface: {e}")

        try:
            await self.webrtc_manager.disconnect()
        except Exception as e:
            logger.error(f"Error disconnecting WebRTC manager: {e}")

        # Clean up event emitter - remove all registered listeners
        # Create a copy of the listeners to avoid modifying while iterating
        listeners_to_remove = [
            (event_type, list(listeners))
            for event_type, listeners in self.event_emitter._listeners.items()
        ]
        
        for event_type, listeners in listeners_to_remove:
            for listener in listeners:
                self.event_emitter.remove_listener(event_type, listener)

        logger.info("Server shutdown complete.")


def main():
    """
    Entry point for starting the StrapTo local server.

    This function creates and runs the server instance, handling any errors
    that occur during server operation.
    """
    server = StrapToServer()
    try:
        asyncio.run(server.start())
    except KeyboardInterrupt:
        logger.info("Server stopped by user.")
    except Exception as e:
        logger.error(f"Server stopped due to error: {e}")
        raise
    finally:
        logger.info("Server process terminated.")


if __name__ == "__main__":
    main()
