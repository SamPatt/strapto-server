"""
Route handlers for the StrapTo local server API.

This package provides HTTP endpoints for server control and monitoring.
Routes are implemented using aiohttp and provide functionality for:
- Server status and health checks
- WebRTC signaling
- Model interface controls
"""

from .api import setup_routes

__all__ = ['setup_routes']