"""
API route handlers for the StrapTo local server.

This module implements the HTTP endpoints using aiohttp. It provides:
- Health check endpoint
- Server status and metrics
- WebRTC signaling endpoints
- Model control endpoints
"""

from datetime import datetime
from typing import Dict, Any
from aiohttp import web
import logging

from ..config import ServerConfig
from ..webrtc_manager import WebRTCManager
from ..model_interface import GenericModelInterface
from ..event_handler import EventEmitter

logger = logging.getLogger(__name__)

routes = web.RouteTableDef()

@routes.get("/health")
async def health_check(request: web.Request) -> web.Response:
    """Simple health check endpoint."""
    return web.json_response({"status": "healthy"})

@routes.get("/status")
async def server_status(request: web.Request) -> web.Response:
    """
    Get detailed server status.
    
    Returns information about:
    - Server uptime
    - Active WebRTC connections
    - Model interface status
    """
    app = request.app
    webrtc: WebRTCManager = app['webrtc_manager']
    model: GenericModelInterface = app['model_interface']
    
    # Calculate actual uptime
    uptime = (datetime.now() - app['start_time']).total_seconds()
    
    status = {
        "server": {
            "status": "running",
            "uptime": uptime,
        },
        "webrtc": {
            "active_connections": len(webrtc.active_connections),
            "data_channels": webrtc.get_active_channel_count(),
        },
        "model": {
            "status": model.connection_status,
            "last_activity": model.last_activity_timestamp,
        }
    }
    
    return web.json_response(status)

@routes.post("/webrtc/offer")
async def handle_offer(request: web.Request) -> web.Response:
    """
    Handle incoming WebRTC offers.
    
    Expected JSON payload:
    {
        "sdp": str,           # Required: Session Description Protocol string
        "type": str,          # Required: Must be "offer"
        "client_id": str      # Required: Unique client identifier
    }
    """
    try:
        data = await request.json()
        
        # Validate required fields
        required_fields = {'sdp', 'type', 'client_id'}
        if not all(field in data for field in required_fields):
            missing = required_fields - set(data.keys())
            return web.json_response(
                {"error": f"Missing required fields: {', '.join(missing)}"},
                status=400
            )
            
        # Validate offer type
        if data['type'] != 'offer':
            return web.json_response(
                {"error": "Invalid offer type"},
                status=400
            )
        
        webrtc: WebRTCManager = request.app['webrtc_manager']
        answer = await webrtc.handle_offer(
            sdp=data['sdp'],
            client_id=data['client_id']
        )
        
        return web.json_response(answer)
        
    except ValueError as e:
        logger.error(f"Invalid WebRTC offer format: {e}")
        return web.json_response(
            {"error": "Invalid offer format"},
            status=400
        )
    except Exception as e:
        logger.error(f"Error handling WebRTC offer: {e}")
        return web.json_response(
            {"error": "Internal server error"},
            status=500
        )

@routes.post("/webrtc/ice-candidate")
async def handle_ice_candidate(request: web.Request) -> web.Response:
    """
    Handle incoming ICE candidates.
    
    Expected JSON payload:
    {
        "candidate": str,     # Required: ICE candidate string
        "sdpMLineIndex": int, # Required: Line index
        "sdpMid": str,       # Required: Media ID
        "client_id": str     # Required: Client identifier
    }
    """
    try:
        data = await request.json()
        
        # Validate required fields
        required_fields = {'candidate', 'sdpMLineIndex', 'sdpMid', 'client_id'}
        if not all(field in data for field in required_fields):
            missing = required_fields - set(data.keys())
            return web.json_response(
                {"error": f"Missing required fields: {', '.join(missing)}"},
                status=400
            )
            
        webrtc: WebRTCManager = request.app['webrtc_manager']
        await webrtc.handle_ice_candidate(
            candidate=data['candidate'],
            sdp_mline_index=data['sdpMLineIndex'],
            sdp_mid=data['sdpMid'],
            client_id=data['client_id']
        )
        
        return web.json_response({"status": "accepted"})
        
    except ValueError as e:
        logger.error(f"Invalid ICE candidate format: {e}")
        return web.json_response(
            {"error": "Invalid ICE candidate format"},
            status=400
        )
    except Exception as e:
        logger.error(f"Error handling ICE candidate: {e}")
        return web.json_response(
            {"error": "Internal server error"},
            status=500
        )

@routes.post("/model/reset")
async def reset_model(request: web.Request) -> web.Response:
    """Reset the model interface connection."""
    try:
        model: GenericModelInterface = request.app['model_interface']
        
        # Check current status before attempting reset
        if not model.is_connected and not model.is_connecting:
            return web.json_response(
                {"error": "Model interface is not connected or is already resetting"},
                status=400
            )
            
        await model.disconnect()
        await model.connect(request.app['config'])
        
        return web.json_response({
            "status": "reset_complete",
            "connection_status": model.connection_status
        })
        
    except Exception as e:
        logger.error(f"Error resetting model interface: {e}")
        return web.json_response(
            {"error": "Failed to reset model interface"},
            status=500
        )

def setup_routes(app: web.Application,
                 config: ServerConfig,
                 webrtc_manager: WebRTCManager,
                 model_interface: GenericModelInterface,
                 event_emitter: EventEmitter) -> None:
    """
    Set up API routes and attach components to the application.
    
    Args:
        app: aiohttp Application instance
        config: Server configuration
        webrtc_manager: WebRTC connection manager
        model_interface: Model interface instance
        event_emitter: Event emitter instance
    """
    # Set start time for uptime calculation
    app['start_time'] = datetime.now()
    
    # Attach components to app
    app['config'] = config
    app['webrtc_manager'] = webrtc_manager
    app['model_interface'] = model_interface
    app['event_emitter'] = event_emitter
    
    # Add routes
    app.add_routes(routes)
    
    # Add CORS middleware if needed
    if config.debug:
        import aiohttp_cors
        
        # Configure default options
        cors = aiohttp_cors.setup(app, defaults={
            "*": aiohttp_cors.ResourceOptions(
                allow_credentials=True,
                expose_headers="*",
                allow_headers="*",
            )
        })
        
        # Configure CORS on all routes
        for route in list(app.router.routes()):
            cors.add(route)
        
        # Store cors instance to verify in tests
        app['cors'] = cors
            
    # Setup cleanup
    async def cleanup_event_listeners(app):
        """Remove all event listeners on shutdown."""
        event_emitter = app['event_emitter']
        await event_emitter.remove_listener()

    app.on_cleanup.append(cleanup_event_listeners)