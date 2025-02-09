"""
Tests for the StrapTo server API routes.

These tests cover both successful and error cases for all API endpoints,
using aiohttp_test_utils for testing HTTP endpoints and mocking dependencies.
"""

import pytest
from datetime import datetime
from aiohttp import web
from unittest.mock import Mock, AsyncMock
import pytest_asyncio  # Import this explicitly

from strapto_server.routes import setup_routes
from strapto_server.config import ServerConfig
from strapto_server.webrtc_manager import WebRTCManager
from strapto_server.model_interface import GenericModelInterface
from strapto_server.event_handler import EventEmitter

pytestmark = pytest.mark.asyncio

@pytest.fixture
def mock_config():
    """Mock server configuration."""
    return Mock(spec=ServerConfig, debug=False)

@pytest.fixture
def mock_webrtc():
    """Mock WebRTC manager with necessary methods and properties."""
    manager = AsyncMock(spec=WebRTCManager)
    manager.active_connections = {}
    manager.get_active_channel_count = Mock(return_value=0)
    manager.handle_offer = AsyncMock()
    manager.handle_ice_candidate = AsyncMock()
    return manager

@pytest.fixture
def mock_model():
    """Mock model interface with necessary methods and properties."""
    model = AsyncMock(spec=GenericModelInterface)
    model.is_connected = True
    model.is_connecting = False
    model.connection_status = "connected"
    model.last_activity_timestamp = 0
    return model

@pytest.fixture
def mock_event_emitter():
    """Mock event emitter."""
    emitter = AsyncMock(spec=EventEmitter)
    emitter.remove_listener = AsyncMock()
    return emitter

@pytest_asyncio.fixture
def app(mock_config, mock_webrtc, mock_model, mock_event_emitter):
    """Create test application with mocked components."""
    app = web.Application()
    setup_routes(
        app=app,
        config=mock_config,
        webrtc_manager=mock_webrtc,
        model_interface=mock_model,
        event_emitter=mock_event_emitter
    )
    return app

@pytest_asyncio.fixture
async def client(aiohttp_client, app):
    """Create test client."""
    return await aiohttp_client(app)

async def test_health_check(client):
    """Test health check endpoint."""
    resp = await client.get("/health")
    assert resp.status == 200
    data = await resp.json()
    assert data["status"] == "healthy"

async def test_server_status(client):
    """Test server status endpoint."""
    resp = await client.get("/status")
    assert resp.status == 200
    data = await resp.json()
    
    # Check structure
    assert "server" in data
    assert "webrtc" in data
    assert "model" in data
    
    # Check server info
    assert data["server"]["status"] == "running"
    assert isinstance(data["server"]["uptime"], (int, float))
    
    # Check WebRTC info
    assert "active_connections" in data["webrtc"]
    assert "data_channels" in data["webrtc"]
    
    # Check model info
    assert data["model"]["status"] == "connected"
    assert "last_activity" in data["model"]

async def test_webrtc_offer_success(client, mock_webrtc):
    """Test successful WebRTC offer handling."""
    mock_webrtc.handle_offer.return_value = {"type": "answer", "sdp": "mock_answer"}
    
    offer_data = {
        "sdp": "mock_offer",
        "type": "offer",
        "client_id": "test_client"
    }
    
    resp = await client.post("/webrtc/offer", json=offer_data)
    assert resp.status == 200
    data = await resp.json()
    
    assert data["type"] == "answer"
    assert data["sdp"] == "mock_answer"
    mock_webrtc.handle_offer.assert_called_once_with(
        sdp="mock_offer",
        client_id="test_client"
    )

async def test_webrtc_offer_missing_fields(client):
    """Test WebRTC offer with missing required fields."""
    offer_data = {
        "sdp": "mock_offer"
        # Missing type and client_id
    }
    
    resp = await client.post("/webrtc/offer", json=offer_data)
    assert resp.status == 400
    data = await resp.json()
    assert "error" in data
    assert "Missing required fields" in data["error"]

async def test_ice_candidate_success(client, mock_webrtc):
    """Test successful ICE candidate handling."""
    candidate_data = {
        "candidate": "mock_candidate",
        "sdpMLineIndex": 0,
        "sdpMid": "mock_mid",
        "client_id": "test_client"
    }
    
    resp = await client.post("/webrtc/ice-candidate", json=candidate_data)
    assert resp.status == 200
    data = await resp.json()
    assert data["status"] == "accepted"
    
    mock_webrtc.handle_ice_candidate.assert_called_once_with(
        candidate="mock_candidate",
        sdp_mline_index=0,
        sdp_mid="mock_mid",
        client_id="test_client"
    )

async def test_ice_candidate_missing_fields(client):
    """Test ICE candidate with missing required fields."""
    candidate_data = {
        "candidate": "mock_candidate"
        # Missing other required fields
    }
    
    resp = await client.post("/webrtc/ice-candidate", json=candidate_data)
    assert resp.status == 400
    data = await resp.json()
    assert "error" in data
    assert "Missing required fields" in data["error"]

async def test_model_reset_success(client, mock_model):
    """Test successful model reset."""
    resp = await client.post("/model/reset")
    assert resp.status == 200
    data = await resp.json()
    
    assert data["status"] == "reset_complete"
    assert "connection_status" in data
    
    # Verify disconnect and connect were called
    mock_model.disconnect.assert_called_once()
    mock_model.connect.assert_called_once()

async def test_model_reset_not_connected(client, mock_model):
    """Test model reset when not connected."""
    mock_model.is_connected = False
    
    resp = await client.post("/model/reset")
    assert resp.status == 400
    data = await resp.json()
    assert "error" in data

async def test_cors_enabled(mock_config):
    """Test CORS setup when debug is enabled."""
    mock_config.debug = True
    app = web.Application()
    
    setup_routes(
        app,
        mock_config,
        Mock(spec=WebRTCManager),
        Mock(spec=GenericModelInterface),
        Mock(spec=EventEmitter)
    )
    
    # Verify CORS was set up
    assert 'cors' in app
    assert app['cors'] is not None

async def test_cleanup_handler(app, mock_event_emitter):
    """Test cleanup handler removes event listeners."""
    # Create a runner to properly handle cleanup
    runner = web.AppRunner(app)
    await runner.setup()
    
    # Trigger cleanup through the runner
    await runner.cleanup()
    
    # Verify event listeners were removed
    mock_event_emitter.remove_listener.assert_called_once()