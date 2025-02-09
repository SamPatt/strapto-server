"""
Tests for the WebRTC manager module.
"""

import asyncio
import json
from datetime import datetime
import pytest
from unittest.mock import AsyncMock, MagicMock, patch, call

from aiortc import RTCPeerConnection, RTCSessionDescription, RTCDataChannel, RTCIceCandidate
from strapto_server.webrtc_manager import WebRTCManager, PeerInfo, PeerMetrics
from strapto_server.event_handler import Event, EventEmitter
from strapto_server.config import ServerConfig

@pytest.fixture
def config():
    """Fixture to provide a test configuration."""
    return ServerConfig(
        webrtc_host="localhost",
        webrtc_port=8765,
        stun_server="stun:stun.l.google.com:19302",
        signaling_url="ws://localhost:8080"
    )

@pytest.fixture
def event_emitter():
    """Fixture to provide a fresh EventEmitter."""
    return EventEmitter()

@pytest.fixture
def webrtc_manager(config, event_emitter):
    """Fixture to provide a WebRTCManager instance."""
    manager = WebRTCManager(config=config, event_emitter=event_emitter)
    yield manager
    # Cleanup after each test
    asyncio.run(manager.close_all_connections())

@pytest.fixture
def mock_session_description():
    """Create a mock session description that can be JSON serialized."""
    desc = MagicMock(spec=RTCSessionDescription)
    desc.sdp = "test sdp"
    desc.type = "answer"
    # Make the mock JSON serializable
    desc.__dict__.update({"sdp": "test sdp", "type": "answer"})
    return desc

def create_mock_data_channel():
    """Create a mock data channel with proper event handling."""
    channel = MagicMock(spec=RTCDataChannel)
    channel.readyState = "open"
    channel.send = MagicMock()
    channel.close = MagicMock()
    
    # Store event handlers
    channel._handlers = {}
    def mock_on(event_name, handler=None):
        if handler:
            channel._handlers[event_name] = handler
            return handler
        return lambda x: None  # Return a dummy callable if no handler is provided
    channel.on = mock_on
    
    # Set up default async message handler
    async def default_message_handler(message):
        pass
    channel._handlers["message"] = default_message_handler
    
    return channel

def create_mock_peer_connection(mock_session_description):
    """Create a mock peer connection with proper event handling."""
    pc = MagicMock(spec=RTCPeerConnection)
    pc.connectionState = "new"
    pc.iceConnectionState = "new"
    pc.iceGatheringState = "new"
    
    # Set up localDescription to return our serializable mock
    pc.localDescription = mock_session_description
    
    # Store event handlers
    pc._handlers = {}
    def mock_on(event_name, handler=None):
        if handler:
            pc._handlers[event_name] = handler
            return handler
        return lambda x: None  # Return a dummy callable if no handler is provided
    pc.on = mock_on
    
    # Set up async methods
    pc.createOffer = AsyncMock(return_value=mock_session_description)
    pc.createAnswer = AsyncMock(return_value=mock_session_description)
    pc.setLocalDescription = AsyncMock()
    pc.setRemoteDescription = AsyncMock()
    pc.addIceCandidate = AsyncMock()
    pc.close = AsyncMock()
    
    # Set up data channel creation
    data_channel = create_mock_data_channel()
    pc.createDataChannel = MagicMock(return_value=data_channel)
    
    return pc

@pytest.fixture
def mock_pc(mock_session_description):
    """Fixture to provide a consistent mock peer connection."""
    return create_mock_peer_connection(mock_session_description)

@pytest.mark.asyncio
async def test_create_connection(webrtc_manager, mock_pc, mock_session_description):
    """Test server-initiated connection creation."""
    with patch('strapto_server.webrtc_manager.RTCPeerConnection', return_value=mock_pc):
        peer_id = "test_peer"
        
        # Create connection and get offer
        new_peer_id, offer = await webrtc_manager.create_connection(peer_id)
        
        # Verify peer was created
        assert new_peer_id == peer_id
        assert peer_id in webrtc_manager.peers
        assert isinstance(webrtc_manager.peers[peer_id], PeerInfo)
        assert webrtc_manager.peers[peer_id].data_channel is not None
        
        # Verify offer
        assert isinstance(offer, RTCSessionDescription)
        assert offer.sdp == "test sdp"
        assert offer.type in ["offer", "answer"]

@pytest.mark.asyncio
async def test_process_answer(webrtc_manager, mock_pc):
    """Test processing of SDP answers."""
    with patch('strapto_server.webrtc_manager.RTCPeerConnection', return_value=mock_pc):
        peer_id = "test_peer"
        
        # First create a connection
        await webrtc_manager.create_connection(peer_id)
        
        # Process answer
        answer_sdp = json.dumps({
            "sdp": "test answer",
            "type": "answer"
        })
        
        await webrtc_manager.process_answer(peer_id, answer_sdp)
        
        # Verify answer was processed
        mock_pc.setRemoteDescription.assert_called_once()

@pytest.mark.asyncio
async def test_process_offer(webrtc_manager, mock_pc, mock_session_description):
    """Test processing of SDP offers (client-initiated connection)."""
    with patch('strapto_server.webrtc_manager.RTCPeerConnection', return_value=mock_pc):
        peer_id = "test_peer"
        offer_sdp = json.dumps({
            "sdp": "test offer",
            "type": "offer"
        })
        
        # Process offer
        answer_sdp = await webrtc_manager.process_offer(peer_id, offer_sdp)
        
        # Verify peer was created
        assert peer_id in webrtc_manager.peers
        assert isinstance(webrtc_manager.peers[peer_id], PeerInfo)
        
        # Verify SDP handling
        mock_pc.setRemoteDescription.assert_called_once()
        mock_pc.createAnswer.assert_called_once()
        mock_pc.setLocalDescription.assert_called_once()
        
        # Verify answer format
        answer = json.loads(answer_sdp)
        assert "sdp" in answer
        assert "type" in answer
        assert answer["type"] == "answer"

@pytest.mark.asyncio
async def test_process_ice_candidate(webrtc_manager, mock_pc):
    """Test processing ICE candidates."""
    with patch('strapto_server.webrtc_manager.RTCPeerConnection', return_value=mock_pc):
        peer_id = "test_peer"
        
        # Create peer connection first
        await webrtc_manager.create_connection(peer_id)
        
        # Set remoteDescription to allow immediate candidate processing
        mock_pc.remoteDescription = True
        
        # Test candidate processing with correct RTCIceCandidate parameters
        candidate = json.dumps({
            "component": 1,
            "foundation": "1",
            "ip": "192.168.1.100",
            "port": 30000,
            "priority": 2013266431,
            "protocol": "udp",
            "type": "host",
            "sdpMLineIndex": 0,
            "sdpMid": "0"
        })
        
        await webrtc_manager.process_ice_candidate(peer_id, candidate)
        
        # Verify candidate was processed
        assert mock_pc.addIceCandidate.called

@pytest.mark.asyncio
async def test_handle_model_output(webrtc_manager, mock_pc):
    """Test handling of model output events."""
    with patch('strapto_server.webrtc_manager.RTCPeerConnection', return_value=mock_pc):
        # Set up mock peer with data channel
        peer_id = "test_peer"
        await webrtc_manager.create_connection(peer_id)
        
        # Create and emit test event
        test_event = Event(
            type="model_output",
            data="test output",
            metadata={"test": "metadata"}
        )
        await webrtc_manager.event_emitter.emit(test_event)
        
        # Verify message was sent
        channel = webrtc_manager.peers[peer_id].data_channel
        channel.send.assert_called_once()
        
        # Verify message format
        sent_message = json.loads(channel.send.call_args[0][0])
        assert sent_message["type"] == "model_output"
        assert sent_message["data"] == "test output"
        assert sent_message["metadata"] == {"test": "metadata"}

@pytest.mark.asyncio
async def test_error_handling_invalid_json(webrtc_manager, mock_pc):
    """Test handling of invalid JSON in messages."""
    with patch('strapto_server.webrtc_manager.RTCPeerConnection', return_value=mock_pc):
        peer_id = "test_peer"
        await webrtc_manager.create_connection(peer_id)
        
        # Get the channel
        channel = webrtc_manager.peers[peer_id].data_channel
        
        # Simulate an invalid JSON message
        message_handler = channel._handlers.get("message")
        assert message_handler is not None
        
        # Call the handler and verify metrics weren't updated
        await message_handler("invalid json{")
        metrics = webrtc_manager.peers[peer_id].metrics
        assert metrics.messages_received == 0

@pytest.mark.asyncio
async def test_close_peer_connection(webrtc_manager, mock_pc):
    """Test closing a peer connection."""
    with patch('strapto_server.webrtc_manager.RTCPeerConnection', return_value=mock_pc):
        peer_id = "test_peer"
        await webrtc_manager.create_connection(peer_id)
        
        # Close connection
        await webrtc_manager.close_peer_connection(peer_id)
        
        # Verify cleanup
        assert peer_id not in webrtc_manager.peers
        mock_pc.close.assert_called_once()

@pytest.mark.asyncio
async def test_get_metrics(webrtc_manager, mock_pc):
    """Test metrics collection."""
    with patch('strapto_server.webrtc_manager.RTCPeerConnection', return_value=mock_pc):
        peer_id = "test_peer"
        await webrtc_manager.create_connection(peer_id)
        
        # Get metrics
        metrics = webrtc_manager.get_metrics()
        
        # Verify metrics structure
        assert peer_id in metrics
        assert isinstance(metrics[peer_id], PeerMetrics)
        assert metrics[peer_id].connected_at is not None
        assert metrics[peer_id].messages_sent == 0
        assert metrics[peer_id].messages_received == 0