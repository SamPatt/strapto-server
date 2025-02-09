"""
Tests for the StrapTo server main module.

This test suite covers both unit tests and integration tests for the StrapToServer class
and its components. It uses pytest and pytest-asyncio for async testing support.
"""

import asyncio
import pytest
import logging
from unittest.mock import Mock, AsyncMock, patch

from strapto_server.main import StrapToServer
from strapto_server.config import ServerConfig
from strapto_server.event_handler import EventEmitter
from strapto_server.webrtc_manager import WebRTCManager
from strapto_server.model_interface import GenericModelInterface

# Disable logging during tests
logging.getLogger('strapto_server.main').setLevel(logging.ERROR)

@pytest.fixture
def mock_config():
    """Fixture to provide a mock server configuration."""
    config = Mock(spec=ServerConfig)
    config.stun_server = "stun:stun.l.google.com:19302"
    config.turn_server = None
    config.webrtc_host = "localhost"
    config.webrtc_port = 8765
    return config

@pytest.fixture
def server(mock_config):
    """Fixture that provides a StrapToServer instance with mocked components."""
    with patch('strapto_server.main.get_config', return_value=mock_config):
        server = StrapToServer()
        
        # Create mocks with the required methods
        webrtc_manager = AsyncMock(spec=WebRTCManager)
        webrtc_manager.connect = AsyncMock()
        webrtc_manager.disconnect = AsyncMock()
        
        model_interface = AsyncMock(spec=GenericModelInterface)
        model_interface.connect = AsyncMock()
        model_interface.disconnect = AsyncMock()
        
        # Replace real components with mocks
        server.webrtc_manager = webrtc_manager
        server.model_interface = model_interface
        
        return server

@pytest.fixture
async def running_server(server):
    """Fixture that provides a running server and handles cleanup."""
    try:
        yield server
    finally:
        if server.running:
            await server.shutdown()

@pytest.mark.asyncio
async def test_server_initialization(mock_config):
    """Test that server initializes with correct component setup."""
    with patch('strapto_server.main.get_config', return_value=mock_config):
        server = StrapToServer()
        
        assert isinstance(server.event_emitter, EventEmitter)
        assert isinstance(server.webrtc_manager, WebRTCManager)
        assert isinstance(server.model_interface, GenericModelInterface)
        assert server.running is False
        assert isinstance(server.tasks, set)

@pytest.mark.asyncio
async def test_component_connection(server):
    """Test that components connect in the correct order."""
    await server.connect()
    
    server.webrtc_manager.connect.assert_called_once_with(server.config)
    server.model_interface.connect.assert_called_once_with(server.config)

@pytest.mark.asyncio
async def test_component_connection_failure(server):
    """Test handling of component connection failures."""
    server.webrtc_manager.connect.side_effect = Exception("Connection failed")
    
    with pytest.raises(Exception):
        await server.connect()
    
    server.webrtc_manager.connect.assert_called_once()
    server.model_interface.connect.assert_not_called()

@pytest.mark.asyncio
async def test_graceful_shutdown(server):
    """Test that shutdown properly closes all components."""
    server.running = True
    
    # Create a proper mock task
    class MockTask:
        def __init__(self):
            self._done = False
            self.cancel_called = False
            
        def done(self):
            return self._done
            
        def cancel(self):
            self.cancel_called = True
            
        def __await__(self):
            if self.cancel_called:
                self._done = True
                raise asyncio.CancelledError()
            yield None
    
    mock_task = MockTask()
    server.tasks.add(mock_task)
    
    await server.shutdown()
    
    assert server.running is False
    assert mock_task.cancel_called
    server.model_interface.disconnect.assert_called_once()
    server.webrtc_manager.disconnect.assert_called_once()

@pytest.mark.asyncio
async def test_shutdown_with_failed_disconnection(server):
    """Test shutdown handling when components fail to disconnect."""
    server.running = True
    server.model_interface.disconnect.side_effect = Exception("Disconnect failed")
    
    await server.shutdown()
    
    server.model_interface.disconnect.assert_called_once()
    server.webrtc_manager.disconnect.assert_called_once()

@pytest.mark.asyncio
async def test_server_start_and_stop(server):
    """Integration test for server startup and shutdown."""
    # Create a task to stop the server after a short delay
    async def stop_server():
        await asyncio.sleep(0.1)
        await server.shutdown()
    
    stop_task = asyncio.create_task(stop_server())
    try:
        await server.start()
    finally:
        if not stop_task.done():
            await stop_task
    
    server.webrtc_manager.connect.assert_called_once()
    server.model_interface.connect.assert_called_once()
    server.webrtc_manager.disconnect.assert_called_once()
    server.model_interface.disconnect.assert_called_once()

@pytest.mark.asyncio
async def test_signal_handling(server):
    """Test server response to shutdown signals."""
    import signal
    
    async def trigger_shutdown():
        await asyncio.sleep(0.1)
        await server.shutdown(signal.SIGTERM)
    
    shutdown_task = asyncio.create_task(trigger_shutdown())
    try:
        await server.start()
    finally:
        if not shutdown_task.done():
            await shutdown_task
    
    assert not server.running
    server.model_interface.disconnect.assert_called_once()
    server.webrtc_manager.disconnect.assert_called_once()

def test_main_function():
    """Test the main() entry point function."""
    mock_server = AsyncMock(spec=StrapToServer)
    
    with patch('strapto_server.main.StrapToServer', return_value=mock_server):
        from strapto_server.main import main
        
        try:
            main()
        except KeyboardInterrupt:
            pass
        
        mock_server.start.assert_called_once()
