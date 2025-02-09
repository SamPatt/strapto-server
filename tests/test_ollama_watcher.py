import asyncio
import logging
import pytest
from strapto_server.model_interface import OllamaInterface
from strapto_server.config import ServerConfig

# Set up logging to see what's happening
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@pytest.mark.asyncio
async def test_ollama_watcher():
    """Test that we can connect to Ollama and watch for activity."""
    # Create and connect the interface
    config = ServerConfig(
        model_name="phi4:latest",  # or whatever model you have
        webrtc_host="localhost",
        webrtc_port=8080,
        stun_server="stun:stun.l.google.com:19302",
        signaling_url="ws://localhost:3000"
    )
    interface = OllamaInterface(config)
    
    try:
        # Test connection
        connected = await interface.connect()
        assert connected, "Failed to connect to Ollama"
        logger.info("Connected to Ollama, watching for activity...")
        
        # Watch for a few seconds to capture any activity
        output_count = 0
        async for output in interface.get_outputs():
            logger.info(f"Received output type: {output.output_type}")
            if output.output_type == "status":
                logger.info(f"Active model status: {output.content}")
            else:
                logger.info(f"Received output: {output.content}")
                if output.metadata:
                    logger.info(f"Metadata: {output.metadata}")
            
            output_count += 1
            if output_count >= 3:  # Get at least 3 outputs before finishing
                break
                
    finally:
        # Clean up
        await interface.disconnect()
        logger.info("Disconnected from Ollama")

# Keep the main() for direct script execution
def main():
    logger.info("Starting Ollama watcher...")
    asyncio.run(watch_ollama())

async def watch_ollama():
    try:
        await test_ollama_watcher()
    except KeyboardInterrupt:
        logger.info("Stopping watcher...")
    except Exception as e:
        logger.error(f"Error in watcher: {e}")

if __name__ == "__main__":
    main() 