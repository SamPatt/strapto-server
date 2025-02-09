import asyncio
import pytest
from strapto_server.model_interface import create_model_interface
from strapto_server.config import get_config

@pytest.mark.asyncio
async def test_ollama_connection():
    """Test basic connection to Ollama server"""
    config = get_config()  # This will load defaults
    config.model_name = "phi4:latest"
    interface = create_model_interface("ollama", config)
    
    try:
        # Test connection
        connected = await interface.connect()
        assert connected, "Failed to connect to Ollama"
        
        # Test sending a message and receiving response
        test_prompt = "Write a haiku about programming"
        
        # Send input
        await interface.send_input(test_prompt)
        
        # Collect responses
        response_text = ""
        async for output in interface.get_outputs():
            response_text += output.content
            if output.metadata and output.metadata.get("done", False):
                break
        
        print("\nOllama Response:")
        print(response_text)
        
        assert len(response_text) > 0, "No response received from Ollama"
        
    finally:
        await interface.disconnect()

if __name__ == "__main__":
    asyncio.run(test_ollama_connection()) 