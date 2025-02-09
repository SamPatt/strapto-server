import asyncio
import pytest
import json
from strapto_server.model_interface import create_model_interface
from strapto_server.config import get_config

def print_model_output(output, prefix=""):
    """Helper to pretty print model outputs"""
    print(f"\n{prefix}Output:")
    print(f"  Type: {output.output_type}")
    print(f"  Content: {output.content}")
    if output.metadata:
        print(f"  Metadata:")
        for key, value in output.metadata.items():
            print(f"    {key}: {value}")
    print(f"  Timestamp: {output.timestamp}")

@pytest.mark.asyncio
async def test_ollama_connection():
    """Test basic connection to Ollama server"""
    config = get_config()  # This will load defaults
    config.model_name = None  # Let it auto-detect
    interface = create_model_interface("ollama", config)
    
    try:
        # Test connection
        connected = await interface.connect()
        assert connected, "Failed to connect to Ollama"
        print(f"\nConnected to model: {interface.model_name}")
        
        # Test sending a message and receiving response
        test_prompt = "Write a haiku about programming"
        print(f"\nSending prompt: {test_prompt}")
        await interface.send_input(test_prompt)
        
        # Collect responses
        response_text = ""
        async for output in interface.get_outputs():
            print_model_output(output)
            
            if output.output_type == "text":
                response_text += str(output.content)
            elif output.output_type == "status":
                print(f"\nModel status update received")
                
            if output.metadata and output.metadata.get("done", False):
                break
        
        print("\nFinal Response:")
        print(response_text)
        
        assert len(response_text) > 0, "No response received from Ollama"
        
    finally:
        if interface.session:
            await interface.disconnect()

@pytest.mark.asyncio
async def test_ollama_auto_model_detection():
    """Test that OllamaInterface can automatically detect and use an available model"""
    config = get_config()
    config.model_name = None  # Explicitly set to None to test auto-detection
    interface = create_model_interface("ollama", config)
    
    try:
        # Test connection with auto-detection
        connected = await interface.connect()
        assert connected, "Failed to connect to Ollama"
        assert interface.model_name is not None, "No model was auto-detected"
        print(f"\nAuto-detected model: {interface.model_name}")
        
        # Test that the auto-detected model works
        test_prompt = "Say hello"
        print(f"\nSending prompt: {test_prompt}")
        await interface.send_input(test_prompt)
        
        # Collect responses
        response_text = ""
        response_received = False
        
        async for output in interface.get_outputs():
            print_model_output(output, "Stream")
            
            if output.output_type == "text":
                response_text += str(output.content)
                if output.content:  # If we got any content
                    response_received = True
                    
            if output.metadata and output.metadata.get("done", False):
                break
                
        assert response_received, f"No response received from auto-detected model. Full response: {response_text}"
        print(f"\nFinal Response: {response_text}")
                
    finally:
        if interface.session:
            await interface.disconnect()

@pytest.mark.asyncio
async def test_model_detection_priority():
    """Test that running models are prioritized over available models"""
    config = get_config()
    config.model_name = None
    interface = create_model_interface("ollama", config)
    
    try:
        # First get the session established
        connected = await interface.connect()
        assert connected, "Failed to connect to Ollama"
        print(f"\nConnected successfully")
        
        # Test the _get_available_model method directly
        model = await interface._get_available_model()
        assert model is not None, "No model found"
        print(f"\nDetected model: {model}")
        
        # Check if it's a running model by querying /api/ps
        async with interface.session.get(f"{interface.base_url}/api/ps") as response:
            data = await response.json()
            running_models = [m['name'] for m in data.get('models', [])]
            print(f"\nCurrently running models: {running_models}")
            
        # The selected model should be in the running models if any exist
        if running_models:
            assert model in running_models, "Selected model is not from running models"
            print(f"\nCorrectly selected running model: {model}")
        else:
            print(f"\nNo running models found, using available model: {model}")
            
    finally:
        if interface.session:
            await interface.disconnect()

if __name__ == "__main__":
    asyncio.run(test_ollama_connection()) 