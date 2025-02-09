"""
Model Interface for the StrapTo Local Server.

This module provides a flexible interface for connecting to and capturing outputs
from various types of self-hosted AI models. It implements an abstract base class
that can be extended to support different model types (Ollama, LMStudio, etc.).

The interface handles:
- Connecting to the model's API/endpoint
- Capturing model outputs in real-time
- Routing consumer inputs back to the model
- Managing the output format and streaming
"""

import abc
import asyncio
import logging
import json
from typing import Any, AsyncGenerator, Dict, Optional, Union
from dataclasses import dataclass

from .config import ServerConfig, get_config

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ModelOutput:
    """
    Represents a single output from the model.
    """
    content: Union[str, bytes, Dict[str, Any]]
    output_type: str  # "text", "json", "image"
    timestamp: float
    metadata: Optional[Dict[str, Any]] = None

class ModelInterface(abc.ABC):
    """
    Abstract base class defining the interface for model interactions.
    All specific model implementations (Ollama, LMStudio, etc.) should inherit from this.
    """
    
    def __init__(self, config: Optional[ServerConfig] = None):
        """
        Initialize the model interface with configuration.
        
        Args:
            config: ServerConfig object containing model settings
        """
        self.config = config or get_config()
        self.ready = False
        self._output_queue = asyncio.Queue()
        logger.info(f"Initializing {self.__class__.__name__}")

    @abc.abstractmethod
    async def connect(self) -> bool:
        """
        Establish connection to the model.
        
        Returns:
            bool: True if connection successful, False otherwise
        """
        pass

    @abc.abstractmethod
    async def disconnect(self) -> None:
        """Clean up and close the model connection."""
        pass

    @abc.abstractmethod
    async def send_input(self, content: Union[str, Dict[str, Any]]) -> None:
        """
        Send input to the model (e.g., from consumer interactions).
        
        Args:
            content: Input content (text or structured data)
        """
        pass

    async def get_outputs(self) -> AsyncGenerator[ModelOutput, None]:
        """
        Generator that yields model outputs as they become available.
        
        Yields:
            ModelOutput: Structured output from the model
        """
        while True:
            try:
                output = await self._output_queue.get()
                yield output
                self._output_queue.task_done()
            except asyncio.CancelledError:
                logger.info("Output stream cancelled")
                break
            except Exception as e:
                logger.error(f"Error in output stream: {e}")
                continue

class GenericModelInterface(ModelInterface):
    """
    Generic implementation of the ModelInterface for testing and development.
    Simulates a basic model that echoes inputs.
    """
    
    async def connect(self) -> bool:
        self.ready = True
        logger.info("Connected to generic model interface")
        return True

    async def disconnect(self) -> None:
        self.ready = False
        logger.info("Disconnected from generic model interface")

    async def send_input(self, content: Union[str, Dict[str, Any]]) -> None:
        """Echo the input back as an output."""
        import time
        
        if isinstance(content, str):
            output = ModelOutput(
                content=f"Echo: {content}",
                output_type="text",
                timestamp=time.time()
            )
        else:
            output = ModelOutput(
                content={"echo": content},
                output_type="json",
                timestamp=time.time()
            )
        
        await self._output_queue.put(output)

class OllamaInterface(ModelInterface):
    """Implementation of ModelInterface for Ollama."""
    
    def __init__(self, config: Optional[ServerConfig] = None):
        super().__init__(config)
        self.base_url = "http://localhost:11434"
        self.model_name = None  # We'll detect this during connection
        self.session = None
        self._watch_task = None
        self._watching = False

    async def _get_available_model(self) -> Optional[str]:
        """
        Get an available model, prioritizing:
        1. Currently running models
        2. Locally available models
        """
        try:
            # First check for any running models
            async with self.session.get(f"{self.base_url}/api/ps") as response:
                if response.status == 200:
                    data = await response.json()
                    if models := data.get('models', []):
                        # Use the first running model
                        self.model_name = models[0]['name']
                        logger.info(f"Using running model: {self.model_name}")
                        return self.model_name

            # If no running models, check locally available models
            async with self.session.get(f"{self.base_url}/api/tags") as response:
                if response.status == 200:
                    data = await response.json()
                    if models := data.get('models', []):
                        # Use the first available model
                        self.model_name = models[0]['name']
                        logger.info(f"Using available model: {self.model_name}")
                        return self.model_name
                
            logger.warning("No models found. Please install a model using 'ollama pull <model>'")
            return None
            
        except Exception as e:
            logger.error(f"Error checking Ollama models: {e}")
            return None

    async def connect(self) -> bool:
        """Connect to local Ollama instance and detect available model."""
        try:
            import aiohttp
            self.session = aiohttp.ClientSession()
            
            # Test connection by getting version
            async with self.session.get(f"{self.base_url}/api/version") as response:
                if response.status != 200:
                    logger.error(f"Failed to connect to Ollama: HTTP {response.status}")
                    return False

            # If no model specified in config, auto-detect
            if not self.model_name:
                if not await self._get_available_model():
                    return False

            self.ready = True
            logger.info(f"Connected to Ollama (model: {self.model_name})")
            await self.start_watching()
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to Ollama: {e}")
            return False

    async def disconnect(self) -> None:
        """Disconnect from Ollama and stop watching."""
        await self.stop_watching()
        if self.session:
            await self.session.close()
        self.ready = False
        logger.info("Disconnected from Ollama")

    async def start_watching(self) -> None:
        """Start watching Ollama activity."""
        if self._watching:
            return
        
        self._watching = True
        self._watch_task = asyncio.create_task(self._watch_ollama())
        logger.info("Started watching Ollama activity")

    async def stop_watching(self) -> None:
        """Stop watching Ollama activity."""
        self._watching = False
        if self._watch_task:
            self._watch_task.cancel()
            try:
                await self._watch_task
            except asyncio.CancelledError:
                pass
            self._watch_task = None
        logger.info("Stopped watching Ollama activity")

    async def _watch_ollama(self) -> None:
        """Watch Ollama activity and capture outputs."""
        import time
        
        while self._watching:
            try:
                # Check for running models
                async with self.session.get(f"{self.base_url}/api/ps") as response:
                    if response.status == 200:
                        data = await response.json()
                        running_models = data.get("models", [])
                        
                        if running_models:
                            # Log active models for debugging
                            model_names = [m.get("name") for m in running_models]
                            logger.debug(f"Active Ollama models: {model_names}")
                            
                            # Create an output for each active model
                            for model in running_models:
                                output = ModelOutput(
                                    content={
                                        "model_name": model.get("name"),
                                        "status": "active",
                                        "size": model.get("size"),
                                        "details": model.get("details", {})
                                    },
                                    output_type="status",
                                    timestamp=time.time(),
                                    metadata={
                                        "size_vram": model.get("size_vram"),
                                        "expires_at": model.get("expires_at")
                                    }
                                )
                                await self._output_queue.put(output)
                
                # Wait before next check
                await asyncio.sleep(1.0)  # Adjust polling interval as needed
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error watching Ollama: {e}")
                await asyncio.sleep(5.0)  # Longer delay on error

    async def send_input(self, content: Union[str, Dict[str, Any]]) -> None:
        """Send prompt to Ollama and stream responses."""
        import time
        
        if not self.session:
            logger.error("Not connected to Ollama")
            return

        try:
            # Convert input to appropriate format
            if isinstance(content, str):
                data = {
                    "model": self.model_name,
                    "prompt": content,
                    "stream": True
                }
            else:
                # Handle JSON input (could be used for advanced parameters)
                data = {
                    "model": self.model_name,
                    "prompt": content.get("prompt", ""),
                    "stream": True,
                    **content  # Include any additional parameters
                }

            logger.debug(f"Sending request to Ollama: {data}")  # Debug log
            
            async with self.session.post(f"{self.base_url}/api/generate", json=data) as response:
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(f"Ollama API error: {error_text}")
                    return

                # Stream the responses
                async for line in response.content:
                    if not line:
                        continue
                        
                    try:
                        result = json.loads(line)
                        logger.debug(f"Received response from Ollama: {result}")  # Debug log
                        
                        # Create model output from response
                        output = ModelOutput(
                            content=result.get("response", ""),
                            output_type="text",
                            timestamp=time.time(),
                            metadata={
                                "done": result.get("done", False),
                                "total_duration": result.get("total_duration"),
                                "eval_count": result.get("eval_count"),
                                "eval_duration": result.get("eval_duration")
                            }
                        )
                        
                        await self._output_queue.put(output)
                        
                        # If this is the final response, we're done
                        if result.get("done", False):
                            break
                            
                    except json.JSONDecodeError as e:
                        logger.error(f"Failed to parse Ollama response: {e}")
                        continue

        except Exception as e:
            logger.error(f"Error communicating with Ollama: {e}")
            # Send error message as output
            error_output = ModelOutput(
                content=f"Error: {str(e)}",
                output_type="error",
                timestamp=time.time()
            )
            await self._output_queue.put(error_output)

def create_model_interface(model_type: str, config: Optional[ServerConfig] = None) -> ModelInterface:
    """
    Factory function to create the appropriate model interface based on type.
    
    Args:
        model_type: Type of model ("generic", "ollama", etc.)
        config: Optional configuration object
        
    Returns:
        ModelInterface: Appropriate interface instance for the model type
    """
    interfaces = {
        "generic": GenericModelInterface,
        "ollama": OllamaInterface,
        # Add more model types here
    }
    
    interface_class = interfaces.get(model_type.lower(), GenericModelInterface)
    return interface_class(config)