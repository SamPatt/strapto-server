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
    """
    Implementation of ModelInterface for Ollama.
    """
    
    async def connect(self) -> bool:
        """Connect to local Ollama instance."""
        try:
            # TODO: Implement Ollama-specific connection logic
            self.ready = True
            logger.info("Connected to Ollama")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to Ollama: {e}")
            return False

    async def disconnect(self) -> None:
        """Disconnect from Ollama."""
        # TODO: Implement cleanup logic
        self.ready = False
        logger.info("Disconnected from Ollama")

    async def send_input(self, content: Union[str, Dict[str, Any]]) -> None:
        """Send prompt to Ollama and stream responses."""
        # TODO: Implement Ollama API interaction
        pass

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