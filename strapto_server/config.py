"""
Configuration management for the StrapTo Local Server.

This module handles all configuration aspects of the StrapTo Local Server, including:
- Loading environment variables
- Reading configuration files
- Setting default values
- Validating configuration parameters

The configuration is centralized here to make it easier to modify settings and
ensure consistency across the application.
"""

import os
from dataclasses import dataclass
from typing import Optional
import json
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ServerConfig:
    """
    Dataclass to store all server configuration parameters.
    Using a dataclass provides type hints and makes the configuration immutable.
    
    Required parameters come first, followed by optional parameters with defaults.
    """
    # Required parameters (no defaults)
    webrtc_host: str
    webrtc_port: int
    stun_server: str
    signaling_url: str
    
    # Optional parameters (with defaults)
    model_host: str = "localhost"
    model_port: int = 8080
    model_type: str = "generic"  # e.g., "ollama", "llamacpp", "lmstudio"
    turn_server: Optional[str] = None
    room_id: Optional[str] = None
    max_connections: int = 10
    enable_chat: bool = True
    enable_suggestions: bool = True
    buffer_size: int = 1024 * 1024  # 1MB buffer for output streaming
    model_name: Optional[str] = None
    host: str = "localhost"
    port: int = 8000
    debug: bool = False

def load_config(config_path: Optional[str] = None) -> ServerConfig:
    """
    Load configuration from environment variables and/or config file.
    Environment variables take precedence over config file values.
    
    Args:
        config_path: Optional path to a JSON configuration file
        
    Returns:
        ServerConfig: Configuration object with all settings
    """
    # Default configuration
    config_dict = {
        "webrtc_host": "0.0.0.0",
        "webrtc_port": 8765,
        "stun_server": "stun:stun.l.google.com:19302",
        "signaling_url": "wss://signaling.strapto.dev",
        "model_host": "localhost",
        "model_port": 8080,
        "model_type": "generic"
    }
    
    # Load from config file if provided
    if config_path and os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                file_config = json.load(f)
                config_dict.update(file_config)
                logger.info(f"Loaded configuration from {config_path}")
        except Exception as e:
            logger.error(f"Error loading config file: {e}")
    
    # Environment variables take precedence
    env_mapping = {
        "STRAPTO_WEBRTC_HOST": "webrtc_host",
        "STRAPTO_WEBRTC_PORT": "webrtc_port",
        "STRAPTO_STUN_SERVER": "stun_server",
        "STRAPTO_TURN_SERVER": "turn_server",
        "STRAPTO_SIGNALING_URL": "signaling_url",
        "STRAPTO_ROOM_ID": "room_id",
        "STRAPTO_MODEL_HOST": "model_host",
        "STRAPTO_MODEL_PORT": "model_port",
        "STRAPTO_MODEL_TYPE": "model_type",
        "STRAPTO_MAX_CONNECTIONS": "max_connections",
        "STRAPTO_ENABLE_CHAT": "enable_chat",
        "STRAPTO_ENABLE_SUGGESTIONS": "enable_suggestions",
        "STRAPTO_BUFFER_SIZE": "buffer_size",
        "STRAPTO_MODEL_NAME": "model_name",
        "STRAPTO_HOST": "host",
        "STRAPTO_PORT": "port",
        "STRAPTO_DEBUG": "debug"
    }
    
    # Update config with environment variables
    for env_var, config_key in env_mapping.items():
        if env_value := os.getenv(env_var):
            # Convert string values to appropriate types
            if config_key in ["webrtc_port", "model_port", "max_connections", "buffer_size", "port"]:
                config_dict[config_key] = int(env_value)
            elif config_key in ["enable_chat", "enable_suggestions", "debug"]:
                config_dict[config_key] = env_value.lower() == "true"
            else:
                config_dict[config_key] = env_value
    
    # Create and return the configuration object
    try:
        config = ServerConfig(**config_dict)
        logger.info("Configuration loaded successfully")
        return config
    except Exception as e:
        logger.error(f"Error creating configuration object: {e}")
        raise

def get_config(config_path: Optional[str] = None) -> ServerConfig:
    """
    Singleton-like function to get the server configuration.
    This ensures we're using the same configuration throughout the application.
    
    Args:
        config_path: Optional path to a JSON configuration file
        
    Returns:
        ServerConfig: Configuration object with all settings
    """
    if not hasattr(get_config, "_config"):
        get_config._config = load_config(config_path)
    return get_config._config