"""
WebRTC Manager for the StrapTo Local Server.

This module handles WebRTC connections and data channels using aiortc.
It manages:
- Connection establishment and ICE negotiation (both client and server initiated)
- Data channel creation and management
- Message streaming and handling
"""

import asyncio
import json
import logging
from typing import Any, Dict, Optional, Set, Tuple
from dataclasses import dataclass
from datetime import datetime

from aiortc import (
    RTCPeerConnection,
    RTCSessionDescription,
    RTCIceCandidate,
    RTCConfiguration,
    RTCIceServer,
    RTCDataChannel
)
from aiortc.contrib.signaling import object_from_string, object_to_string

from .config import ServerConfig, get_config
from .event_handler import Event, EventEmitter

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class PeerMetrics:
    """Metrics for a peer connection."""
    connected_at: datetime
    messages_sent: int = 0
    messages_received: int = 0
    bytes_sent: int = 0
    bytes_received: int = 0
    last_activity: Optional[datetime] = None

@dataclass
class PeerInfo:
    """Information about a connected peer."""
    connection: RTCPeerConnection
    data_channel: Optional[RTCDataChannel] = None
    ice_candidates: Set[str] = None  # Store candidates until remote description is set
    metrics: Optional[PeerMetrics] = None
    
    def __post_init__(self):
        if self.ice_candidates is None:
            self.ice_candidates = set()
        if self.metrics is None:
            self.metrics = PeerMetrics(connected_at=datetime.now())

class WebRTCManager:
    """
    Manages WebRTC connections and data channels for streaming model outputs.
    """
    
    def __init__(
        self,
        config: Optional[ServerConfig] = None,
        event_emitter: Optional[EventEmitter] = None
    ):
        """
        Initialize the WebRTC manager.
        
        Args:
            config: Server configuration
            event_emitter: Event emitter for handling messages
        """
        self.config = config or get_config()
        self.event_emitter = event_emitter or EventEmitter()
        self.peers: Dict[str, PeerInfo] = {}
        
        # Configure ICE servers
        ice_servers = [
            RTCIceServer(urls=self.config.stun_server)
        ]
        if self.config.turn_server:
            ice_servers.append(RTCIceServer(urls=self.config.turn_server))
        
        self.rtc_config = RTCConfiguration(iceServers=ice_servers)
        
        # Register event handlers
        self.event_emitter.add_listener("model_output", self._handle_model_output)
        logger.info("WebRTC manager initialized with ICE servers")

    def _create_peer_connection(self, peer_id: str) -> RTCPeerConnection:
        """
        Create a new peer connection with appropriate handlers.
        
        Args:
            peer_id: Unique identifier for the peer
            
        Returns:
            RTCPeerConnection: The created peer connection
        """
        pc = RTCPeerConnection(configuration=self.rtc_config)
        
        @pc.on("datachannel")
        def on_datachannel(channel):
            logger.info(f"Data channel established for peer {peer_id}")
            if peer_id in self.peers:
                self.peers[peer_id].data_channel = channel
                self._setup_data_channel(channel, peer_id)
        
        @pc.on("connectionstatechange")
        async def on_connectionstatechange():
            logger.info(f"Connection state changed for peer {peer_id}: {pc.connectionState}")
            if pc.connectionState == "failed":
                await self.close_peer_connection(peer_id)
            elif pc.connectionState == "closed":
                if peer_id in self.peers:
                    await self.close_peer_connection(peer_id)
        
        @pc.on("iceconnectionstatechange")
        async def on_iceconnectionstatechange():
            logger.info(f"ICE connection state for peer {peer_id}: {pc.iceConnectionState}")
            if pc.iceConnectionState == "failed":
                await pc.close()
                await self.close_peer_connection(peer_id)

        @pc.on("icegatheringstatechange")
        async def on_icegatheringstatechange():
            logger.info(f"ICE gathering state for peer {peer_id}: {pc.iceGatheringState}")
        
        return pc

    def _setup_data_channel(self, channel: RTCDataChannel, peer_id: str) -> None:
        """
        Set up handlers for a data channel.
        
        Args:
            channel: The data channel to set up
            peer_id: ID of the peer this channel belongs to
        """
        @channel.on("message")
        async def on_message(message):
            try:
                if isinstance(message, str):
                    # Update metrics
                    if peer_id in self.peers:
                        metrics = self.peers[peer_id].metrics
                        metrics.messages_received += 1
                        metrics.bytes_received += len(message.encode())
                        metrics.last_activity = datetime.now()
                    
                    # Log raw message in debug mode
                    logger.debug(f"Raw message from peer {peer_id}: {message[:200]}...")
                    
                    data = json.loads(message)
                    event = Event(
                        type=data.get("type", "consumer_message"),
                        data=data.get("data"),
                        metadata={"peer_id": peer_id}
                    )
                    await self.event_emitter.emit(event)
                else:
                    logger.warning(f"Received non-string message from peer {peer_id}: {type(message)}")
            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON from peer {peer_id}. Error: {e}. Message preview: {message[:200]}...")
            except Exception as e:
                logger.error(f"Error processing message from peer {peer_id}: {e}")

        @channel.on("close")
        def on_close():
            logger.info(f"Data channel closed for peer {peer_id}")
            if peer_id in self.peers:
                self.peers[peer_id].data_channel = None

    async def create_connection(self, peer_id: str) -> Tuple[str, RTCSessionDescription]:
        """
        Create a new connection and generate an offer (server-initiated connection).
        
        Args:
            peer_id: ID to assign to the new peer
            
        Returns:
            Tuple[str, RTCSessionDescription]: The peer ID and the offer
        """
        pc = self._create_peer_connection(peer_id)
        channel = pc.createDataChannel("data")
        
        # Store peer info
        self.peers[peer_id] = PeerInfo(
            connection=pc,
            data_channel=channel
        )
        
        # Set up the data channel
        self._setup_data_channel(channel, peer_id)
        
        # Create and set local description (offer)
        offer = await pc.createOffer()
        await pc.setLocalDescription(offer)
        
        logger.info(f"Created new connection offer for peer {peer_id}")
        return peer_id, pc.localDescription

    async def process_answer(self, peer_id: str, answer_sdp: str) -> None:
        """
        Process an SDP answer from a peer.
        
        Args:
            peer_id: ID of the peer sending the answer
            answer_sdp: SDP answer string
        """
        if peer_id not in self.peers:
            raise ValueError(f"No pending connection for peer {peer_id}")
        
        try:
            answer = RTCSessionDescription(**json.loads(answer_sdp))
            await self.peers[peer_id].connection.setRemoteDescription(answer)
            
            # Apply any pending ICE candidates
            if pending_candidates := self.peers[peer_id].ice_candidates:
                for candidate_str in pending_candidates:
                    candidate = RTCIceCandidate(**json.loads(candidate_str))
                    await self.peers[peer_id].connection.addIceCandidate(candidate)
                self.peers[peer_id].ice_candidates.clear()
            
            logger.info(f"Processed answer from peer {peer_id}")
        except json.JSONDecodeError as e:
            logger.error(f"Invalid answer SDP from peer {peer_id}. Error: {e}")
            raise
        except Exception as e:
            logger.error(f"Error processing answer from peer {peer_id}: {e}")
            raise

    async def process_offer(self, peer_id: str, offer_sdp: str) -> str:
        """
        Process an SDP offer and create an answer (client-initiated connection).
        
        Args:
            peer_id: ID of the peer sending the offer
            offer_sdp: SDP offer string
            
        Returns:
            str: SDP answer string
        """
        try:
            # Create peer connection if it doesn't exist
            if peer_id not in self.peers:
                pc = self._create_peer_connection(peer_id)
                self.peers[peer_id] = PeerInfo(connection=pc)
            else:
                pc = self.peers[peer_id].connection
            
            # Set remote description
            offer = RTCSessionDescription(**json.loads(offer_sdp))
            await pc.setRemoteDescription(offer)
            
            # Apply any pending ICE candidates
            if pending_candidates := self.peers[peer_id].ice_candidates:
                for candidate_str in pending_candidates:
                    try:
                        candidate = RTCIceCandidate(**json.loads(candidate_str))
                        await pc.addIceCandidate(candidate)
                    except json.JSONDecodeError as e:
                        logger.error(f"Invalid ICE candidate for peer {peer_id}. Error: {e}")
                        continue
                self.peers[peer_id].ice_candidates.clear()
            
            # Create and set local description (answer)
            answer = await pc.createAnswer()
            await pc.setLocalDescription(answer)
            
            logger.info(f"Created answer for peer {peer_id}")
            return json.dumps({
                "sdp": pc.localDescription.sdp,
                "type": pc.localDescription.type
            })
            
        except json.JSONDecodeError as e:
            logger.error(f"Invalid offer SDP from peer {peer_id}. Error: {e}")
            raise
        except Exception as e:
            logger.error(f"Error processing offer from peer {peer_id}: {e}")
            raise
    
    async def handle_offer(self, sdp: str, client_id: str) -> Dict[str, str]:
        """
        Handle WebRTC offer from client (wrapper for process_offer with different signature).
        
        Args:
            sdp: SDP offer string
            client_id: Client identifier
            
        Returns:
            Dict containing answer SDP and type
        """
        answer_str = await self.process_offer(client_id, sdp)
        answer_dict = json.loads(answer_str)
        return answer_dict

    async def process_ice_candidate(self, peer_id: str, candidate: str) -> None:
        """
        Process an ICE candidate from a peer.
        
        Args:
            peer_id: ID of the peer sending the candidate
            candidate: ICE candidate string
        """
        if peer_id not in self.peers:
            logger.warning(f"Received ICE candidate for unknown peer {peer_id}")
            return
        
        try:
            # If remote description isn't set yet, store the candidate
            if not self.peers[peer_id].connection.remoteDescription:
                self.peers[peer_id].ice_candidates.add(candidate)
                return
            
            # Otherwise, add it immediately
            candidate_obj = RTCIceCandidate(**json.loads(candidate))
            await self.peers[peer_id].connection.addIceCandidate(candidate_obj)
            
        except json.JSONDecodeError as e:
            logger.error(f"Invalid ICE candidate from peer {peer_id}. Error: {e}. Candidate: {candidate}")
        except Exception as e:
            logger.error(f"Error processing ICE candidate from peer {peer_id}: {e}")
    
    async def handle_ice_candidate(self, candidate: str, sdp_mline_index: int, sdp_mid: str, client_id: str) -> None:
        """
        Handle ICE candidate from client (wrapper for process_ice_candidate with different signature).
        
        Args:
            candidate: ICE candidate string
            sdp_mline_index: SDP m-line index
            sdp_mid: SDP media ID
            client_id: Client identifier
        """
        # Build the candidate JSON string
        candidate_dict = {
            "candidate": candidate,
            "sdpMLineIndex": sdp_mline_index,
            "sdpMid": sdp_mid
        }
        candidate_str = json.dumps(candidate_dict)
        await self.process_ice_candidate(client_id, candidate_str)

    async def close_peer_connection(self, peer_id: str) -> None:
        """
        Close a peer connection and clean up resources.
        
        Args:
            peer_id: ID of the peer connection to close
        """
        if peer_info := self.peers.pop(peer_id, None):
            try:
                if peer_info.data_channel:
                    peer_info.data_channel.close()
                await peer_info.connection.close()
            except Exception as e:
                logger.error(f"Error closing peer connection {peer_id}: {e}")
        
        logger.info(f"Closed connection for peer {peer_id}")

    async def close_all_connections(self) -> None:
        """Close all peer connections and clean up."""
        close_tasks = [
            self.close_peer_connection(peer_id)
            for peer_id in list(self.peers.keys())
        ]
        if close_tasks:
            await asyncio.gather(*close_tasks, return_exceptions=True)
        logger.info("All connections closed")

    def get_metrics(self) -> Dict[str, PeerMetrics]:
        """
        Get metrics for all connected peers.
        
        Returns:
            Dict[str, PeerMetrics]: Mapping of peer IDs to their metrics
        """
        return {
            peer_id: peer_info.metrics
            for peer_id, peer_info in self.peers.items()
        }
    
    def get_active_channel_count(self) -> int:
        """
        Get the count of active data channels.
        
        Returns:
            int: Number of open data channels
        """
        return sum(
            1 for peer_info in self.peers.values()
            if peer_info.data_channel and peer_info.data_channel.readyState == "open"
        )
    
    async def disconnect(self) -> None:
        """Disconnect all peer connections."""
        await self.close_all_connections()

    async def _handle_model_output(self, event: Event) -> None:
        """
        Handle model output events by sending them to all connected peers.
        
        Args:
            event: The model output event to handle
        """
        message = json.dumps({
            "type": "model_output",
            "data": event.data,
            "timestamp": event.timestamp,
            "metadata": event.metadata
        })
        
        # Send to all connected peers with open data channels
        for peer_id, peer_info in self.peers.items():
            if (channel := peer_info.data_channel) and channel.readyState == "open":
                try:
                    channel.send(message)
                    
                    # Update metrics
                    peer_info.metrics.messages_sent += 1
                    peer_info.metrics.bytes_sent += len(message.encode())
                    peer_info.metrics.last_activity = datetime.now()
                    
                except Exception as e:
                    logger.error(f"Error sending to peer {peer_id}: {e}")
                    await self.close_peer_connection(peer_id)