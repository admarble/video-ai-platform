from typing import Dict, List, Optional, Set, Any, Union
from dataclasses import dataclass
import logging
import re
import json
from enum import Enum
from pathlib import Path
import asyncio
import websockets
from websockets.exceptions import InvalidHandshake
from cors_config import CORSConfig, CORSRule, CORSMode
from auth_manager import AuthenticationManager
from websocket_message_processor import process_websocket_message

class WebSocketProtocol(Enum):
    """Supported WebSocket protocols"""
    WS = "ws"
    WSS = "wss"

@dataclass
class WebSocketRule(CORSRule):
    """Extended CORS rule for WebSocket connections"""
    allowed_protocols: List[WebSocketProtocol]
    ping_interval: int = 30  # seconds
    max_message_size: int = 1024 * 1024  # 1MB
    max_connections_per_client: int = 5
    heartbeat_required: bool = True
    allowed_subprotocols: Optional[List[str]] = None

class WebSocketManager:
    """Manages WebSocket connections with CORS support"""
    
    def __init__(
        self,
        cors_config: CORSConfig,
        auth_manager: Optional['AuthenticationManager'] = None
    ):
        self.cors_config = cors_config
        self.auth_manager = auth_manager
        self.logger = logging.getLogger(__name__)
        self.active_connections: Dict[str, Set[websockets.WebSocketServerProtocol]] = {}
        self.connection_counts: Dict[str, int] = {}
        
    async def handle_websocket_request(
        self,
        websocket: websockets.WebSocketServerProtocol,
        path: str,
        rule_name: str = "default"
    ) -> bool:
        """Handle incoming WebSocket connection request"""
        try:
            # Get WebSocket rule
            rule = self.cors_config.get_rule(rule_name)
            if not isinstance(rule, WebSocketRule):
                self.logger.error(f"Invalid rule type for WebSocket: {rule_name}")
                return False
                
            # Verify origin
            origin = websocket.origin if hasattr(websocket, 'origin') else None
            if not origin or not self.cors_config.is_origin_allowed(origin, rule_name):
                self.logger.warning(f"Rejected WebSocket connection from origin: {origin}")
                return False
                
            # Check protocol
            protocol = WebSocketProtocol.WSS if websocket.secure else WebSocketProtocol.WS
            if protocol not in rule.allowed_protocols:
                self.logger.warning(f"Rejected WebSocket connection with protocol: {protocol}")
                return False
                
            # Verify authentication if required
            if rule.require_auth:
                auth_header = websocket.request_headers.get('Authorization')
                if not auth_header:
                    self.logger.warning("Missing authentication for protected WebSocket")
                    return False
                    
                if self.auth_manager:
                    user = await self.auth_manager.verify_token(auth_header)
                    if not user or not self._check_roles(user, rule.allowed_roles):
                        return False
                        
            # Check connection limits
            client_id = self._get_client_id(websocket)
            if not self._check_connection_limit(client_id, rule):
                return False
                
            # Setup connection
            await self._setup_connection(websocket, client_id, rule)
            return True
            
        except Exception as e:
            self.logger.error(f"Error handling WebSocket request: {str(e)}")
            return False
            
    def _get_client_id(self, websocket: websockets.WebSocketServerProtocol) -> str:
        """Get unique client identifier"""
        if self.auth_manager and hasattr(websocket, 'user'):
            return f"user_{websocket.user.id}"
        return f"ip_{websocket.remote_address[0]}"
        
    def _check_connection_limit(self, client_id: str, rule: WebSocketRule) -> bool:
        """Check if client has exceeded connection limit"""
        current_count = self.connection_counts.get(client_id, 0)
        if current_count >= rule.max_connections_per_client:
            self.logger.warning(f"Connection limit exceeded for client: {client_id}")
            return False
        return True
        
    async def _setup_connection(
        self,
        websocket: websockets.WebSocketServerProtocol,
        client_id: str,
        rule: WebSocketRule
    ) -> None:
        """Setup new WebSocket connection"""
        # Add to active connections
        if client_id not in self.active_connections:
            self.active_connections[client_id] = set()
        self.active_connections[client_id].add(websocket)
        
        # Update connection count
        self.connection_counts[client_id] = self.connection_counts.get(client_id, 0) + 1
        
        # Setup heartbeat if required
        if rule.heartbeat_required:
            asyncio.create_task(self._heartbeat_monitor(websocket, rule.ping_interval))
            
    async def _heartbeat_monitor(
        self,
        websocket: websockets.WebSocketServerProtocol,
        interval: int
    ) -> None:
        """Monitor connection with periodic pings"""
        try:
            while True:
                await asyncio.sleep(interval)
                try:
                    pong_waiter = await websocket.ping()
                    await asyncio.wait_for(pong_waiter, timeout=5)
                except asyncio.TimeoutError:
                    self.logger.warning("WebSocket heartbeat failed")
                    break
        except Exception as e:
            self.logger.error(f"Error in heartbeat monitor: {str(e)}")
        finally:
            await self._cleanup_connection(websocket)
            
    async def _cleanup_connection(
        self,
        websocket: websockets.WebSocketServerProtocol
    ) -> None:
        """Clean up closed connection"""
        try:
            # Find and remove connection
            client_id = self._get_client_id(websocket)
            if client_id in self.active_connections:
                self.active_connections[client_id].discard(websocket)
                if not self.active_connections[client_id]:
                    del self.active_connections[client_id]
                    
            # Update connection count
            if client_id in self.connection_counts:
                self.connection_counts[client_id] -= 1
                if self.connection_counts[client_id] <= 0:
                    del self.connection_counts[client_id]
                    
        except Exception as e:
            self.logger.error(f"Error cleaning up connection: {str(e)}")
            
    def _check_roles(self, user: Any, allowed_roles: Optional[List[str]]) -> bool:
        """Check if user has required roles"""
        if not allowed_roles:
            return True
        return any(role in user.roles for role in allowed_roles)
            
    async def broadcast_message(
        self,
        message: Union[str, bytes],
        rule_name: str = "default"
    ) -> None:
        """Broadcast message to all connected clients under a rule"""
        rule = self.cors_config.get_rule(rule_name)
        if not isinstance(rule, WebSocketRule):
            return
            
        for connections in self.active_connections.values():
            for websocket in connections:
                try:
                    await websocket.send(message)
                except Exception as e:
                    self.logger.error(f"Error broadcasting message: {str(e)}")
                    await self._cleanup_connection(websocket)
                    
    async def send_message(
        self,
        client_id: str,
        message: Union[str, bytes]
    ) -> bool:
        """Send message to specific client"""
        if client_id not in self.active_connections:
            return False
            
        success = False
        for websocket in self.active_connections[client_id]:
            try:
                await websocket.send(message)
                success = True
            except Exception as e:
                self.logger.error(f"Error sending message: {str(e)}")
                await self._cleanup_connection(websocket)
                
        return success

# Example WebSocket handler function
async def websocket_handler(
    websocket: websockets.WebSocketServerProtocol,
    path: str,
    ws_manager: WebSocketManager
) -> None:
    """Handle WebSocket connection"""
    if not await ws_manager.handle_websocket_request(websocket, path):
        await websocket.close(1008)  # Policy violation
        return
        
    try:
        async for message in websocket:
            # Process message
            response = await process_websocket_message(websocket, message)
            if response:
                await websocket.send(response)
    except Exception as e:
        ws_manager.logger.error(f"Error in WebSocket handler: {str(e)}")
    finally:
        await ws_manager._cleanup_connection(websocket) 