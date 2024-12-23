import asyncio
import ssl
import logging
from pathlib import Path
from websockets.server import serve

from cors_config import CORSConfig, CORSMode
from websocket_cors import WebSocketRule, WebSocketProtocol, WebSocketManager, websocket_handler
from auth_manager import AuthenticationManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_ssl_context() -> ssl.SSLContext:
    """Create SSL context for secure WebSocket connections"""
    ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    
    # Update paths to your SSL certificate and key
    ssl_context.load_cert_chain(
        Path("path/to/cert.pem"),
        Path("path/to/key.pem")
    )
    
    return ssl_context

async def main():
    # Create authentication manager (implement your own or use existing)
    auth_manager = AuthenticationManager()
    
    # Create CORS config with WebSocket rule
    cors_config = CORSConfig(mode=CORSMode.SECURE)
    cors_config.add_rule("websocket_api", WebSocketRule(
        allowed_origins=["https://app.example.com"],
        allowed_methods=["GET", "POST"],
        allowed_headers=["Content-Type", "Authorization"],
        allow_credentials=True,
        require_auth=True,
        allowed_roles=["user", "admin"],
        # WebSocket specific settings
        allowed_protocols=[WebSocketProtocol.WSS],  # Secure WebSocket only
        ping_interval=30,
        max_message_size=1024 * 1024,  # 1MB
        max_connections_per_client=5,
        heartbeat_required=True,
        allowed_subprotocols=["v1.notification", "v1.chat"]
    ))
    
    # Create WebSocket manager
    ws_manager = WebSocketManager(cors_config, auth_manager)
    
    # Create SSL context
    ssl_context = create_ssl_context()
    
    async def handler(websocket, path):
        await websocket_handler(websocket, path, ws_manager)
    
    # Start WebSocket server
    async with serve(
        handler,
        "localhost",
        8765,
        ping_interval=20,
        ping_timeout=10,
        ssl=ssl_context,
        max_size=1024 * 1024,  # 1MB max message size
        max_queue=32  # Max queued messages
    ):
        logger.info("WebSocket server started on wss://localhost:8765")
        await asyncio.Future()  # run forever

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Server error: {str(e)}") 