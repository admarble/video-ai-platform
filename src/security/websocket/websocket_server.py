import asyncio
import ssl
import logging
from pathlib import Path
from websockets.server import serve

from cors_config import CORSConfig, CORSMode
from websocket_cors import WebSocketRule, WebSocketProtocol, WebSocketManager, websocket_handler
from src.auth.auth_manager import AuthenticationManager 