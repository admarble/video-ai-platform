import json
from typing import Optional, Dict, Any, Union
import logging
from websockets.server import WebSocketServerProtocol
from src.security.xss.xss_protection import XSSProtector, XSSProtectionMode

logger = logging.getLogger(__name__) 