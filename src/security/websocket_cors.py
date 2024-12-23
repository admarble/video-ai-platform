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
from .cors.cors_config import CORSConfig, CORSRule, CORSMode
from .auth.auth_manager import AuthenticationManager
from .websocket.websocket_message_processor import process_websocket_message

# Rest of the file remains unchanged... 