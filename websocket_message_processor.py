import json
from typing import Optional, Dict, Any, Union
import logging
from websockets.server import WebSocketServerProtocol
from xss_protection import XSSProtector, XSSProtectionMode

logger = logging.getLogger(__name__)

# Initialize XSS protector in STRICT mode for chat messages
xss_protector = XSSProtector(mode=XSSProtectionMode.STRICT)

async def process_websocket_message(
    websocket: WebSocketServerProtocol,
    message: str
) -> Optional[str]:
    """Process incoming WebSocket messages"""
    try:
        # Parse message
        data = json.loads(message)
        
        # Validate request for XSS attempts
        if not xss_protector.validate_request(
            headers=getattr(websocket.request_headers, {}) if hasattr(websocket, 'request_headers') else {},
            body=data,
            query_params={}
        ):
            return json.dumps({
                "error": "Invalid request content detected"
            })
        
        # Get authenticated user
        user = getattr(websocket, 'user', None)
        if not user:
            return json.dumps({
                "error": "Authentication required"
            })
            
        # Process message based on type
        if data.get("type") == "subscribe":
            return await handle_subscription(websocket, data, user)
        elif data.get("type") == "message":
            return await handle_chat_message(websocket, data, user)
        else:
            return json.dumps({
                "error": "Unknown message type"
            })
            
    except json.JSONDecodeError:
        return json.dumps({
            "error": "Invalid JSON format"
        })
    except Exception as e:
        logger.error(f"Error processing message: {str(e)}")
        return json.dumps({
            "error": f"Error processing message: {str(e)}"
        })

async def handle_subscription(
    websocket: WebSocketServerProtocol,
    data: Dict[str, Any],
    user: Dict[str, Any]
) -> str:
    """Handle subscription requests"""
    try:
        # Sanitize channel name
        channel = xss_protector.sanitize_input(data.get("channel"), context='text')
        if not channel:
            return json.dumps({
                "error": "Channel not specified or invalid"
            })
            
        # Add user to channel subscribers
        if not hasattr(websocket, 'subscriptions'):
            websocket.subscriptions = set()
        websocket.subscriptions.add(channel)
        
        return json.dumps({
            "type": "subscription_confirmed",
            "channel": channel
        })
        
    except Exception as e:
        logger.error(f"Error handling subscription: {str(e)}")
        return json.dumps({
            "error": f"Error handling subscription: {str(e)}"
        })

async def handle_chat_message(
    websocket: WebSocketServerProtocol,
    data: Dict[str, Any],
    user: Dict[str, Any]
) -> str:
    """Handle chat messages"""
    try:
        # Sanitize message content - allow some HTML formatting but strictly controlled
        message_content = xss_protector.sanitize_input(data.get("message"), context='html')
        if not message_content:
            return json.dumps({
                "error": "Message content is required or invalid"
            })
            
        # Sanitize channel name
        channel = xss_protector.sanitize_input(data.get("channel"), context='text')
        if not channel:
            return json.dumps({
                "error": "Channel not specified or invalid"
            })
            
        # Verify user is subscribed to channel
        if not hasattr(websocket, 'subscriptions') or channel not in websocket.subscriptions:
            return json.dumps({
                "error": "Not subscribed to channel"
            })
            
        # Create message payload with sanitized content
        message = {
            "type": "chat_message",
            "channel": channel,
            "user": xss_protector.sanitize_input(user.get("username", "anonymous"), context='text'),
            "message": message_content,
            "timestamp": data.get("timestamp")
        }
        
        return json.dumps(message)
        
    except Exception as e:
        logger.error(f"Error handling chat message: {str(e)}")
        return json.dumps({
            "error": f"Error handling chat message: {str(e)}"
        }) 