# Retryable Monitoring System

A robust monitoring system with built-in retry capabilities for handling transient failures in metric collection and alert notifications.

## Project Status

✅ Core monitoring functionality
✅ Retry mechanisms with multiple strategies
✅ Alert management system
✅ Metric collection and storage
✅ Configuration management
⬜️ Dashboard UI
⬜️ Real-time monitoring
⬜️ Advanced analytics
⬜️ Integration tests
⬜️ Documentation website

## Features

- Multiple retry strategies:
  - Fixed delay
  - Exponential backoff
  - Random delay with jitter
- Configurable retry parameters
- Support for various alert channels:
  - Email
  - Slack
  - Custom webhooks
- Metric collection and storage
- Automatic cleanup of old metrics
- Health checking capabilities

## Installation

1. Clone the repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

```python
from pathlib import Path
from monitoring import create_retryable_monitoring

# Configuration for alert channels
config = {
    'thresholds': [
        {
            'metric_path': 'cpu.percent',
            'max_value': 90
        },
        {
            'metric_path': 'memory.percent',
            'max_value': 85
        }
    ],
    'email': {
        'from': 'alerts@example.com',
        'to': 'admin@example.com',
        'smtp_host': 'smtp.example.com',
        'smtp_port': 587,
        'username': 'alerts@example.com',
        'password': 'your-password',
        'use_tls': True
    },
    'slack': {
        'webhook_url': 'https://hooks.slack.com/services/YOUR/WEBHOOK/URL',
        'channel': '#monitoring'
    }
}

# Create monitoring system
monitoring = create_retryable_monitoring(
    base_dir=Path('/path/to/metrics'),
    config=config
)

# Collect metrics (with automatic retries)
metrics = monitoring.collect_system_metrics()

# Process alerts if thresholds are breached
monitoring.process_alerts(metrics)

# Update metrics for a specific video processing job
monitoring.update_metrics('video123', {
    'processing_time': 45.2,
    'quality_score': 0.98
})

# Cleanup when done
monitoring.cleanup()
```

### Custom Retry Configurations

You can create custom retry configurations for different scenarios:

```python
from retry import RetryConfig, RetryStrategy

custom_retry = RetryConfig(
    max_attempts=5,
    initial_delay=2.0,
    max_delay=30.0,
    strategy=RetryStrategy.EXPONENTIAL,
    jitter=True,
    exceptions=(ConnectionError, TimeoutError)
)
```

## Configuration

### Alert Thresholds

Thresholds are configured using dot notation to access nested metrics:

```python
{
    'thresholds': [
        {
            'metric_path': 'cpu.percent',
            'max_value': 90  # Alert when CPU usage > 90%
        },
        {
            'metric_path': 'memory.percent',
            'min_value': 10,  # Alert when free memory < 10%
            'max_value': 85   # Alert when used memory > 85%
        }
    ]
}
```

### Alert Channels

#### Email Configuration

```python
{
    'email': {
        'from': 'alerts@example.com',
        'to': 'admin@example.com',
        'smtp_host': 'smtp.example.com',
        'smtp_port': 587,
        'username': 'alerts@example.com',
        'password': 'your-password',
        'use_tls': True
    }
}
```

#### Slack Configuration

```python
{
    'slack': {
        'webhook_url': 'https://hooks.slack.com/services/YOUR/WEBHOOK/URL',
        'channel': '#monitoring'
    }
}
```

#### Webhook Configuration

```python
{
    'webhook': {
        'url': 'https://api.example.com/alerts',
        'headers': {
            'Authorization': 'Bearer your-token'
        }
    }
}
```

## License

MIT 

# Security Components

This repository contains various security components for web applications.

## XSS Protection

The XSS Protection module provides robust protection against Cross-Site Scripting (XSS) attacks. It supports three operational modes:

- **STRICT**: Most secure mode that only allows a limited set of safe HTML tags and attributes
- **SANITIZE**: More permissive mode that allows more HTML tags and attributes while still maintaining security
- **ESCAPE**: Most restrictive mode that escapes all HTML

### Usage

```python
from xss_protection import XSSProtector, XSSProtectionMode

# Create an XSS protector instance
xss_protector = XSSProtector(mode=XSSProtectionMode.STRICT)

# Sanitize text input
clean_text = xss_protector.sanitize_input("Hello <script>alert('xss')</script>", context='text')

# Sanitize HTML content
clean_html = xss_protector.sanitize_input(
    '<p>Hello</p><script>alert("xss")</script>',
    context='html'
)

# Sanitize URLs
clean_url = xss_protector.sanitize_input(
    "javascript:alert('xss')",
    context='url'
)

# Validate complete requests
is_safe = xss_protector.validate_request(
    headers=request.headers,
    body=request.body,
    query_params=request.query_params
)
```

### Features

- Context-aware sanitization (text, HTML, URL, JavaScript)
- Configurable allowed HTML tags and attributes
- URL sanitization with protocol validation
- JSON data sanitization
- Request validation for headers, body, and query parameters
- Protection against:
  - Script injection
  - Event handler injection
  - JavaScript protocol abuse
  - Data URL exploitation
  - Expression injection
  - Malformed HTML
  - Unclosed tags
  - Dangerous attributes

### Integration with WebSocket Server

The XSS protection is integrated with the WebSocket server to protect against XSS attacks in real-time messaging:

```python
# In websocket_message_processor.py
from xss_protection import XSSProtector, XSSProtectionMode

xss_protector = XSSProtector(mode=XSSProtectionMode.STRICT)

async def handle_chat_message(websocket, data, user):
    # Sanitize message content
    message_content = xss_protector.sanitize_input(
        data.get("message"),
        context='html'
    )
    
    # Sanitize channel name
    channel = xss_protector.sanitize_input(
        data.get("channel"),
        context='text'
    )
    
    # Create sanitized message payload
    message = {
        "type": "chat_message",
        "channel": channel,
        "user": xss_protector.sanitize_input(
            user.get("username", "anonymous"),
            context='text'
        ),
        "message": message_content,
        "timestamp": data.get("timestamp")
    }
    
    return json.dumps(message)
```

### Testing

The XSS protection includes comprehensive tests covering all modes and contexts. Run the tests using pytest:

```bash
pytest tests/test_xss_protection.py
``` 

### Integration with Video Processing

The XSS protection module is integrated with the secure video processor to protect against XSS attacks in video metadata and processing:

```python
from secure_video_processor import SecureVideoProcessorWithXSS, SecurityLevel
from cors_config import CORSConfig

# Create CORS config
cors_config = CORSConfig(allowed_origins=['https://example.com'])

# Create secure video processor with XSS protection
processor = SecureVideoProcessorWithXSS(
    cors_config=cors_config,
    security_level=SecurityLevel.HIGH  # Uses STRICT XSS mode
)

# Process video with XSS protection
result = await processor.process_video(
    video_path="video.mp4",
    metadata={
        "title": "My Video",
        "description": "Video description"
    },
    request_origin="https://example.com"
)

# Validate file upload with XSS protection
is_valid = processor.validate_upload({
    'content_type': 'video/mp4',
    'size': 1024 * 1024,
    'filename': 'video.mp4'
})
```

The secure video processor provides:
- XSS protection for video metadata
- Sanitization of file paths and names
- Protection against malicious file uploads
- Security headers including CSP and XSS protection
- Multiple security levels (HIGH, MEDIUM, LOW)
- Integration with CORS configuration

Security levels map to XSS protection modes:
- HIGH → STRICT mode (most secure)
- MEDIUM → SANITIZE mode (balanced)
- LOW → ESCAPE mode (most permissive)

You can also specify a custom XSS mode:
```python
from xss_protection import XSSProtectionMode

processor = SecureVideoProcessorWithXSS(
    cors_config=cors_config,
    security_level=SecurityLevel.HIGH,
    xss_mode=XSSProtectionMode.SANITIZE  # Override default mode
)
``` 