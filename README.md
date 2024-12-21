# Retryable Monitoring System

A robust monitoring system with built-in retry capabilities for handling transient failures in metric collection and alert notifications.

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