DEFAULT_CONFIG = {
    'scene_processor': {
        'min_scene_length': 30,
        'threshold': 0.3
    },
    'object_detector': {
        'confidence_threshold': 0.5,
        'enable_tracking': True
    },
    'audio_processor': {
        'sample_rate': 44100,
        'channels': 2
    },
    "alert_config": {
        "max_errors_per_hour": 100,
        "max_avg_processing_time": 300,
        "max_cpu_percent": 90,
        "max_memory_percent": 85,
        "max_disk_percent": 85,
        "email": {
            "smtp_host": "smtp.example.com",
            "smtp_port": 587,
            "use_tls": True,
            "username": "alerts@example.com",
            "password": "",  # Set via environment variable
            "from": "alerts@example.com",
            "to": ["team@example.com"]
        },
        "slack": {
            "webhook_url": ""  # Set via environment variable
        }
    },
    "alert_history_path": "data/alert_history.json",
    "monitoring_config": {
        "metrics": {
            "error_rate": {
                "threshold": 10.0,
                "window_minutes": 5
            },
            "processing_time": {
                "threshold": 300.0,
                "window_minutes": 15
            },
            "cpu_usage": {
                "threshold": 90.0,
                "window_minutes": 5
            },
            "memory_usage": {
                "threshold": 85.0,
                "window_minutes": 5
            },
            "gpu_usage": {
                "threshold": 95.0,
                "window_minutes": 5
            }
        },
        "email": {
            "smtp_host": "smtp.example.com",
            "smtp_port": 587,
            "use_tls": True,
            "username": "alerts@example.com",
            "password": "",  # Set via environment variable
            "from": "alerts@example.com",
            "to": ["team@example.com"]
        },
        "slack": {
            "webhook_url": ""  # Set via environment variable
        }
    }
}

# Add logging configuration
LOGGING_CONFIG = {
    'log_dir': 'logs',
    'retention_days': 30,
    'max_queue_size': 10000,
    'file_max_bytes': 10 * 1024 * 1024,  # 10MB
    'backup_count': 5
} 