"""Input type definitions."""

from enum import Enum

class InputType(Enum):
    """Types of input that require validation."""
    VIDEO_FILE = "video_file"
    IMAGE_FILE = "image_file"
    AUDIO_FILE = "audio_file"
    JSON_DATA = "json_data"
    USERNAME = "username"
    PASSWORD = "password"
    EMAIL = "email"
    URL = "url"
    PATH = "path"
    TIMESTAMP = "timestamp"
    NUMERIC = "numeric"
    TEXT = "text" 