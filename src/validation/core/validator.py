"""Core validation functionality."""

import re
import json
import logging
import mimetypes
import magic
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Union, Optional
from urllib.parse import urlparse

from .types import InputType
from .rules import ValidationRule
from .exceptions import ValidationError

class InputValidator:
    """Validates and sanitizes input data."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Initialize default validation rules
        self._setup_default_rules()
        
    def _setup_default_rules(self) -> None:
        """Setup default validation rules."""
        self.default_rules = {
            InputType.VIDEO_FILE: ValidationRule(
                input_type=InputType.VIDEO_FILE,
                allowed_mime_types=[
                    'video/mp4', 'video/mpeg', 'video/quicktime',
                    'video/x-msvideo', 'video/x-matroska'
                ],
                max_size_bytes=self.config.get('max_video_size', 1024 * 1024 * 1024),  # 1GB
                pattern=r'.*\.(mp4|avi|mov|mkv|mpeg)$'
            ),
            
            InputType.IMAGE_FILE: ValidationRule(
                input_type=InputType.IMAGE_FILE,
                allowed_mime_types=[
                    'image/jpeg', 'image/png', 'image/gif',
                    'image/webp', 'image/tiff'
                ],
                max_size_bytes=self.config.get('max_image_size', 50 * 1024 * 1024),  # 50MB
                pattern=r'.*\.(jpg|jpeg|png|gif|webp|tiff)$'
            ),
            
            InputType.AUDIO_FILE: ValidationRule(
                input_type=InputType.AUDIO_FILE,
                allowed_mime_types=[
                    'audio/mpeg', 'audio/wav', 'audio/ogg',
                    'audio/x-m4a', 'audio/aac'
                ],
                max_size_bytes=self.config.get('max_audio_size', 100 * 1024 * 1024),  # 100MB
                pattern=r'.*\.(mp3|wav|ogg|m4a|aac)$'
            ),
            
            InputType.JSON_DATA: ValidationRule(
                input_type=InputType.JSON_DATA,
                custom_validator=self._validate_json
            ),
            
            InputType.USERNAME: ValidationRule(
                input_type=InputType.USERNAME,
                min_length=3,
                max_length=32,
                pattern=r'^[a-zA-Z0-9_-]+$',
                sanitize=True
            ),
            
            InputType.PASSWORD: ValidationRule(
                input_type=InputType.PASSWORD,
                min_length=8,
                max_length=128,
                pattern=r'^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[@$!%*?&])[A-Za-z\d@$!%*?&]+$',
                sanitize=False  # Don't sanitize passwords
            ),
            
            InputType.EMAIL: ValidationRule(
                input_type=InputType.EMAIL,
                pattern=r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$',
                max_length=254,
                sanitize=True
            ),
            
            InputType.URL: ValidationRule(
                input_type=InputType.URL,
                custom_validator=self._validate_url,
                sanitize=True
            ),
            
            InputType.PATH: ValidationRule(
                input_type=InputType.PATH,
                custom_validator=self._validate_path,
                sanitize=True
            ),
            
            InputType.TIMESTAMP: ValidationRule(
                input_type=InputType.TIMESTAMP,
                pattern=r'^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d+)?(Z|[+-]\d{2}:?\d{2})?$',
                sanitize=True
            ),
            
            InputType.NUMERIC: ValidationRule(
                input_type=InputType.NUMERIC,
                pattern=r'^-?\d*\.?\d+$',
                sanitize=True
            ),
            
            InputType.TEXT: ValidationRule(
                input_type=InputType.TEXT,
                max_length=self.config.get('max_text_length', 1000),
                sanitize=True
            )
        }
        
    def validate(
        self,
        value: Any,
        input_type: InputType,
        custom_rule: Optional[ValidationRule] = None
    ) -> Any:
        """Validate and sanitize input value."""
        # Get validation rule
        rule = custom_rule or self.default_rules[input_type]
        
        # Check if required
        if rule.required and value is None:
            raise ValidationError(f"{input_type.value} is required")
            
        if value is None:
            return None
            
        try:
            # Type-specific validation
            if input_type in [InputType.VIDEO_FILE, InputType.IMAGE_FILE, InputType.AUDIO_FILE]:
                return self._validate_file(value, rule)
            
            # Convert to string for text-based validation
            if isinstance(value, (int, float)):
                value = str(value)
                
            # Length validation
            if rule.min_length is not None and len(value) < rule.min_length:
                raise ValidationError(
                    f"Value must be at least {rule.min_length} characters long"
                )
                
            if rule.max_length is not None and len(value) > rule.max_length:
                raise ValidationError(
                    f"Value must be at most {rule.max_length} characters long"
                )
                
            # Pattern validation
            if rule.pattern and not re.match(rule.pattern, value):
                raise ValidationError(f"Invalid format for {input_type.value}")
                
            # Allowed values validation
            if rule.allowed_values is not None and value not in rule.allowed_values:
                raise ValidationError(
                    f"Value must be one of: {', '.join(map(str, rule.allowed_values))}"
                )
                
            # Custom validation
            if rule.custom_validator and not rule.custom_validator(value):
                raise ValidationError(f"Invalid {input_type.value}")
                
            # Sanitize if required
            if rule.sanitize:
                value = self._sanitize_input(value, input_type)
                
            return value
            
        except ValidationError:
            raise
        except Exception as e:
            raise ValidationError(f"Validation error: {str(e)}")
            
    def _validate_file(self, file_path: Union[str, Path], rule: ValidationRule) -> Path:
        """Validate file input."""
        file_path = Path(file_path)
        
        # Check file exists
        if not file_path.exists():
            raise ValidationError("File does not exist")
            
        # Check file size
        if rule.max_size_bytes:
            size = file_path.stat().st_size
            if size > rule.max_size_bytes:
                raise ValidationError(
                    f"File size ({size} bytes) exceeds maximum allowed "
                    f"({rule.max_size_bytes} bytes)"
                )
                
        # Check file extension
        if rule.pattern and not re.match(rule.pattern, file_path.name.lower()):
            raise ValidationError("Invalid file extension")
            
        # Check MIME type
        if rule.allowed_mime_types:
            mime_type = magic.from_file(str(file_path), mime=True)
            if mime_type not in rule.allowed_mime_types:
                raise ValidationError(f"Invalid file type: {mime_type}")
                
        return file_path
        
    def _validate_json(self, value: Any) -> bool:
        """Validate JSON data."""
        try:
            if isinstance(value, str):
                json.loads(value)
            elif isinstance(value, (dict, list)):
                json.dumps(value)
            else:
                return False
            return True
        except Exception:
            return False
            
    def _validate_url(self, value: str) -> bool:
        """Validate URL format and scheme."""
        try:
            result = urlparse(value)
            return all([result.scheme, result.netloc])
        except Exception:
            return False
            
    def _validate_path(self, value: str) -> bool:
        """Validate file path."""
        try:
            path = Path(value)
            return not bool(re.search(r'[<>:"|?*]', str(path)))
        except Exception:
            return False
            
    def _sanitize_input(self, value: str, input_type: InputType) -> str:
        """Sanitize input based on type."""
        if input_type == InputType.TEXT:
            # Remove potentially dangerous characters
            value = re.sub(r'[<>]', '', value)
            # Convert special characters to HTML entities
            value = value.replace('&', '&amp;').replace('"', '&quot;')
            
        elif input_type == InputType.USERNAME:
            # Remove anything that's not alphanumeric, underscore, or hyphen
            value = re.sub(r'[^a-zA-Z0-9_-]', '', value)
            
        elif input_type == InputType.EMAIL:
            # Convert to lowercase and remove whitespace
            value = value.lower().strip()
            
        elif input_type == InputType.URL:
            # Remove whitespace and invalid characters
            value = re.sub(r'\s+', '', value)
            value = re.sub(r'[<>"\']', '', value)
            
        elif input_type == InputType.PATH:
            # Remove potentially dangerous characters
            value = re.sub(r'[<>:"|?*]', '', value)
            
        return value

    def _get_file_mime_type(self, file_path: Path) -> str:
        """Get MIME type of a file."""
        try:
            mime_type = magic.from_file(str(file_path), mime=True)
            return mime_type
        except Exception as e:
            self.logger.error(f"Error getting MIME type: {str(e)}")
            return mimetypes.guess_type(str(file_path))[0] or ''

    def _calculate_file_hash(self, file_path: Path, algorithm: str = 'sha256') -> str:
        """Calculate file hash using specified algorithm."""
        hash_obj = hashlib.new(algorithm)
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b''):
                hash_obj.update(chunk)
        return hash_obj.hexdigest()

    def _validate_numeric(self, value: Any) -> bool:
        """Validate numeric value."""
        try:
            float(value)
            return True
        except (TypeError, ValueError):
            return False

    def _validate_timestamp(self, value: str) -> bool:
        """Validate ISO 8601 timestamp."""
        try:
            datetime.fromisoformat(value.replace('Z', '+00:00'))
            return True
        except (ValueError, TypeError):
            return False

    def validate_dict(
        self,
        data: Dict[str, Any],
        schema: Dict[str, Union[InputType, ValidationRule]]
    ) -> Dict[str, Any]:
        """Validate dictionary of values against schema."""
        validated = {}
        errors = {}
        
        for field, rule in schema.items():
            try:
                if isinstance(rule, InputType):
                    # Use default rule for type
                    validated[field] = self.validate(
                        data.get(field),
                        rule
                    )
                else:
                    # Use custom rule
                    validated[field] = self.validate(
                        data.get(field),
                        rule.input_type,
                        rule
                    )
            except ValidationError as e:
                errors[field] = str(e)
                
        if errors:
            raise ValidationError({
                "message": "Validation failed",
                "errors": errors
            })
            
        return validated
    
    def create_schema(
        self,
        **fields: Dict[str, Dict[str, Any]]
    ) -> Dict[str, ValidationRule]:
        """Create validation schema from field definitions."""
        schema = {}
        
        for field_name, field_config in fields.items():
            input_type = field_config.pop('type')
            if not isinstance(input_type, InputType):
                input_type = InputType(input_type)
                
            # Start with default rule for type
            base_rule = self.default_rules[input_type]
            
            # Create new rule with overrides
            schema[field_name] = ValidationRule(
                input_type=input_type,
                **{
                    **{
                        k: v for k, v in vars(base_rule).items()
                        if k != 'input_type'
                    },
                    **field_config
                }
            )
            
        return schema

def create_validator(config_path: Optional[Path] = None) -> InputValidator:
    """Create input validator instance."""
    config = {}
    if config_path and config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
            
    return InputValidator(config) 