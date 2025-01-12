"""Basic usage examples for the validation framework."""

from typing import Dict, Any
import json
from pathlib import Path
import re

from ..core.types import InputType
from ..core.rules import ValidationRule
from ..core.exceptions import ValidationError
from ..core.validator import create_validator

def example_basic_validation():
    """Example of basic validation usage."""
    validator = create_validator()
    
    try:
        # Validate username
        username = validator.validate("john_doe123", InputType.USERNAME)
        print(f"Valid username: {username}")
        
        # Validate email
        email = validator.validate("user@example.com", InputType.EMAIL)
        print(f"Valid email: {email}")
        
        # Validate password
        password = validator.validate("SecureP@ss123", InputType.PASSWORD)
        print("Valid password")
        
        # Validate URL
        url = validator.validate("https://example.com/video", InputType.URL)
        print(f"Valid URL: {url}")
        
    except ValidationError as e:
        print(f"Validation error: {str(e)}")

def example_file_validation():
    """Example of file validation."""
    validator = create_validator()
    
    try:
        # Validate video file
        video_path = validator.validate(
            "path/to/video.mp4",
            InputType.VIDEO_FILE
        )
        print(f"Valid video file: {video_path}")
        
        # Validate image with custom size limit
        custom_rule = ValidationRule(
            input_type=InputType.IMAGE_FILE,
            max_size_bytes=10 * 1024 * 1024  # 10MB
        )
        image_path = validator.validate(
            "path/to/image.jpg",
            InputType.IMAGE_FILE,
            custom_rule
        )
        print(f"Valid image file: {image_path}")
        
    except ValidationError as e:
        print(f"File validation error: {str(e)}")

def example_schema_validation():
    """Example of schema validation."""
    validator = create_validator()
    
    # Create validation schema
    schema = validator.create_schema(
        username={
            'type': InputType.USERNAME,
            'required': True,
            'min_length': 5
        },
        email={
            'type': InputType.EMAIL,
            'required': True
        },
        profile_image={
            'type': InputType.IMAGE_FILE,
            'required': False,
            'max_size_bytes': 5 * 1024 * 1024  # 5MB
        }
    )
    
    # Data to validate
    data = {
        'username': 'john_doe',
        'email': 'john@example.com',
        'profile_image': 'path/to/profile.jpg'
    }
    
    try:
        validated_data = validator.validate_dict(data, schema)
        print("Validated data:", json.dumps(validated_data, indent=2))
    except ValidationError as e:
        print(f"Schema validation error: {str(e)}")

def example_custom_validation():
    """Example of custom validation rules."""
    validator = create_validator()
    
    # Custom validation function
    def validate_phone(value: str) -> bool:
        return bool(re.match(r'^\+?1?\d{9,15}$', value))
    
    # Create custom rule
    phone_rule = ValidationRule(
        input_type=InputType.TEXT,
        pattern=r'^\+?1?\d{9,15}$',
        custom_validator=validate_phone,
        sanitize=True
    )
    
    try:
        # Validate phone number
        phone = validator.validate("+1234567890", InputType.TEXT, phone_rule)
        print(f"Valid phone number: {phone}")
    except ValidationError as e:
        print(f"Custom validation error: {str(e)}")

if __name__ == "__main__":
    print("\nBasic Validation Example:")
    example_basic_validation()
    
    print("\nFile Validation Example:")
    example_file_validation()
    
    print("\nSchema Validation Example:")
    example_schema_validation()
    
    print("\nCustom Validation Example:")
    example_custom_validation() 