"""Validation rule definitions."""

from dataclasses import dataclass
from typing import Any, List, Optional, Callable

from .types import InputType

@dataclass
class ValidationRule:
    """Defines validation rules for an input."""
    input_type: InputType
    required: bool = True
    min_length: Optional[int] = None
    max_length: Optional[int] = None
    pattern: Optional[str] = None
    allowed_values: Optional[List[Any]] = None
    custom_validator: Optional[Callable[[Any], bool]] = None
    allowed_mime_types: Optional[List[str]] = None
    max_size_bytes: Optional[int] = None
    sanitize: bool = True 