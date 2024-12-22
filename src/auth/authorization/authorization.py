from typing import Dict, List, Optional, Set, Any, Union
from enum import Enum, auto
from dataclasses import dataclass
import logging
from pathlib import Path
import json
import re

from ..authentication.auth_manager import AuthenticationManager

class ResourceType(Enum):
    """Types of resources that can be protected"""
    VIDEO = auto()
    MODEL = auto()
    CONFIGURATION = auto()
    METRICS = auto()
    USER = auto()
    SYSTEM = auto()

class Action(Enum):
    """Possible actions on resources"""
    CREATE = "create"
    READ = "read"
    UPDATE = "update"
    DELETE = "delete"
    EXECUTE = "execute"
    LIST = "list"
    MANAGE = "manage"

@dataclass
class Permission:
    """Represents a permission to perform an action on a resource"""
    resource_type: ResourceType
    action: Action
    resource_id: Optional[str] = None  # None means all resources of this type

class Policy:
    """Represents an access control policy"""
    
    def __init__(
        self,
        name: str,
        permissions: List[Permission],
        description: Optional[str] = None
    ):
        self.name = name
        self.permissions = permissions
        self.description = description

    def to_dict(self) -> Dict[str, Any]:
        """Convert policy to dictionary"""
        return {
            "name": self.name,
            "description": self.description,
            "permissions": [
                {
                    "resource_type": p.resource_type.name,
                    "action": p.action.value,
                    "resource_id": p.resource_id
                }
                for p in self.permissions
            ]
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Policy':
        """Create policy from dictionary"""
        return cls(
            name=data["name"],
            description=data.get("description"),
            permissions=[
                Permission(
                    resource_type=ResourceType[p["resource_type"]],
                    action=Action(p["action"]),
                    resource_id=p.get("resource_id")
                )
                for p in data["permissions"]
            ]
        ) 