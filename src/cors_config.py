from typing import List, Optional, Dict, Any
from dataclasses import dataclass
import re
from enum import Enum
from pathlib import Path
import json
import logging

class CORSMode(Enum):
    """CORS operation modes"""
    PERMISSIVE = "permissive"  # Development mode - more relaxed
    STRICT = "strict"          # Production mode - more restrictive
    CUSTOM = "custom"          # Custom configuration

@dataclass
class CORSRule:
    """Defines a CORS rule configuration"""
    allowed_origins: List[str]
    allowed_methods: List[str]
    allowed_headers: List[str]
    expose_headers: List[str]
    max_age: int
    allow_credentials: bool
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CORSRule':
        return cls(
            allowed_origins=data.get('allowed_origins', ['*']),
            allowed_methods=data.get('allowed_methods', ['GET', 'POST']),
            allowed_headers=data.get('allowed_headers', ['Content-Type', 'Authorization']),
            expose_headers=data.get('expose_headers', []),
            max_age=data.get('max_age', 86400),
            allow_credentials=data.get('allow_credentials', False)
        )

class CORSConfig:
    """Manages CORS configuration and rule processing"""
    
    def __init__(
        self,
        mode: CORSMode = CORSMode.STRICT,
        config_path: Optional[Path] = None
    ):
        self.mode = mode
        self.logger = logging.getLogger(__name__)
        self.rules: Dict[str, CORSRule] = {}
        
        # Load configuration
        if config_path:
            self._load_config(config_path)
        else:
            self._setup_default_rules()
            
    def _load_config(self, config_path: Path) -> None:
        """Load CORS configuration from file"""
        try:
            with open(config_path) as f:
                config = json.load(f)
                
            self.mode = CORSMode(config.get('mode', 'strict'))
            
            # Load rules
            for name, rule_data in config.get('rules', {}).items():
                self.rules[name] = CORSRule.from_dict(rule_data)
                
        except Exception as e:
            self.logger.error(f"Error loading CORS config: {str(e)}")
            self._setup_default_rules()
            
    def _setup_default_rules(self) -> None:
        """Setup default CORS rules based on mode"""
        if self.mode == CORSMode.PERMISSIVE:
            # Development mode - more permissive
            self.rules['default'] = CORSRule(
                allowed_origins=['*'],
                allowed_methods=['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS'],
                allowed_headers=['*'],
                expose_headers=['Content-Length', 'Content-Range'],
                max_age=86400,
                allow_credentials=True
            )
            
        elif self.mode == CORSMode.STRICT:
            # Production mode - more restrictive
            self.rules['default'] = CORSRule(
                allowed_origins=[],  # Must be explicitly set
                allowed_methods=['GET', 'POST'],
                allowed_headers=['Content-Type', 'Authorization'],
                expose_headers=[],
                max_age=3600,
                allow_credentials=False
            )
            
    def add_rule(self, name: str, rule: CORSRule) -> None:
        """Add a new CORS rule"""
        self.rules[name] = rule
        
    def get_rule(self, name: str) -> Optional[CORSRule]:
        """Get CORS rule by name"""
        return self.rules.get(name)
        
    def is_origin_allowed(self, origin: str, rule_name: str = 'default') -> bool:
        """Check if origin is allowed for given rule"""
        rule = self.rules.get(rule_name)
        if not rule:
            return False
            
        # Handle wildcard
        if '*' in rule.allowed_origins:
            return True
            
        # Check exact matches
        if origin in rule.allowed_origins:
            return True
            
        # Check pattern matches
        return any(
            re.match(pattern.replace('*', '.*'), origin)
            for pattern in rule.allowed_origins
            if '*' in pattern
        )
        
    def get_cors_headers(
        self,
        origin: str,
        request_method: str,
        request_headers: Optional[List[str]] = None,
        rule_name: str = 'default'
    ) -> Dict[str, str]:
        """Get CORS headers for response"""
        rule = self.rules.get(rule_name)
        if not rule or not self.is_origin_allowed(origin, rule_name):
            return {}
            
        headers = {
            'Access-Control-Allow-Origin': origin if origin != '*' else '*',
            'Access-Control-Allow-Methods': ', '.join(rule.allowed_methods),
            'Access-Control-Max-Age': str(rule.max_age)
        }
        
        # Add credentials header if enabled
        if rule.allow_credentials:
            headers['Access-Control-Allow-Credentials'] = 'true'
            
        # Add allowed headers
        if rule.allowed_headers:
            if '*' in rule.allowed_headers:
                # If request headers provided, reflect them
                if request_headers:
                    headers['Access-Control-Allow-Headers'] = ', '.join(request_headers)
                else:
                    headers['Access-Control-Allow-Headers'] = '*'
            else:
                headers['Access-Control-Allow-Headers'] = ', '.join(rule.allowed_headers)
                
        # Add exposed headers
        if rule.expose_headers:
            headers['Access-Control-Expose-Headers'] = ', '.join(rule.expose_headers)
            
        return headers
        
    def handle_preflight(
        self,
        origin: str,
        request_method: str,
        request_headers: Optional[List[str]] = None,
        rule_name: str = 'default'
    ) -> Dict[str, str]:
        """Handle CORS preflight request"""
        rule = self.rules.get(rule_name)
        if not rule:
            return {}
            
        # Check if origin is allowed
        if not self.is_origin_allowed(origin, rule_name):
            return {}
            
        # Check if method is allowed
        if request_method not in rule.allowed_methods:
            return {}
            
        # Check if headers are allowed
        if request_headers:
            if '*' not in rule.allowed_headers:
                if not all(h in rule.allowed_headers for h in request_headers):
                    return {}
                    
        return self.get_cors_headers(
            origin,
            request_method,
            request_headers,
            rule_name
        )

def create_cors_config(
    mode: CORSMode = CORSMode.STRICT,
    config_path: Optional[Path] = None
) -> CORSConfig:
    """Create CORS configuration instance"""
    return CORSConfig(mode=mode, config_path=config_path) 