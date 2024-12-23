from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from enum import Enum

class CORSMode(Enum):
    """CORS operation mode"""
    PERMISSIVE = "permissive"  # Development mode - allows all origins
    SELECTIVE = "selective"    # Production mode - allows specific origins
    STRICT = "strict"         # Strict mode - minimal allowed origins

@dataclass
class CORSConfig:
    """CORS configuration settings"""
    mode: CORSMode = CORSMode.STRICT
    allowed_origins: List[str] = field(default_factory=list)
    allowed_methods: List[str] = field(default_factory=lambda: ["GET", "POST"])
    allowed_headers: List[str] = field(default_factory=list)
    expose_headers: List[str] = field(default_factory=list)
    max_age: int = 3600
    allow_credentials: bool = False
    
    def __post_init__(self):
        """Initialize default values based on mode"""
        if self.mode == CORSMode.PERMISSIVE:
            self.allowed_origins = ["*"]
            self.allowed_methods = ["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"]
            self.allowed_headers = ["*"]
            self.expose_headers = ["*"]
            self.allow_credentials = True
        elif self.mode == CORSMode.SELECTIVE:
            if not self.allowed_origins:
                raise ValueError("Selective mode requires explicit allowed_origins")
            if not self.allowed_headers:
                self.allowed_headers = [
                    "Content-Type",
                    "Authorization",
                    "X-Requested-With"
                ]
    
    def get_cors_headers(self, origin: Optional[str] = None) -> Dict[str, str]:
        """Get CORS headers based on configuration"""
        headers = {}
        
        # Handle Access-Control-Allow-Origin
        if origin and self.allowed_origins != ["*"]:
            if origin in self.allowed_origins:
                headers["Access-Control-Allow-Origin"] = origin
        else:
            headers["Access-Control-Allow-Origin"] = self.allowed_origins[0]
            
        # Add other CORS headers
        if self.allowed_methods:
            headers["Access-Control-Allow-Methods"] = ", ".join(self.allowed_methods)
            
        if self.allowed_headers:
            headers["Access-Control-Allow-Headers"] = ", ".join(self.allowed_headers)
            
        if self.expose_headers:
            headers["Access-Control-Expose-Headers"] = ", ".join(self.expose_headers)
            
        if self.max_age:
            headers["Access-Control-Max-Age"] = str(self.max_age)
            
        if self.allow_credentials:
            headers["Access-Control-Allow-Credentials"] = "true"
            
        return headers
    
    def is_origin_allowed(self, origin: str) -> bool:
        """Check if an origin is allowed"""
        if "*" in self.allowed_origins:
            return True
        return origin in self.allowed_origins
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary"""
        return {
            "mode": self.mode.value,
            "allowed_origins": self.allowed_origins,
            "allowed_methods": self.allowed_methods,
            "allowed_headers": self.allowed_headers,
            "expose_headers": self.expose_headers,
            "max_age": self.max_age,
            "allow_credentials": self.allow_credentials
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CORSConfig':
        """Create config from dictionary"""
        mode = CORSMode(data.get("mode", CORSMode.STRICT.value))
        return cls(
            mode=mode,
            allowed_origins=data.get("allowed_origins", []),
            allowed_methods=data.get("allowed_methods", ["GET", "POST"]),
            allowed_headers=data.get("allowed_headers", []),
            expose_headers=data.get("expose_headers", []),
            max_age=data.get("max_age", 3600),
            allow_credentials=data.get("allow_credentials", False)
        )