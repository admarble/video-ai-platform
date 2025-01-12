from dataclasses import dataclass
from typing import Dict, List, Optional, Any
from enum import Enum
import json
from pathlib import Path
import logging
from cors_config import CORSConfig

class SecurityLevel(Enum):
    """Security levels for header configuration"""
    LOW = "low"        # Basic security for development
    MEDIUM = "medium"  # Standard security for staging
    HIGH = "high"     # Maximum security for production

@dataclass
class MediaPolicy:
    """Media-specific security policy configuration"""
    allowed_sources: List[str]
    allowed_codecs: List[str]
    max_file_size: str
    require_encryption: bool
    allow_downloads: bool

@dataclass
class CSPDirective:
    """Content Security Policy directive configuration"""
    default_src: List[str] = None
    script_src: List[str] = None
    style_src: List[str] = None
    img_src: List[str] = None
    media_src: List[str] = None
    worker_src: List[str] = None
    child_src: List[str] = None
    connect_src: List[str] = None
    font_src: List[str] = None
    frame_src: List[str] = None
    frame_ancestors: List[str] = None
    object_src: List[str] = None
    report_uri: Optional[str] = None
    report_only: bool = False
    require_trusted_types_for: List[str] = None
    trusted_types: List[str] = None

class SecurityHeaders:
    """Manages security headers configuration with video processing support"""
    
    def __init__(
        self,
        security_level: SecurityLevel = SecurityLevel.HIGH,
        config_path: Optional[Path] = None,
        cors_config: Optional['CORSConfig'] = None
    ):
        self.security_level = security_level
        self.cors_config = cors_config
        self.logger = logging.getLogger(__name__)
        self.csp_config = CSPDirective()
        self.media_policy = self._get_default_media_policy()
        self._permissions_policy = ""
        
        # Load configuration or set defaults
        if config_path:
            self._load_config(config_path)
        else:
            self._setup_default_headers()
            
    def _get_default_media_policy(self) -> MediaPolicy:
        """Get default media policy based on security level"""
        if self.security_level == SecurityLevel.HIGH:
            return MediaPolicy(
                allowed_sources=["'self'", "blob:"],
                allowed_codecs=["avc1", "mp4a", "vp8", "vp9", "opus"],
                max_file_size="1GB",
                require_encryption=True,
                allow_downloads=False
            )
        elif self.security_level == SecurityLevel.MEDIUM:
            return MediaPolicy(
                allowed_sources=["'self'", "blob:", "data:", "https:"],
                allowed_codecs=["*"],
                max_file_size="2GB",
                require_encryption=False,
                allow_downloads=True
            )
        else:
            return MediaPolicy(
                allowed_sources=["*"],
                allowed_codecs=["*"],
                max_file_size="unlimited",
                require_encryption=False,
                allow_downloads=True
            )
            
    def _setup_default_headers(self) -> None:
        """Setup default security headers with video processing rules"""
        if self.security_level == SecurityLevel.HIGH:
            self.csp_config = CSPDirective(
                default_src=["'self'"],
                script_src=[
                    "'self'", 
                    "'unsafe-inline'", 
                    "https://cdnjs.cloudflare.com"
                ],
                style_src=["'self'", "'unsafe-inline'"],
                img_src=["'self'", "data:", "blob:"],
                media_src=[
                    "'self'",
                    "blob:",
                    "https://cdnjs.cloudflare.com"
                ],
                worker_src=["'self'", "blob:"],
                child_src=["'self'", "blob:"],
                connect_src=["'self'"],
                font_src=["'self'", "https://cdnjs.cloudflare.com"],
                frame_ancestors=["'none'"],
                object_src=["'none'"],
                report_uri="/api/csp-report",
                require_trusted_types_for=["'script'"],
                trusted_types=["'allow-duplicates'"]
            )
        elif self.security_level == SecurityLevel.MEDIUM:
            self.csp_config = CSPDirective(
                default_src=["'self'"],
                script_src=[
                    "'self'",
                    "'unsafe-inline'",
                    "'unsafe-eval'",
                    "https:"
                ],
                style_src=["'self'", "'unsafe-inline'"],
                img_src=["'self'", "data:", "blob:", "*"],
                media_src=["'self'", "blob:", "data:", "https:"],
                worker_src=["'self'", "blob:"],
                connect_src=["'self'", "*"],
                frame_ancestors=["'self'"]
            )
        else:  # LOW
            self.csp_config = CSPDirective(
                default_src=["'self'", "*"],
                script_src=["'self'", "'unsafe-inline'", "'unsafe-eval'", "*"],
                style_src=["'self'", "'unsafe-inline'", "*"],
                img_src=["'self'", "data:", "blob:", "*"],
                media_src=["*"],
                connect_src=["'self'", "*"]
            )

    def _load_config(self, config_path: Path) -> None:
        """Load configuration from a JSON file"""
        try:
            with open(config_path) as f:
                config = json.load(f)
                
            # Load CSP configuration
            if 'csp' in config:
                self.csp_config = CSPDirective(**config['csp'])
                
            # Load media policy
            if 'media_policy' in config:
                self.media_policy = MediaPolicy(**config['media_policy'])
                
            self.logger.info(f"Loaded configuration from {config_path}")
        except Exception as e:
            self.logger.error(f"Failed to load configuration: {e}")
            self._setup_default_headers()
            
    def _build_csp_header(self) -> str:
        """Build Content Security Policy header value"""
        directives = []
        
        # Helper function to format directive
        def format_directive(name: str, values: List[str]) -> str:
            return f"{name} {' '.join(values)}"
            
        # Add directives if they have values
        if self.csp_config.default_src:
            directives.append(format_directive("default-src", self.csp_config.default_src))
        if self.csp_config.script_src:
            directives.append(format_directive("script-src", self.csp_config.script_src))
        if self.csp_config.style_src:
            directives.append(format_directive("style-src", self.csp_config.style_src))
        if self.csp_config.img_src:
            directives.append(format_directive("img-src", self.csp_config.img_src))
        if self.csp_config.media_src:
            directives.append(format_directive("media-src", self.csp_config.media_src))
        if self.csp_config.worker_src:
            directives.append(format_directive("worker-src", self.csp_config.worker_src))
        if self.csp_config.child_src:
            directives.append(format_directive("child-src", self.csp_config.child_src))
        if self.csp_config.connect_src:
            directives.append(format_directive("connect-src", self.csp_config.connect_src))
        if self.csp_config.font_src:
            directives.append(format_directive("font-src", self.csp_config.font_src))
        if self.csp_config.frame_src:
            directives.append(format_directive("frame-src", self.csp_config.frame_src))
        if self.csp_config.frame_ancestors:
            directives.append(format_directive("frame-ancestors", self.csp_config.frame_ancestors))
        if self.csp_config.object_src:
            directives.append(format_directive("object-src", self.csp_config.object_src))
        if self.csp_config.report_uri:
            directives.append(f"report-uri {self.csp_config.report_uri}")
        if self.csp_config.require_trusted_types_for:
            directives.append(format_directive("require-trusted-types-for", self.csp_config.require_trusted_types_for))
        if self.csp_config.trusted_types:
            directives.append(format_directive("trusted-types", self.csp_config.trusted_types))
            
        return "; ".join(directives)
            
    def get_security_headers(self, include_cors: bool = True) -> Dict[str, str]:
        """Get all configured security headers with optional CORS integration"""
        headers = {
            'X-Content-Type-Options': 'nosniff',
            'X-Frame-Options': 'DENY' if self.security_level == SecurityLevel.HIGH else 'SAMEORIGIN',
            'X-XSS-Protection': '1; mode=block',
            'Referrer-Policy': 'strict-origin-when-cross-origin',
            'Permissions-Policy': self._get_permissions_policy(),
            'Cross-Origin-Embedder-Policy': 'require-corp',
            'Cross-Origin-Opener-Policy': 'same-origin',
            'Cross-Origin-Resource-Policy': 'same-origin',
            'Feature-Policy': self._get_feature_policy()
        }
        
        # Add HSTS for high security
        if self.security_level == SecurityLevel.HIGH:
            headers['Strict-Transport-Security'] = 'max-age=31536000; includeSubDomains; preload'
            
        # Add Content Security Policy
        csp_header = self._build_csp_header()
        if csp_header:
            header_name = 'Content-Security-Policy-Report-Only' if self.csp_config.report_only else 'Content-Security-Policy'
            headers[header_name] = csp_header
            
        # Add CORS headers if requested and configured
        if include_cors and self.cors_config:
            headers.update(self._get_cors_headers())
            
        return headers
        
    def _get_permissions_policy(self) -> str:
        """Get Permissions-Policy header value"""
        return self._permissions_policy if self._permissions_policy else "*"
        
    def _get_feature_policy(self) -> str:
        """Get Feature-Policy header value for media features"""
        if self.security_level == SecurityLevel.HIGH:
            return (
                "autoplay 'self'; "
                "camera 'none'; "
                "microphone 'none'; "
                "fullscreen 'self'; "
                "picture-in-picture 'self'"
            )
        elif self.security_level == SecurityLevel.MEDIUM:
            return (
                "autoplay 'self' https:; "
                "camera 'self'; "
                "microphone 'self'; "
                "fullscreen *; "
                "picture-in-picture *"
            )
        return "*"

    def _get_cors_headers(self) -> Dict[str, str]:
        """Get CORS headers from CORS configuration"""
        if not self.cors_config:
            return {}
            
        rule = self.cors_config.get_rule("default")
        if not rule:
            return {}
            
        return {
            'Access-Control-Allow-Origin': ', '.join(rule.allowed_origins),
            'Access-Control-Allow-Methods': ', '.join(rule.allowed_methods),
            'Access-Control-Allow-Headers': ', '.join(rule.allowed_headers),
            'Access-Control-Max-Age': str(rule.max_age)
        }
        
    def update_for_video_processing(
        self,
        worker_enabled: bool = True,
        media_download: bool = False
    ) -> None:
        """Update security configuration for video processing"""
        # Update media sources
        media_src = ["'self'", "blob:"]
        if self.security_level != SecurityLevel.HIGH:
            media_src.extend(["data:", "https:"])
        
        # Update worker sources for video processing
        worker_src = ["'self'"]
        if worker_enabled:
            worker_src.append("blob:")
            
        # Update CSP for video processing
        self.csp_config.media_src = media_src
        self.csp_config.worker_src = worker_src
        
        # Update media policy
        self.media_policy.allow_downloads = media_download
        
        # Update permissions policy for video features
        if self.security_level != SecurityLevel.LOW:
            self._update_media_permissions()
            
    def _update_media_permissions(self) -> None:
        """Update permissions policy for media features"""
        permissions = []
        
        if self.security_level == SecurityLevel.HIGH:
            permissions.extend([
                "autoplay=(self)",
                "fullscreen=(self)",
                "picture-in-picture=(self)"
            ])
        else:
            permissions.extend([
                "autoplay=(self https:)",
                "fullscreen=*",
                "picture-in-picture=*"
            ])
            
        self._permissions_policy = '; '.join(permissions)

# Helper function to create security headers instance
def create_security_headers(
    security_level: SecurityLevel = SecurityLevel.HIGH,
    config_path: Optional[Path] = None,
    cors_config: Optional['CORSConfig'] = None
) -> SecurityHeaders:
    """Create security headers configuration"""
    return SecurityHeaders(
        security_level=security_level,
        config_path=config_path,
        cors_config=cors_config
    ) 