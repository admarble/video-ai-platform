from dataclasses import dataclass
from typing import Dict, List, Optional, Any
from enum import Enum
import json
from pathlib import Path
import logging

class SecurityLevel(Enum):
    """Security levels for header configuration"""
    LOW = "low"        # Basic security for development
    MEDIUM = "medium"  # Standard security for staging
    HIGH = "high"     # Maximum security for production

@dataclass
class CSPDirective:
    """Content Security Policy directive configuration"""
    default_src: List[str] = None
    script_src: List[str] = None
    style_src: List[str] = None
    img_src: List[str] = None
    connect_src: List[str] = None
    font_src: List[str] = None
    media_src: List[str] = None
    frame_src: List[str] = None
    frame_ancestors: List[str] = None
    report_uri: Optional[str] = None
    report_only: bool = False

class SecurityHeaders:
    """Manages security headers configuration"""
    
    def __init__(
        self,
        security_level: SecurityLevel = SecurityLevel.HIGH,
        config_path: Optional[Path] = None
    ):
        self.security_level = security_level
        self.logger = logging.getLogger(__name__)
        self.csp_config = CSPDirective()
        
        # Load configuration or set defaults
        if config_path:
            self._load_config(config_path)
        else:
            self._setup_default_headers()
            
    def _load_config(self, config_path: Path) -> None:
        """Load security headers configuration from file"""
        try:
            with open(config_path) as f:
                config = json.load(f)
            
            self.security_level = SecurityLevel(config.get('security_level', 'high'))
            
            # Load CSP configuration
            if 'csp' in config:
                self.csp_config = CSPDirective(**config['csp'])
                
        except Exception as e:
            self.logger.error(f"Error loading security config: {str(e)}")
            self._setup_default_headers()
            
    def _setup_default_headers(self) -> None:
        """Setup default security headers based on security level"""
        if self.security_level == SecurityLevel.HIGH:
            self.csp_config = CSPDirective(
                default_src=["'self'"],
                script_src=["'self'", "'unsafe-inline'", "https://cdnjs.cloudflare.com"],
                style_src=["'self'", "'unsafe-inline'"],
                img_src=["'self'", "data:", "blob:"],
                connect_src=["'self'"],
                font_src=["'self'", "https://cdnjs.cloudflare.com"],
                frame_ancestors=["'none'"],
                report_uri="/api/csp-report"
            )
        elif self.security_level == SecurityLevel.MEDIUM:
            self.csp_config = CSPDirective(
                default_src=["'self'"],
                script_src=["'self'", "'unsafe-inline'", "'unsafe-eval'"],
                style_src=["'self'", "'unsafe-inline'"],
                img_src=["'self'", "data:", "blob:", "*"],
                connect_src=["'self'", "*"],
                frame_ancestors=["'self'"]
            )
        else:  # LOW
            self.csp_config = CSPDirective(
                default_src=["'self'", "*"],
                script_src=["'self'", "'unsafe-inline'", "'unsafe-eval'", "*"],
                style_src=["'self'", "'unsafe-inline'", "*"],
                img_src=["'self'", "data:", "blob:", "*"],
                connect_src=["'self'", "*"]
            )
            
    def get_security_headers(self) -> Dict[str, str]:
        """Get all configured security headers"""
        headers = {
            'X-Content-Type-Options': 'nosniff',
            'X-Frame-Options': 'DENY' if self.security_level == SecurityLevel.HIGH else 'SAMEORIGIN',
            'X-XSS-Protection': '1; mode=block',
            'Referrer-Policy': 'strict-origin-when-cross-origin',
            'Permissions-Policy': self._get_permissions_policy(),
            'Cross-Origin-Embedder-Policy': 'require-corp',
            'Cross-Origin-Opener-Policy': 'same-origin',
            'Cross-Origin-Resource-Policy': 'same-origin'
        }
        
        # Add HSTS for high security
        if self.security_level == SecurityLevel.HIGH:
            headers['Strict-Transport-Security'] = 'max-age=31536000; includeSubDomains; preload'
            
        # Add Content Security Policy
        csp_header = self._build_csp_header()
        if csp_header:
            header_name = 'Content-Security-Policy-Report-Only' if self.csp_config.report_only else 'Content-Security-Policy'
            headers[header_name] = csp_header
            
        return headers
        
    def _build_csp_header(self) -> str:
        """Build Content Security Policy header value"""
        directives = []
        
        if self.csp_config.default_src:
            directives.append(f"default-src {' '.join(self.csp_config.default_src)}")
            
        if self.csp_config.script_src:
            directives.append(f"script-src {' '.join(self.csp_config.script_src)}")
            
        if self.csp_config.style_src:
            directives.append(f"style-src {' '.join(self.csp_config.style_src)}")
            
        if self.csp_config.img_src:
            directives.append(f"img-src {' '.join(self.csp_config.img_src)}")
            
        if self.csp_config.connect_src:
            directives.append(f"connect-src {' '.join(self.csp_config.connect_src)}")
            
        if self.csp_config.font_src:
            directives.append(f"font-src {' '.join(self.csp_config.font_src)}")
            
        if self.csp_config.media_src:
            directives.append(f"media-src {' '.join(self.csp_config.media_src)}")
            
        if self.csp_config.frame_src:
            directives.append(f"frame-src {' '.join(self.csp_config.frame_src)}")
            
        if self.csp_config.frame_ancestors:
            directives.append(f"frame-ancestors {' '.join(self.csp_config.frame_ancestors)}")
            
        if self.csp_config.report_uri:
            directives.append(f"report-uri {self.csp_config.report_uri}")
            
        return '; '.join(directives)
        
    def _get_permissions_policy(self) -> str:
        """Get Permissions-Policy header value"""
        if self.security_level == SecurityLevel.HIGH:
            return (
                "accelerometer=(), camera=(), geolocation=(), gyroscope=(), "
                "magnetometer=(), microphone=(), payment=(), usb=()"
            )
        elif self.security_level == SecurityLevel.MEDIUM:
            return "camera=(), microphone=(), payment=(), usb=()"
        return ""  # No restrictions for LOW security

    def update_csp_directive(self, directive: str, values: List[str]) -> None:
        """Update specific CSP directive"""
        if hasattr(self.csp_config, directive):
            setattr(self.csp_config, directive, values)
            
    def update_security_level(self, level: SecurityLevel) -> None:
        """Update security level and reset headers"""
        self.security_level = level
        self._setup_default_headers()

# Helper function to create security headers instance
def create_security_headers(
    security_level: SecurityLevel = SecurityLevel.HIGH,
    config_path: Optional[Path] = None
) -> SecurityHeaders:
    """Create security headers configuration"""
    return SecurityHeaders(security_level=security_level, config_path=config_path) 