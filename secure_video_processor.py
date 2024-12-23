from typing import Dict, Any, Optional, Tuple
from pathlib import Path
import re
from security_headers import SecurityHeaders, SecurityLevel, create_security_headers
from cors_config import CORSConfig
from xss_protection import XSSProtector, XSSProtectionMode

def setup_security(
    cors_config: 'CORSConfig',
    security_level: SecurityLevel = SecurityLevel.HIGH,
    config_path: Optional[Path] = None
) -> Tuple[SecurityHeaders, Dict[str, Any]]:
    """Setup security configuration with CORS integration"""
    
    # Initialize security headers with CORS config
    security_headers = create_security_headers(
        security_level=security_level,
        config_path=config_path,
        cors_config=cors_config
    )
    
    # Update security configuration for video processing
    security_headers.update_for_video_processing(
        worker_enabled=True,  # Enable Web Workers for video processing
        media_download=security_level != SecurityLevel.HIGH  # Allow downloads except in HIGH security
    )
    
    # Create configuration object for the video processing system
    security_config = {
        'security_level': security_level.value,
        'headers': security_headers.get_security_headers(include_cors=True),
        'media_policy': security_headers.media_policy,
        'cors_enabled': True,
        'allowed_upload_types': [
            'video/mp4',
            'video/webm',
            'video/quicktime',
            'video/x-msvideo'
        ] if security_level == SecurityLevel.HIGH else ['video/*'],
        'max_upload_size': '1GB' if security_level == SecurityLevel.HIGH else '2GB'
    }
    
    return security_headers, security_config

class SecureVideoProcessor:
    """Video processor with integrated security"""
    
    def __init__(
        self,
        cors_config: 'CORSConfig',
        security_level: SecurityLevel = SecurityLevel.HIGH
    ):
        # Setup security
        self.security_headers, self.security_config = setup_security(
            cors_config,
            security_level
        )
        
        # Initialize other components
        self.service_manager = ServiceManager(self.security_config)
        
    def get_response_headers(self, request_origin: Optional[str] = None) -> Dict[str, str]:
        """Get response headers for video processing endpoints"""
        headers = self.security_headers.get_security_headers(include_cors=True)
        
        # Add CORS headers if origin is provided
        if request_origin:
            cors_headers = self.security_headers.cors_config.get_cors_headers(
                origin=request_origin,
                request_method='POST',
                request_headers=['Content-Type', 'Authorization']
            )
            headers.update(cors_headers)
            
        return headers
        
    async def process_video(self, video_path: str, request_origin: Optional[str] = None) -> Dict[str, Any]:
        """Process video with security headers"""
        try:
            # Get response headers
            headers = self.get_response_headers(request_origin)
            
            # Process video
            frames, fps = self.service_manager.extract_frames(video_path)
            
            # Add security headers to response
            return {
                'success': True,
                'frames': len(frames),
                'fps': fps,
                'headers': headers
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'headers': headers
            }
            
    def validate_upload(self, file_info: Dict[str, Any]) -> bool:
        """Validate video upload against security policy"""
        # Check file type
        if self.security_config['security_level'] == SecurityLevel.HIGH.value:
            if file_info['content_type'] not in self.security_config['allowed_upload_types']:
                return False
                
        # Check file size
        max_size = parse_size(self.security_config['max_upload_size'])
        if file_info['size'] > max_size:
            return False
            
        return True

def parse_size(size_str: str) -> int:
    """Parse size string (e.g., '1GB') to bytes"""
    units = {
        'B': 1,
        'KB': 1024,
        'MB': 1024 * 1024,
        'GB': 1024 * 1024 * 1024
    }
    
    size = size_str.strip().upper()
    if not size:
        return 0
        
    # Extract number and unit
    match = re.match(r'^(\d+(?:\.\d+)?)\s*([A-Z]+)$', size)
    if not match:
        raise ValueError(f"Invalid size format: {size_str}")
        
    number = float(match.group(1))
    unit = match.group(2)
    
    if unit not in units:
        raise ValueError(f"Invalid size unit: {unit}")
        
    return int(number * units[unit])

class ServiceManager:
    """Service manager for video processing"""
    
    def __init__(self, security_config: Dict[str, Any]):
        self.security_config = security_config
        
    def extract_frames(self, video_path: str) -> Tuple[list, float]:
        """Extract frames from video file"""
        # This is a placeholder for actual video processing logic
        # In a real implementation, this would use a video processing library
        frames = []
        fps = 30.0
        return frames, fps 

class SecureVideoProcessorWithCSP(SecureVideoProcessor):
    """Video processor with integrated security and CSP reporting"""
    
    def __init__(
        self,
        cors_config: 'CORSConfig',
        security_level: SecurityLevel = SecurityLevel.HIGH,
        config_path: Optional[Path] = None
    ):
        # Initialize parent class
        super().__init__(cors_config, security_level)
        
        # Setup CSP reporting
        self._setup_csp_reporting(config_path)
        
    def _setup_csp_reporting(self, config_path: Optional[Path] = None) -> None:
        """Configure CSP reporting"""
        if config_path:
            self.security_headers.setup_csp_reporting(config_path)
        else:
            # Default CSP reporting configuration
            self.security_headers.csp_config.report_uri = "/csp-report"
            self.security_headers.csp_config.report_to = "csp-endpoint"
            
    async def process_video(
        self,
        video_path: str,
        metadata: Dict[str, Any],
        request_origin: Optional[str] = None
    ) -> Dict[str, Any]:
        """Process video with CSP headers"""
        try:
            # Get response headers with CSP
            headers = self.get_response_headers(request_origin)
            
            # Add CSP nonce for any inline scripts
            nonce = self.security_headers.generate_nonce()
            headers['Content-Security-Policy'] = self.security_headers.get_csp_header(nonce)
            
            # Process video
            result = await super().process_video(video_path, request_origin)
            result['headers'].update(headers)
            result['nonce'] = nonce
            
            return result
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'headers': headers if 'headers' in locals() else {}
            }

class SecureVideoProcessorWithXSS(SecureVideoProcessorWithCSP):
    """Video processor with integrated security, CSP reporting, and XSS protection"""
    
    def __init__(
        self,
        cors_config: 'CORSConfig',
        security_level: SecurityLevel = SecurityLevel.HIGH,
        config_path: Optional[Path] = None,
        xss_mode: Optional[XSSProtectionMode] = None
    ):
        # Initialize parent class
        super().__init__(cors_config, security_level, config_path)
        
        # Initialize XSS protection
        if xss_mode is None:
            xss_mode = {
                SecurityLevel.HIGH: XSSProtectionMode.STRICT,
                SecurityLevel.MEDIUM: XSSProtectionMode.SANITIZE,
                SecurityLevel.LOW: XSSProtectionMode.ESCAPE
            }[security_level]
            
        self.xss_protector = XSSProtector(mode=xss_mode)
        
        # Update security headers for XSS
        self._setup_xss_headers()
        
    def _setup_xss_headers(self) -> None:
        """Configure additional headers for XSS protection"""
        # Enable browser's XSS filter in block mode
        self.security_headers.headers['X-XSS-Protection'] = '1; mode=block'
        
        # Prevent MIME type sniffing
        self.security_headers.headers['X-Content-Type-Options'] = 'nosniff'
        
        # Update CSP for better XSS protection
        if self.security_level == SecurityLevel.HIGH:
            self.security_headers.csp_config.script_src.append("'nonce-${NONCE}'")
            self.security_headers.csp_config.require_trusted_types_for = ["'script'"]
            
    async def process_video(
        self,
        video_path: str,
        metadata: Dict[str, Any],
        request_origin: Optional[str] = None
    ) -> Dict[str, Any]:
        """Process video with XSS protection"""
        try:
            # Validate request for XSS attempts
            headers = getattr(metadata, 'headers', {})
            if not self.xss_protector.validate_request(
                headers=headers,
                body=metadata,
                query_params={}
            ):
                return {
                    'success': False,
                    'error': 'Invalid request content detected',
                    'headers': self.get_response_headers(request_origin)
                }
            
            # Sanitize metadata
            safe_metadata = self.xss_protector.sanitize_input(metadata)
            
            # Sanitize video path
            safe_video_path = self.xss_protector.sanitize_input(video_path, context='text')
            
            # Process video with sanitized inputs
            result = await super().process_video(safe_video_path, safe_metadata, request_origin)
            
            # Sanitize response data
            if result['success']:
                result['frames'] = self.xss_protector.sanitize_input(result['frames'], context='text')
                result['fps'] = float(self.xss_protector.sanitize_input(str(result['fps']), context='text'))
                
            return result
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'headers': self.get_response_headers(request_origin)
            }
            
    def validate_upload(self, file_info: Dict[str, Any]) -> bool:
        """Validate video upload with XSS protection"""
        try:
            # Sanitize file info
            safe_file_info = self.xss_protector.sanitize_input(file_info)
            
            # Validate with parent class
            return super().validate_upload(safe_file_info)
            
        except Exception:
            return False
            
    def get_response_headers(self, request_origin: Optional[str] = None) -> Dict[str, str]:
        """Get response headers with XSS protection"""
        # Get base headers
        headers = super().get_response_headers(request_origin)
        
        # Sanitize origin if provided
        if request_origin:
            request_origin = self.xss_protector.sanitize_input(request_origin, context='url')
            
        # Add or update XSS protection headers
        headers['X-XSS-Protection'] = '1; mode=block'
        headers['X-Content-Type-Options'] = 'nosniff'
        
        return headers