import pytest
from pathlib import Path
from src.processors.secure_video_processor import (
    SecureVideoProcessorWithXSS,
    SecurityLevel,
    CORSConfig
)
from src.security.xss.xss_protection import XSSProtectionMode

@pytest.fixture
def cors_config():
    return CORSConfig(allowed_origins=['https://example.com'])

@pytest.fixture
def secure_processor(cors_config):
    return SecureVideoProcessorWithXSS(
        cors_config=cors_config,
        security_level=SecurityLevel.HIGH
    )

@pytest.mark.asyncio
async def test_process_video_with_xss_protection(secure_processor):
    """Test video processing with XSS protection"""
    # Test with clean input
    result = await secure_processor.process_video(
        video_path="test.mp4",
        metadata={
            "title": "Test Video",
            "description": "A test video"
        },
        request_origin="https://example.com"
    )
    assert result['success']
    assert 'X-XSS-Protection' in result['headers']
    assert result['headers']['X-XSS-Protection'] == '1; mode=block'
    
    # Test with malicious input
    result = await secure_processor.process_video(
        video_path="test.mp4<script>alert('xss')</script>",
        metadata={
            "title": "<script>alert('xss')</script>",
            "description": "javascript:alert('xss')"
        },
        request_origin="javascript:alert('xss')"
    )
    assert not result['success']
    assert 'Invalid request content detected' in result['error']

def test_validate_upload_with_xss_protection(secure_processor):
    """Test file upload validation with XSS protection"""
    # Test with clean input
    assert secure_processor.validate_upload({
        'content_type': 'video/mp4',
        'size': 1024 * 1024,  # 1MB
        'filename': 'test.mp4'
    })
    
    # Test with malicious input
    assert not secure_processor.validate_upload({
        'content_type': 'video/mp4<script>alert("xss")</script>',
        'size': 1024 * 1024,
        'filename': 'test.mp4<script>alert("xss")</script>'
    })

def test_security_modes(cors_config):
    """Test different security modes"""
    # Test HIGH security
    high_security = SecureVideoProcessorWithXSS(
        cors_config=cors_config,
        security_level=SecurityLevel.HIGH
    )
    assert high_security.xss_protector.mode == XSSProtectionMode.STRICT
    
    # Test MEDIUM security
    medium_security = SecureVideoProcessorWithXSS(
        cors_config=cors_config,
        security_level=SecurityLevel.MEDIUM
    )
    assert medium_security.xss_protector.mode == XSSProtectionMode.SANITIZE
    
    # Test LOW security
    low_security = SecureVideoProcessorWithXSS(
        cors_config=cors_config,
        security_level=SecurityLevel.LOW
    )
    assert low_security.xss_protector.mode == XSSProtectionMode.ESCAPE
    
    # Test custom XSS mode
    custom_security = SecureVideoProcessorWithXSS(
        cors_config=cors_config,
        security_level=SecurityLevel.HIGH,
        xss_mode=XSSProtectionMode.SANITIZE
    )
    assert custom_security.xss_protector.mode == XSSProtectionMode.SANITIZE

def test_response_headers(secure_processor):
    """Test response headers with XSS protection"""
    headers = secure_processor.get_response_headers("https://example.com")
    
    # Check XSS protection headers
    assert headers['X-XSS-Protection'] == '1; mode=block'
    assert headers['X-Content-Type-Options'] == 'nosniff'
    
    # Check CSP headers
    assert 'Content-Security-Policy' in headers
    csp = headers['Content-Security-Policy']
    assert "'nonce-${NONCE}'" in csp
    assert "require-trusted-types-for 'script'" in csp

@pytest.mark.asyncio
async def test_metadata_sanitization(secure_processor):
    """Test metadata sanitization"""
    metadata = {
        "title": "Test Video <script>alert('xss')</script>",
        "description": "Normal description",
        "tags": ["<script>alert('xss')</script>", "normal"],
        "nested": {
            "field": "<img src=x onerror=alert('xss')>"
        }
    }
    
    result = await secure_processor.process_video(
        video_path="test.mp4",
        metadata=metadata,
        request_origin="https://example.com"
    )
    
    assert result['success']
    assert '<script>' not in str(result)
    assert 'alert' not in str(result)
    assert 'Normal description' in str(result) 