import pytest
from src.security.xss.xss_protection import XSSProtector, XSSProtectionMode
import json

@pytest.fixture
def strict_protector():
    return XSSProtector(mode=XSSProtectionMode.STRICT)

@pytest.fixture
def sanitize_protector():
    return XSSProtector(mode=XSSProtectionMode.SANITIZE)

@pytest.fixture
def escape_protector():
    return XSSProtector(mode=XSSProtectionMode.ESCAPE)

def test_basic_text_sanitization(strict_protector):
    """Test basic text sanitization"""
    input_text = "Hello <script>alert('xss')</script> World"
    sanitized = strict_protector.sanitize_input(input_text, context='text')
    assert '<script>' not in sanitized
    assert 'alert' not in sanitized
    assert 'Hello  World' in sanitized

def test_html_sanitization(strict_protector):
    """Test HTML sanitization with allowed tags"""
    input_html = '<p>Hello</p><script>alert("xss")</script><strong>World</strong>'
    sanitized = strict_protector.sanitize_input(input_html, context='html')
    assert '<p>Hello</p>' in sanitized
    assert '<strong>World</strong>' in sanitized
    assert '<script>' not in sanitized
    assert 'alert' not in sanitized

def test_url_sanitization(strict_protector):
    """Test URL sanitization"""
    urls = [
        "javascript:alert('xss')",  # Should be rejected
        "data:text/html,<script>alert('xss')</script>",  # Should be rejected
        "https://example.com?q=<script>alert('xss')</script>",  # Should be sanitized
        "https://example.com",  # Should be allowed
    ]
    
    for url in urls:
        sanitized = strict_protector.sanitize_input(url, context='url')
        assert 'javascript:' not in sanitized.lower()
        assert '<script>' not in sanitized
        if 'example.com' in url:
            assert 'example.com' in sanitized

def test_json_sanitization(strict_protector):
    """Test JSON data sanitization"""
    input_data = {
        "name": "<script>alert('xss')</script>",
        "description": "Normal text",
        "nested": {
            "field": "<img src=x onerror=alert('xss')>",
        },
        "list": ["<script>alert('xss')</script>", "normal"]
    }
    
    sanitized = strict_protector.sanitize_input(input_data)
    assert isinstance(sanitized, dict)
    assert '<script>' not in json.dumps(sanitized)
    assert 'alert' not in json.dumps(sanitized)
    assert 'Normal text' in sanitized['description']

def test_request_validation(strict_protector):
    """Test complete request validation"""
    headers = {
        "User-Agent": "Mozilla/5.0",
        "X-Custom": "<script>alert('xss')</script>"
    }
    body = {
        "message": "Hello <script>alert('xss')</script>",
    }
    query_params = {
        "q": "<script>alert('xss')</script>"
    }
    
    # Should fail due to XSS content
    assert not strict_protector.validate_request(headers, body, query_params)
    
    # Should pass with clean content
    clean_request = {
        "headers": {"User-Agent": "Mozilla/5.0"},
        "body": {"message": "Hello World"},
        "query_params": {"q": "search term"}
    }
    assert strict_protector.validate_request(
        clean_request["headers"],
        clean_request["body"],
        clean_request["query_params"]
    )

def test_sanitize_mode_html(sanitize_protector):
    """Test SANITIZE mode with more permissive HTML"""
    input_html = '''
        <div class="container">
            <p>Hello</p>
            <a href="https://example.com" target="_blank">Link</a>
            <img src="image.jpg" alt="Image">
            <script>alert('xss')</script>
        </div>
    '''
    sanitized = sanitize_protector.sanitize_input(input_html, context='html')
    assert '<div class="container">' in sanitized
    assert '<p>Hello</p>' in sanitized
    assert '<a href="https://example.com"' in sanitized
    assert '<img src="image.jpg"' in sanitized
    assert '<script>' not in sanitized

def test_escape_mode(escape_protector):
    """Test ESCAPE mode which should escape all HTML"""
    input_text = '<p>Hello</p><script>alert("xss")</script>'
    sanitized = escape_protector.sanitize_input(input_text, context='html')
    assert '&lt;p&gt;' in sanitized
    assert '&lt;script&gt;' in sanitized 