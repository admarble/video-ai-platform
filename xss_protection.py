from typing import Dict, List, Optional, Any, Union
import re
import html
import json
from dataclasses import dataclass
from enum import Enum
import logging
from urllib.parse import urlencode, parse_qs, urlparse, urlunparse

class XSSProtectionMode(Enum):
    """XSS protection operational modes"""
    SANITIZE = "sanitize"  # Clean and sanitize input
    ESCAPE = "escape"      # Escape all special characters
    STRICT = "strict"      # Reject suspicious input

@dataclass
class XSSProtectionRule:
    """Defines XSS protection rules"""
    allowed_tags: List[str]
    allowed_attributes: Dict[str, List[str]]
    allowed_protocols: List[str]
    allow_data_urls: bool
    max_url_length: int
    sanitize_json: bool

class XSSProtector:
    """Handles XSS protection and input sanitization"""
    
    def __init__(self, mode: XSSProtectionMode = XSSProtectionMode.STRICT):
        self.mode = mode
        self.logger = logging.getLogger(__name__)
        
        # Initialize protection rules
        self.rules = self._get_default_rules()
        
        # Compile regex patterns
        self._compile_patterns()
        
    def _get_default_rules(self) -> XSSProtectionRule:
        """Get default XSS protection rules based on mode"""
        if self.mode == XSSProtectionMode.STRICT:
            return XSSProtectionRule(
                allowed_tags=[
                    'p', 'br', 'strong', 'em', 'u', 'h1', 'h2', 'h3',
                    'h4', 'h5', 'h6', 'ul', 'ol', 'li', 'blockquote'
                ],
                allowed_attributes={
                    'a': ['href', 'title'],
                    'img': ['src', 'alt', 'title', 'width', 'height'],
                    'blockquote': ['cite']
                },
                allowed_protocols=['http', 'https'],
                allow_data_urls=False,
                max_url_length=2000,
                sanitize_json=True
            )
        elif self.mode == XSSProtectionMode.SANITIZE:
            return XSSProtectionRule(
                allowed_tags=[
                    'p', 'br', 'strong', 'em', 'u', 'h1', 'h2', 'h3', 'h4', 
                    'h5', 'h6', 'ul', 'ol', 'li', 'blockquote', 'a', 'img',
                    'table', 'tr', 'td', 'th', 'thead', 'tbody', 'span', 'div'
                ],
                allowed_attributes={
                    'a': ['href', 'title', 'target'],
                    'img': ['src', 'alt', 'title', 'width', 'height'],
                    'blockquote': ['cite'],
                    'table': ['border', 'cellpadding', 'cellspacing'],
                    'td': ['colspan', 'rowspan'],
                    'div': ['class', 'id'],
                    'span': ['class', 'id']
                },
                allowed_protocols=['http', 'https', 'mailto'],
                allow_data_urls=True,
                max_url_length=5000,
                sanitize_json=True
            )
        else:  # ESCAPE mode
            return XSSProtectionRule(
                allowed_tags=[],  # Escape everything
                allowed_attributes={},
                allowed_protocols=['http', 'https'],
                allow_data_urls=False,
                max_url_length=2000,
                sanitize_json=True
            )
            
    def _compile_patterns(self) -> None:
        """Compile regex patterns for XSS detection"""
        self.patterns = {
            'script_pattern': re.compile(r'<script.*?>.*?</script>', re.I | re.S),
            'event_pattern': re.compile(r'\bon\w+\s*=', re.I),
            'js_protocol_pattern': re.compile(r'javascript:', re.I),
            'data_pattern': re.compile(r'data:', re.I),
            'expression_pattern': re.compile(r'expression\s*\(', re.I),
            'url_pattern': re.compile(r'url\s*\(', re.I),
            'html_comment_pattern': re.compile(r'<!--.*?-->', re.S),
            'entities_pattern': re.compile(r'&[#\w]+;')
        }
        
    def sanitize_input(
        self,
        value: Union[str, Dict, List],
        context: str = 'text'
    ) -> Union[str, Dict, List]:
        """Sanitize input based on context"""
        if value is None:
            return value
            
        if isinstance(value, (dict, list)) and self.rules.sanitize_json:
            return self._sanitize_json(value)
            
        if not isinstance(value, str):
            value = str(value)
            
        if context == 'url':
            return self._sanitize_url(value)
        elif context == 'html':
            return self._sanitize_html(value)
        elif context == 'js':
            return self._sanitize_javascript(value)
        else:  # text
            return self._sanitize_text(value)
            
    def _sanitize_text(self, text: str) -> str:
        """Sanitize plain text input"""
        if self.mode == XSSProtectionMode.ESCAPE:
            return html.escape(text, quote=True)
            
        # Remove potentially dangerous patterns
        text = self.patterns['script_pattern'].sub('', text)
        text = self.patterns['event_pattern'].sub('', text)
        text = self.patterns['js_protocol_pattern'].sub('', text)
        text = self.patterns['expression_pattern'].sub('', text)
        
        # Handle HTML entities
        if self.mode == XSSProtectionMode.STRICT:
            text = self.patterns['entities_pattern'].sub('', text)
            
        return html.escape(text, quote=True)
        
    def _sanitize_html(self, html_text: str) -> str:
        """Sanitize HTML content"""
        if self.mode == XSSProtectionMode.ESCAPE:
            return html.escape(html_text, quote=True)
            
        try:
            # First pass: remove potentially dangerous patterns
            html_text = self.patterns['script_pattern'].sub('', html_text)
            html_text = self.patterns['event_pattern'].sub('', html_text)
            html_text = self.patterns['js_protocol_pattern'].sub('', html_text)
            html_text = self.patterns['expression_pattern'].sub('', html_text)
            
            # Remove unclosed tags
            html_text = re.sub(r'<([^>]*?)[^>]*$', '', html_text)
            
            # Remove disallowed tags
            for tag in re.findall(r'<([^>]+)>', html_text):
                tag_name = tag.split()[0].lower()
                if tag_name not in self.rules.allowed_tags:
                    # Remove both opening and closing tags
                    html_text = re.sub(f'<{tag}[^>]*>', '', html_text)
                    html_text = re.sub(f'</{tag_name}>', '', html_text)
                
            # Clean attributes
            for tag, allowed_attrs in self.rules.allowed_attributes.items():
                pattern = fr'<{tag}\s+([^>]*?)>'
                for match in re.finditer(pattern, html_text, re.I):
                    attrs = match.group(1)
                    cleaned_attrs = []
                    
                    for attr in re.finditer(r'(\w+)\s*=\s*["\']([^"\']*)["\']', attrs):
                        name, value = attr.groups()
                        if name.lower() in allowed_attrs:
                            if name.lower() in ['src', 'href']:
                                value = self._sanitize_url(value)
                            if value:  # Only add if value is not empty after sanitization
                                cleaned_attrs.append(f'{name}="{value}"')
                            
                    html_text = html_text.replace(
                        match.group(0),
                        f'<{tag}{" " + " ".join(cleaned_attrs) if cleaned_attrs else ""}>'
                    )
                    
            # Remove any remaining disallowed attributes
            html_text = re.sub(r'\s+on\w+\s*=\s*["\'][^"\']*["\']', '', html_text)
            html_text = re.sub(r'\s+style\s*=\s*["\'][^"\']*["\']', '', html_text)
            
            return html_text
            
        except Exception as e:
            self.logger.error(f"Error sanitizing HTML: {str(e)}")
            # On error, escape everything
            return html.escape(html_text, quote=True)
        
    def _sanitize_url(self, url: str) -> str:
        """Sanitize URL"""
        if not url:
            return ''
            
        # Check URL length
        if len(url) > self.rules.max_url_length:
            return ''
            
        try:
            parsed = urlparse(url)
            
            # Check protocol
            if parsed.scheme and parsed.scheme not in self.rules.allowed_protocols:
                return ''
                
            # Check for data URLs
            if parsed.scheme == 'data' and not self.rules.allow_data_urls:
                return ''
                
            # Clean query parameters
            if parsed.query:
                query_params = parse_qs(parsed.query)
                cleaned_params = {
                    self._sanitize_text(k): [self._sanitize_text(v) for v in vals]
                    for k, vals in query_params.items()
                }
                new_query = urlencode(cleaned_params, doseq=True)
                
                # Reconstruct URL with cleaned query
                return urlunparse((
                    parsed.scheme,
                    parsed.netloc,
                    parsed.path,
                    parsed.params,
                    new_query,
                    parsed.fragment
                ))
                
            return url
            
        except Exception as e:
            self.logger.error(f"Error sanitizing URL: {str(e)}")
            return ''
            
    def _sanitize_javascript(self, js_text: str) -> str:
        """Sanitize JavaScript content"""
        if self.mode != XSSProtectionMode.SANITIZE:
            return ''  # Only allow JS in sanitize mode
            
        # Remove potentially dangerous patterns
        js_text = self.patterns['html_comment_pattern'].sub('', js_text)
        js_text = re.sub(r'eval\s*\(', '', js_text)
        js_text = re.sub(r'Function\s*\(', '', js_text)
        js_text = re.sub(r'setTimeout\s*\(', '', js_text)
        js_text = re.sub(r'setInterval\s*\(', '', js_text)
        
        return js_text
        
    def _sanitize_json(self, data: Union[Dict, List]) -> Union[Dict, List]:
        """Recursively sanitize JSON data"""
        if isinstance(data, dict):
            return {
                self._sanitize_text(k): self._sanitize_json(v)
                for k, v in data.items()
            }
        elif isinstance(data, list):
            return [self._sanitize_json(item) for item in data]
        elif isinstance(data, str):
            return self._sanitize_text(data)
        return data

    def validate_request(
        self,
        headers: Dict[str, str],
        body: Any,
        query_params: Dict[str, str]
    ) -> bool:
        """Validate complete request for XSS attempts"""
        try:
            # Check headers
            for header, value in headers.items():
                if self._contains_xss_patterns(value):
                    self.logger.warning(f"XSS attempt detected in header: {header}")
                    return False
                    
            # Check query parameters
            for param, value in query_params.items():
                if self._contains_xss_patterns(value):
                    self.logger.warning(f"XSS attempt detected in query param: {param}")
                    return False
                    
            # Check body content
            if isinstance(body, (str, dict, list)):
                if self._contains_xss_patterns(json.dumps(body)):
                    self.logger.warning("XSS attempt detected in request body")
                    return False
                    
            return True
            
        except Exception as e:
            self.logger.error(f"Error validating request: {str(e)}")
            return False
            
    def _contains_xss_patterns(self, value: str) -> bool:
        """Check if value contains XSS patterns"""
        if not isinstance(value, str):
            value = str(value)
            
        return any(
            pattern.search(value)
            for pattern in [
                self.patterns['script_pattern'],
                self.patterns['event_pattern'],
                self.patterns['js_protocol_pattern'],
                self.patterns['expression_pattern']
            ]
        )

def create_xss_protector(
    mode: XSSProtectionMode = XSSProtectionMode.STRICT
) -> XSSProtector:
    """Create XSS protection instance"""
    return XSSProtector(mode=mode) 