from fastapi import FastAPI
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response
from .security import create_security_headers, SecurityLevel

class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Middleware to add security headers to all responses"""
    
    def __init__(self, app: FastAPI, security_level: SecurityLevel = SecurityLevel.HIGH):
        super().__init__(app)
        self.security_headers = create_security_headers(security_level=security_level)
        
    async def dispatch(self, request: Request, call_next) -> Response:
        response = await call_next(request)
        
        # Add security headers to response
        headers = self.security_headers.get_security_headers()
        for header_name, header_value in headers.items():
            response.headers[header_name] = header_value
            
        return response 