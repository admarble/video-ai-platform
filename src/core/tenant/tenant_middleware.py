from typing import Optional, Callable, Awaitable, Dict
from fastapi import Request, Response
from fastapi.responses import JSONResponse
import time
import asyncio
from .tenant_manager import TenantManager

class TenantMiddleware:
    """Middleware for tenant isolation and resource management"""
    
    def __init__(self, tenant_manager: TenantManager):
        self.tenant_manager = tenant_manager
        self._rate_limits: Dict[str, Dict[str, float]] = {}  # tenant_id -> {ip -> last_request_time}
        
    async def __call__(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        # Extract tenant ID from header or subdomain
        tenant_id = self._get_tenant_id(request)
        if not tenant_id:
            return JSONResponse(
                status_code=400,
                content={"error": "Missing tenant identification"}
            )
            
        # Validate tenant
        tenant = self.tenant_manager.get_tenant(tenant_id)
        if not tenant:
            return JSONResponse(
                status_code=404,
                content={"error": "Tenant not found"}
            )
            
        # Check rate limits
        if not self._check_rate_limit(tenant_id, request.client.host, tenant.api_quota):
            return JSONResponse(
                status_code=429,
                content={"error": "Rate limit exceeded"}
            )
            
        # Check storage quota for upload endpoints
        if request.method == "POST" and "upload" in request.url.path:
            if not self.tenant_manager.check_storage_quota(tenant_id):
                return JSONResponse(
                    status_code=413,
                    content={"error": "Storage quota exceeded"}
                )
                
        # Add tenant context to request state
        request.state.tenant_id = tenant_id
        request.state.tenant = tenant
        
        # Process request
        try:
            response = await call_next(request)
            
            # Add tenant tracking headers
            response.headers["X-Tenant-ID"] = tenant_id
            
            return response
        except Exception as e:
            return JSONResponse(
                status_code=500,
                content={"error": f"Internal server error: {str(e)}"}
            )
            
    def _get_tenant_id(self, request: Request) -> Optional[str]:
        """Extract tenant ID from request"""
        # Try header first
        tenant_id = request.headers.get("X-Tenant-ID")
        if tenant_id:
            return tenant_id
            
        # Try subdomain
        host = request.headers.get("host", "")
        if "." in host:
            subdomain = host.split(".")[0]
            if subdomain != "www":
                return f"tenant_{subdomain}"
                
        # Try custom domain mapping
        for tenant in self.tenant_manager.tenants.values():
            if tenant.custom_domain and tenant.custom_domain == host:
                return tenant.tenant_id
                
        return None
        
    def _check_rate_limit(self, tenant_id: str, ip: str, limit: int) -> bool:
        """Check if request is within rate limits"""
        now = time.time()
        
        # Initialize rate limit tracking for tenant
        if tenant_id not in self._rate_limits:
            self._rate_limits[tenant_id] = {}
            
        # Clean old entries
        self._rate_limits[tenant_id] = {
            ip_: last_time
            for ip_, last_time in self._rate_limits[tenant_id].items()
            if now - last_time < 60  # Keep last minute only
        }
        
        # Check rate limit
        tenant_ips = self._rate_limits[tenant_id]
        request_count = len([t for t in tenant_ips.values() if now - t < 60])
        
        if request_count >= limit:
            return False
            
        # Update last request time
        tenant_ips[ip] = now
        return True 