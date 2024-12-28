from typing import Dict, Optional, List
from dataclasses import dataclass
from datetime import datetime
import json
import asyncio
from pathlib import Path

@dataclass
class TenantConfig:
    tenant_id: str
    name: str
    storage_quota: int  # in bytes
    api_quota: int  # requests per minute
    created_at: datetime
    features: Dict[str, bool]  # feature flags
    custom_domain: Optional[str] = None
    
class TenantManager:
    """Manages tenant isolation and resources"""
    
    def __init__(self, config_path: Path):
        self.config_path = config_path
        self.tenants: Dict[str, TenantConfig] = {}
        self._load_tenants()
        
    def _load_tenants(self) -> None:
        """Load tenant configurations from disk"""
        if not self.config_path.exists():
            self.config_path.mkdir(parents=True)
            return
            
        for tenant_file in self.config_path.glob("*.json"):
            with open(tenant_file) as f:
                data = json.load(f)
                self.tenants[data["tenant_id"]] = TenantConfig(
                    tenant_id=data["tenant_id"],
                    name=data["name"],
                    storage_quota=data["storage_quota"],
                    api_quota=data["api_quota"],
                    created_at=datetime.fromisoformat(data["created_at"]),
                    features=data["features"],
                    custom_domain=data.get("custom_domain")
                )
                
    def create_tenant(self, name: str, storage_quota: int, api_quota: int,
                     features: Optional[Dict[str, bool]] = None) -> TenantConfig:
        """Create a new tenant with isolation"""
        tenant_id = f"tenant_{len(self.tenants) + 1}"
        tenant = TenantConfig(
            tenant_id=tenant_id,
            name=name,
            storage_quota=storage_quota,
            api_quota=api_quota,
            created_at=datetime.utcnow(),
            features=features or {}
        )
        
        # Save tenant config
        self.tenants[tenant_id] = tenant
        self._save_tenant(tenant)
        
        # Create isolated storage directories
        tenant_path = self.config_path / tenant_id
        (tenant_path / "storage").mkdir(parents=True)
        (tenant_path / "cache").mkdir(parents=True)
        (tenant_path / "processed").mkdir(parents=True)
        
        return tenant
        
    def get_tenant(self, tenant_id: str) -> Optional[TenantConfig]:
        """Get tenant configuration"""
        return self.tenants.get(tenant_id)
        
    def update_tenant(self, tenant_id: str, **kwargs) -> Optional[TenantConfig]:
        """Update tenant configuration"""
        tenant = self.tenants.get(tenant_id)
        if not tenant:
            return None
            
        # Update fields
        for key, value in kwargs.items():
            if hasattr(tenant, key):
                setattr(tenant, key, value)
                
        self._save_tenant(tenant)
        return tenant
        
    def delete_tenant(self, tenant_id: str) -> bool:
        """Delete tenant and all associated data"""
        if tenant_id not in self.tenants:
            return False
            
        # Remove tenant config
        del self.tenants[tenant_id]
        tenant_file = self.config_path / f"{tenant_id}.json"
        if tenant_file.exists():
            tenant_file.unlink()
            
        # Remove tenant directories
        tenant_path = self.config_path / tenant_id
        if tenant_path.exists():
            import shutil
            shutil.rmtree(tenant_path)
            
        return True
        
    def _save_tenant(self, tenant: TenantConfig) -> None:
        """Save tenant configuration to disk"""
        tenant_file = self.config_path / f"{tenant.tenant_id}.json"
        with open(tenant_file, 'w') as f:
            json.dump({
                "tenant_id": tenant.tenant_id,
                "name": tenant.name,
                "storage_quota": tenant.storage_quota,
                "api_quota": tenant.api_quota,
                "created_at": tenant.created_at.isoformat(),
                "features": tenant.features,
                "custom_domain": tenant.custom_domain
            }, f, indent=2)
            
    def get_tenant_path(self, tenant_id: str, path_type: str = "storage") -> Optional[Path]:
        """Get isolated storage path for tenant"""
        tenant = self.tenants.get(tenant_id)
        if not tenant:
            return None
            
        base_path = self.config_path / tenant_id
        if path_type == "storage":
            return base_path / "storage"
        elif path_type == "cache":
            return base_path / "cache"
        elif path_type == "processed":
            return base_path / "processed"
        return None
        
    def check_storage_quota(self, tenant_id: str) -> bool:
        """Check if tenant is within storage quota"""
        tenant = self.tenants.get(tenant_id)
        if not tenant:
            return False
            
        # Calculate total storage used
        total_size = 0
        tenant_path = self.config_path / tenant_id
        for path_type in ["storage", "cache", "processed"]:
            path = tenant_path / path_type
            if path.exists():
                total_size += sum(f.stat().st_size for f in path.rglob('*') if f.is_file())
                
        return total_size <= tenant.storage_quota 