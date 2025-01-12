from typing import Optional, BinaryIO, List
from pathlib import Path
import shutil
import os
from .tenant_manager import TenantManager

class TenantStorageAdapter:
    """Handles tenant-isolated file storage operations"""
    
    def __init__(self, tenant_manager: TenantManager):
        self.tenant_manager = tenant_manager
        
    def store_file(self, tenant_id: str, file: BinaryIO, filename: str,
                   path_type: str = "storage") -> Optional[Path]:
        """Store file in tenant's isolated storage"""
        # Get tenant storage path
        storage_path = self.tenant_manager.get_tenant_path(tenant_id, path_type)
        if not storage_path:
            return None
            
        # Check quota before storing
        if not self.tenant_manager.check_storage_quota(tenant_id):
            return None
            
        # Create safe filename
        safe_filename = self._sanitize_filename(filename)
        file_path = storage_path / safe_filename
        
        # Ensure unique filename
        counter = 1
        while file_path.exists():
            name, ext = os.path.splitext(safe_filename)
            file_path = storage_path / f"{name}_{counter}{ext}"
            counter += 1
            
        # Store file
        with open(file_path, 'wb') as f:
            shutil.copyfileobj(file, f)
            
        return file_path
        
    def get_file(self, tenant_id: str, filename: str,
                 path_type: str = "storage") -> Optional[Path]:
        """Get file from tenant's storage"""
        storage_path = self.tenant_manager.get_tenant_path(tenant_id, path_type)
        if not storage_path:
            return None
            
        file_path = storage_path / self._sanitize_filename(filename)
        return file_path if file_path.exists() else None
        
    def delete_file(self, tenant_id: str, filename: str,
                    path_type: str = "storage") -> bool:
        """Delete file from tenant's storage"""
        file_path = self.get_file(tenant_id, filename, path_type)
        if not file_path:
            return False
            
        try:
            file_path.unlink()
            return True
        except Exception:
            return False
            
    def list_files(self, tenant_id: str, path_type: str = "storage") -> List[str]:
        """List files in tenant's storage"""
        storage_path = self.tenant_manager.get_tenant_path(tenant_id, path_type)
        if not storage_path or not storage_path.exists():
            return []
            
        return [f.name for f in storage_path.iterdir() if f.is_file()]
        
    def get_storage_usage(self, tenant_id: str) -> int:
        """Get total storage usage for tenant in bytes"""
        tenant_path = self.tenant_manager.config_path / tenant_id
        if not tenant_path.exists():
            return 0
            
        total_size = 0
        for path_type in ["storage", "cache", "processed"]:
            path = tenant_path / path_type
            if path.exists():
                total_size += sum(f.stat().st_size for f in path.rglob('*') if f.is_file())
                
        return total_size
        
    @staticmethod
    def _sanitize_filename(filename: str) -> str:
        """Create safe filename"""
        # Remove potentially dangerous characters
        safe_chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-._"
        filename = ''.join(c for c in filename if c in safe_chars)
        
        # Ensure filename is not empty and has reasonable length
        if not filename:
            filename = "unnamed_file"
        if len(filename) > 255:
            name, ext = os.path.splitext(filename)
            filename = name[:255-len(ext)] + ext
            
        return filename 