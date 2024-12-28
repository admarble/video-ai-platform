import pytest
from pathlib import Path
import tempfile
import io
from datetime import datetime
from src.core.tenant.tenant_manager import TenantManager, TenantConfig
from src.core.tenant.storage_adapter import TenantStorageAdapter
from src.core.tenant.tenant_middleware import TenantMiddleware
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

@pytest.fixture
def temp_tenant_dir():
    """Create temporary directory for tenant data"""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)

@pytest.fixture
def tenant_manager(temp_tenant_dir):
    """Create TenantManager instance"""
    return TenantManager(temp_tenant_dir)

@pytest.fixture
def storage_adapter(tenant_manager):
    """Create TenantStorageAdapter instance"""
    return TenantStorageAdapter(tenant_manager)

@pytest.fixture
def test_tenant(tenant_manager):
    """Create a test tenant"""
    return tenant_manager.create_tenant(
        name="Test Tenant",
        storage_quota=1024 * 1024,  # 1MB
        api_quota=60,  # 60 requests per minute
        features={"feature1": True}
    )

def test_tenant_creation(tenant_manager):
    """Test tenant creation and isolation"""
    tenant = tenant_manager.create_tenant(
        name="Test",
        storage_quota=1024 * 1024,
        api_quota=60
    )
    
    assert tenant.tenant_id.startswith("tenant_")
    assert tenant.name == "Test"
    assert tenant.storage_quota == 1024 * 1024
    assert tenant.api_quota == 60
    
    # Check directory creation
    tenant_path = tenant_manager.config_path / tenant.tenant_id
    assert (tenant_path / "storage").exists()
    assert (tenant_path / "cache").exists()
    assert (tenant_path / "processed").exists()

def test_tenant_isolation(storage_adapter, test_tenant):
    """Test tenant storage isolation"""
    # Create test files
    file1 = io.BytesIO(b"test data 1")
    file2 = io.BytesIO(b"test data 2")
    
    # Store files for tenant
    path1 = storage_adapter.store_file(test_tenant.tenant_id, file1, "test1.txt")
    path2 = storage_adapter.store_file(test_tenant.tenant_id, file2, "test2.txt")
    
    assert path1 and path2
    assert path1.parent == path2.parent
    assert path1.parent == storage_adapter.tenant_manager.get_tenant_path(
        test_tenant.tenant_id, "storage"
    )
    
    # Verify file contents
    assert path1.read_bytes() == b"test data 1"
    assert path2.read_bytes() == b"test data 2"
    
    # Try accessing nonexistent tenant
    assert storage_adapter.store_file("nonexistent", file1, "test.txt") is None

def test_storage_quota(storage_adapter, test_tenant):
    """Test storage quota enforcement"""
    # Create large file exceeding quota
    large_data = b"x" * (test_tenant.storage_quota + 1024)
    large_file = io.BytesIO(large_data)
    
    # Attempt to store file
    path = storage_adapter.store_file(test_tenant.tenant_id, large_file, "large.txt")
    assert path is None  # Should fail due to quota

def test_tenant_middleware():
    """Test tenant middleware functionality"""
    app = FastAPI()
    tenant_manager = TenantManager(Path(tempfile.mkdtemp()))
    middleware = TenantMiddleware(tenant_manager)
    
    # Create test tenant
    tenant = tenant_manager.create_tenant(
        name="Test API",
        storage_quota=1024 * 1024,
        api_quota=60
    )
    
    @app.get("/test")
    async def test_endpoint(request: Request):
        return {"tenant_id": request.state.tenant_id}
    
    app.middleware("http")(middleware)
    client = TestClient(app)
    
    # Test with valid tenant
    response = client.get("/test", headers={"X-Tenant-ID": tenant.tenant_id})
    assert response.status_code == 200
    assert response.json()["tenant_id"] == tenant.tenant_id
    
    # Test with invalid tenant
    response = client.get("/test", headers={"X-Tenant-ID": "nonexistent"})
    assert response.status_code == 404
    
    # Test rate limiting
    responses = [
        client.get("/test", headers={"X-Tenant-ID": tenant.tenant_id})
        for _ in range(tenant.api_quota + 1)
    ]
    assert responses[-1].status_code == 429  # Last request should be rate limited

def test_tenant_cleanup(tenant_manager, storage_adapter, test_tenant):
    """Test tenant deletion and cleanup"""
    # Store some files
    file1 = io.BytesIO(b"test data")
    path1 = storage_adapter.store_file(test_tenant.tenant_id, file1, "test.txt")
    assert path1.exists()
    
    # Delete tenant
    assert tenant_manager.delete_tenant(test_tenant.tenant_id)
    
    # Verify cleanup
    assert not path1.exists()
    assert test_tenant.tenant_id not in tenant_manager.tenants
    assert not (tenant_manager.config_path / test_tenant.tenant_id).exists() 