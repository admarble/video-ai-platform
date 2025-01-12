from pathlib import Path
from typing import Optional
from cryptography.fernet import Fernet

class SecretsManager:
    """Manages secure storage and retrieval of secrets"""
    
    def __init__(self, vault_path: str, encryption_key: str):
        self.vault_path = Path(vault_path)
        self.fernet = Fernet(encryption_key.encode())
        self.vault_path.mkdir(parents=True, exist_ok=True)
        
    def store_secret(self, key: str, value: str) -> None:
        """Store encrypted secret"""
        encrypted = self.fernet.encrypt(value.encode())
        secret_path = self.vault_path / f"{key}.enc"
        with open(secret_path, 'wb') as f:
            f.write(encrypted)
            
    def get_secret(self, key: str) -> Optional[str]:
        """Retrieve and decrypt secret"""
        secret_path = self.vault_path / f"{key}.enc"
        if not secret_path.exists():
            return None
            
        with open(secret_path, 'rb') as f:
            encrypted = f.read()
        return self.fernet.decrypt(encrypted).decode()
        
    def rotate_encryption_key(self, new_key: str) -> None:
        """Rotate encryption key and re-encrypt all secrets"""
        new_fernet = Fernet(new_key.encode())
        
        for secret_file in self.vault_path.glob("*.enc"):
            with open(secret_file, 'rb') as f:
                encrypted = f.read()
            decrypted = self.fernet.decrypt(encrypted)
            new_encrypted = new_fernet.encrypt(decrypted)
            
            with open(secret_file, 'wb') as f:
                f.write(new_encrypted)
                
        self.fernet = new_fernet 