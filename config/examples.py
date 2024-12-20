from config.config_manager import ConfigManager, Environment
from pathlib import Path

def basic_usage():
    """Basic configuration usage example"""
    # Initialize configuration manager
    config_manager = ConfigManager()
    
    # Load configuration for development environment
    config = config_manager.load_config(
        config_path="config/config.base.yaml",
        environment=Environment.DEV
    )
    
    # Access configuration values
    print(f"Batch size: {config.models.batch_size}")  # 16 (from dev config)
    print(f"Cache directory: {config.cache_dir}")  # ~/.video_ai_cache (from base config)
    print(f"Log level: {config.log_level}")  # DEBUG (from dev config)

def environment_specific_usage():
    """Demonstrate loading different environment configurations"""
    config_manager = ConfigManager()
    
    # Load configurations for different environments
    dev_config = config_manager.load_config("config/config.base.yaml", Environment.DEV)
    staging_config = config_manager.load_config("config/config.base.yaml", Environment.STAGING)
    prod_config = config_manager.load_config("config/config.base.yaml", Environment.PROD)
    
    # Compare settings across environments
    print(f"Dev batch size: {dev_config.models.batch_size}")  # 16
    print(f"Staging batch size: {staging_config.models.batch_size}")  # 24
    print(f"Prod batch size: {prod_config.models.batch_size}")  # 32

def update_configuration():
    """Demonstrate runtime configuration updates"""
    config_manager = ConfigManager()
    config = config_manager.load_config("config/config.base.yaml", Environment.DEV)
    
    # Update configuration
    config_manager.update_config({
        "models": {
            "batch_size": 64,
            "device": "cuda"
        },
        "processing": {
            "max_video_size": 2_000_000_000
        }
    })
    
    # Access updated values
    updated_config = config_manager.get_config()
    print(f"Updated batch size: {updated_config.models.batch_size}")  # 64
    print(f"Updated device: {updated_config.models.device}")  # cuda

def secrets_management():
    """Demonstrate secrets management with environment variables"""
    import os
    
    # Set environment variables
    os.environ["API_KEY"] = "your-api-key"
    os.environ["SECRET_KEY"] = "your-secret-key"
    
    config_manager = ConfigManager()
    config = config_manager.load_config("config/config.base.yaml", Environment.PROD)
    
    # Access secrets
    print(f"API Key: {config.security.api_key}")  # from API_KEY env var
    print(f"Secret Key: {config.security.secret_key}")  # from SECRET_KEY env var

def main():
    """Run all examples"""
    print("\n=== Basic Usage ===")
    basic_usage()
    
    print("\n=== Environment-specific Usage ===")
    environment_specific_usage()
    
    print("\n=== Configuration Updates ===")
    update_configuration()
    
    print("\n=== Secrets Management ===")
    secrets_management()

if __name__ == "__main__":
    main() 