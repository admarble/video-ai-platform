from pydantic import BaseSettings, SecretStr

class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # Stripe Configuration
    STRIPE_SECRET_KEY: SecretStr
    STRIPE_PUBLIC_KEY: str
    STRIPE_WEBHOOK_SECRET: SecretStr
    
    # Database Configuration
    DATABASE_URL: str
    
    # Application Configuration
    APP_NAME: str = "Cuthrough"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False
    
    class Config:
        env_file = ".env"
        case_sensitive = True 