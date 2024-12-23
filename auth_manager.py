from typing import Dict, Optional, Any
import jwt
from datetime import datetime, timedelta
import logging

class AuthenticationManager:
    """Manages authentication and token verification"""
    
    def __init__(
        self,
        secret_key: str,
        token_expiry: int = 3600,  # 1 hour default
        algorithm: str = 'HS256'
    ):
        self.secret_key = secret_key
        self.token_expiry = token_expiry
        self.algorithm = algorithm
        self.logger = logging.getLogger(__name__)

    async def create_token(self, user_data: Dict[str, Any]) -> str:
        """Create a new JWT token for a user"""
        try:
            payload = {
                **user_data,
                'exp': datetime.utcnow() + timedelta(seconds=self.token_expiry),
                'iat': datetime.utcnow()
            }
            return jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
        except Exception as e:
            self.logger.error(f"Token creation error: {str(e)}")
            raise

    async def verify_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Verify and decode a JWT token"""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            return payload
        except jwt.ExpiredSignatureError:
            self.logger.warning("Token has expired")
            return None
        except jwt.InvalidTokenError as e:
            self.logger.error(f"Invalid token: {str(e)}")
            return None
        except Exception as e:
            self.logger.error(f"Token verification error: {str(e)}")
            return None

    async def refresh_token(self, token: str) -> Optional[str]:
        """Refresh an existing token"""
        user_data = await self.verify_token(token)
        if user_data:
            # Remove previous token metadata
            user_data.pop('exp', None)
            user_data.pop('iat', None)
            return await self.create_token(user_data)
        return None

    def validate_credentials(self, username: str, password: str) -> Optional[Dict[str, Any]]:
        """
        Validate user credentials
        This is a placeholder - implement your own authentication logic
        """
        raise NotImplementedError("Implement your own credential validation logic") 