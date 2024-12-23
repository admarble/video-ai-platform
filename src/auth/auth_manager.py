from typing import Dict, Optional, List, Any
from enum import Enum
from datetime import datetime, timedelta
import jwt
import bcrypt
from dataclasses import dataclass
import logging
from pathlib import Path
import json
import secrets
import time
import re
import requests
from typing_extensions import Literal

class UserRole(Enum):
    """User permission roles"""
    ADMIN = "admin"
    OPERATOR = "operator" 
    VIEWER = "viewer"

class AuthError(Exception):
    """Base authentication error"""
    pass

class InvalidCredentialsError(AuthError):
    """Invalid username or password"""
    pass

class TokenError(AuthError):
    """Invalid or expired token"""
    pass

class PermissionError(AuthError):
    """Insufficient permissions"""
    pass

@dataclass
class UserCredentials:
    """User authentication credentials"""
    username: str
    password_hash: str
    role: UserRole
    is_active: bool = True
    last_login: Optional[float] = None
    failed_attempts: int = 0
    last_failed_attempt: Optional[float] = None

@dataclass 
class AuthToken:
    """JWT authentication token"""
    token: str
    expires_at: float
    token_type: Literal["access", "refresh"]

class AuthenticationManager:
    """Manages user authentication and authorization"""
    
    def __init__(
        self,
        config: Dict[str, Any],
        user_db_path: Optional[Path] = None
    ):
        self.config = config
        self.user_db_path = user_db_path or Path("users.json")
        self.logger = logging.getLogger(__name__)
        
        # Load user database
        self._load_users()
        
        # Initialize JWT signing key
        self.jwt_secret = config.get("jwt_secret") or secrets.token_hex(32)
        
        # CAPTCHA configuration
        self.recaptcha_secret = config.get("recaptcha_secret_key")
        self.recaptcha_verify_url = "https://www.google.com/recaptcha/api/siteverify"
        self.max_login_attempts = config.get("max_login_attempts", 3)
        self.login_attempt_window = config.get("login_attempt_window", 300)  # 5 minutes
        
    def _load_users(self) -> None:
        """Load user database from file"""
        self.users: Dict[str, UserCredentials] = {}
        
        if self.user_db_path.exists():
            try:
                with open(self.user_db_path) as f:
                    user_data = json.load(f)
                    
                for username, data in user_data.items():
                    self.users[username] = UserCredentials(
                        username=username,
                        password_hash=data["password_hash"],
                        role=UserRole(data["role"]),
                        is_active=data.get("is_active", True),
                        last_login=data.get("last_login")
                    )
            except Exception as e:
                self.logger.error(f"Error loading user database: {str(e)}")
                
    def _save_users(self) -> None:
        """Save user database to file"""
        try:
            user_data = {
                username: {
                    "password_hash": user.password_hash,
                    "role": user.role.value,
                    "is_active": user.is_active,
                    "last_login": user.last_login
                }
                for username, user in self.users.items()
            }
            
            with open(self.user_db_path, 'w') as f:
                json.dump(user_data, f, indent=2)
                
        except Exception as e:
            self.logger.error(f"Error saving user database: {str(e)}")
            
    def _validate_password(self, password: str) -> bool:
        """Validate password meets requirements"""
        if len(password) < self.config.get("min_password_length", 8):
            return False
            
        # Check for required character types
        has_upper = bool(re.search(r'[A-Z]', password))
        has_lower = bool(re.search(r'[a-z]', password))
        has_digit = bool(re.search(r'\d', password))
        has_special = bool(re.search(r'[!@#$%^&*(),.?":{}|<>]', password))
        
        return has_upper and has_lower and has_digit and has_special
        
    def create_user(
        self,
        username: str,
        password: str,
        role: UserRole,
        check_password_strength: bool = True
    ) -> None:
        """Create a new user account"""
        # Validate username
        if username in self.users:
            raise AuthError("Username already exists")
            
        # Validate password
        if check_password_strength and not self._validate_password(password):
            raise AuthError(
                "Password must be at least 8 characters and contain uppercase, "
                "lowercase, digit and special characters"
            )
            
        # Hash password
        password_hash = bcrypt.hashpw(
            password.encode(),
            bcrypt.gensalt()
        ).decode()
        
        # Create user
        self.users[username] = UserCredentials(
            username=username,
            password_hash=password_hash,
            role=role
        )
        
        # Save changes
        self._save_users()
        
    def _verify_captcha(self, captcha_response: str) -> bool:
        """Verify reCAPTCHA response"""
        if not self.recaptcha_secret:
            self.logger.warning("reCAPTCHA secret key not configured")
            return True
            
        try:
            response = requests.post(
                self.recaptcha_verify_url,
                data={
                    "secret": self.recaptcha_secret,
                    "response": captcha_response
                }
            )
            result = response.json()
            return result.get("success", False)
            
        except Exception as e:
            self.logger.error(f"CAPTCHA verification error: {str(e)}")
            return False
            
    def _check_requires_captcha(self, username: str) -> bool:
        """Check if login requires CAPTCHA"""
        if username not in self.users:
            return False
            
        user = self.users[username]
        current_time = time.time()
        
        # Reset failed attempts if window has expired
        if (user.last_failed_attempt and 
            current_time - user.last_failed_attempt >= self.login_attempt_window):
            user.failed_attempts = 0
            user.last_failed_attempt = None
            self._save_users()
            
        return user.failed_attempts >= self.max_login_attempts
        
    def _record_failed_attempt(self, username: str) -> None:
        """Record a failed login attempt"""
        if username in self.users:
            user = self.users[username]
            user.failed_attempts += 1
            user.last_failed_attempt = time.time()
            self._save_users()

    def _reset_failed_attempts(self, username: str) -> None:
        """Reset failed login attempts after successful login"""
        if username in self.users:
            user = self.users[username]
            user.failed_attempts = 0
            user.last_failed_attempt = None
            self._save_users()

    def authenticate(
        self,
        username: str,
        password: str,
        captcha_response: Optional[str] = None
    ) -> tuple[AuthToken, AuthToken]:
        """Authenticate user and return access/refresh tokens"""
        if username not in self.users:
            raise InvalidCredentialsError("Invalid username or password")
            
        user = self.users[username]
        
        if not user.is_active:
            raise AuthError("Account is inactive")
            
        # Check if CAPTCHA is required
        if self._check_requires_captcha(username):
            if not captcha_response:
                raise AuthError("CAPTCHA required")
                
            if not self._verify_captcha(captcha_response):
                self._record_failed_attempt(username)
                raise AuthError("Invalid CAPTCHA")
            
        # Verify password
        if not bcrypt.checkpw(
            password.encode(),
            user.password_hash.encode()
        ):
            self._record_failed_attempt(username)
            raise InvalidCredentialsError("Invalid username or password")
            
        # Reset failed attempts on successful login
        self._reset_failed_attempts(username)
            
        # Update last login
        user.last_login = time.time()
        self._save_users()
        
        # Generate tokens
        access_token = self._generate_token(
            username,
            "access",
            self.config.get("access_token_expiry", 3600)
        )
        
        refresh_token = self._generate_token(
            username,
            "refresh", 
            self.config.get("refresh_token_expiry", 86400)
        )
        
        return access_token, refresh_token
        
    def refresh_token(self, refresh_token: str) -> AuthToken:
        """Generate new access token from refresh token"""
        try:
            # Verify refresh token
            payload = jwt.decode(
                refresh_token,
                self.jwt_secret,
                algorithms=["HS256"]
            )
            
            if payload["type"] != "refresh":
                raise TokenError("Invalid token type")
                
            username = payload["sub"]
            
            if username not in self.users:
                raise TokenError("User not found")
                
            if not self.users[username].is_active:
                raise TokenError("User account inactive")
                
            # Generate new access token
            return self._generate_token(
                username,
                "access",
                self.config.get("access_token_expiry", 3600)
            )
            
        except jwt.InvalidTokenError as e:
            raise TokenError(f"Invalid refresh token: {str(e)}")
            
    def _generate_token(
        self,
        username: str,
        token_type: Literal["access", "refresh"],
        expiry: int
    ) -> AuthToken:
        """Generate a new JWT token"""
        now = datetime.utcnow()
        expires_at = now + timedelta(seconds=expiry)
        
        payload = {
            "sub": username,
            "type": token_type,
            "role": self.users[username].role.value,
            "iat": now,
            "exp": expires_at
        }
        
        token = jwt.encode(
            payload,
            self.jwt_secret,
            algorithm="HS256"
        )
        
        return AuthToken(
            token=token,
            expires_at=expires_at.timestamp(),
            token_type=token_type
        )
        
    def verify_token(self, token: str) -> Dict[str, Any]:
        """Verify and decode JWT token"""
        try:
            payload = jwt.decode(
                token,
                self.jwt_secret,
                algorithms=["HS256"]
            )
            
            username = payload["sub"]
            
            if username not in self.users:
                raise TokenError("User not found")
                
            if not self.users[username].is_active:
                raise TokenError("User account inactive")
                
            return payload
            
        except jwt.InvalidTokenError as e:
            raise TokenError(f"Invalid token: {str(e)}")
            
    def check_permission(
        self,
        token: str,
        required_role: UserRole
    ) -> None:
        """Check if token has required role permission"""
        payload = self.verify_token(token)
        
        user_role = UserRole(payload["role"])
        
        # Admin has access to everything
        if user_role == UserRole.ADMIN:
            return
            
        # Check role hierarchy
        role_hierarchy = {
            UserRole.ADMIN: 3,
            UserRole.OPERATOR: 2, 
            UserRole.VIEWER: 1
        }
        
        if role_hierarchy[user_role] < role_hierarchy[required_role]:
            raise PermissionError(
                f"Insufficient permissions. Required: {required_role.value}"
            )
            
    def change_password(
        self,
        username: str,
        old_password: str,
        new_password: str
    ) -> None:
        """Change user password"""
        if username not in self.users:
            raise AuthError("User not found")
            
        user = self.users[username]
        
        # Verify old password
        if not bcrypt.checkpw(
            old_password.encode(),
            user.password_hash.encode()
        ):
            raise InvalidCredentialsError("Invalid current password")
            
        # Validate new password
        if not self._validate_password(new_password):
            raise AuthError(
                "Password must be at least 8 characters and contain uppercase, "
                "lowercase, digit and special characters"
            )
            
        # Update password
        user.password_hash = bcrypt.hashpw(
            new_password.encode(),
            bcrypt.gensalt()
        ).decode()
        
        self._save_users()
        
    def reset_password(
        self,
        username: str,
        new_password: str,
        admin_token: str
    ) -> None:
        """Reset user password (admin only)"""
        # Verify admin permissions
        try:
            self.check_permission(admin_token, UserRole.ADMIN)
        except AuthError:
            raise PermissionError("Admin privileges required")
            
        if username not in self.users:
            raise AuthError("User not found")
            
        # Validate new password
        if not self._validate_password(new_password):
            raise AuthError(
                "Password must be at least 8 characters and contain uppercase, "
                "lowercase, digit and special characters"
            )
            
        # Update password
        user = self.users[username]
        user.password_hash = bcrypt.hashpw(
            new_password.encode(),
            bcrypt.gensalt()
        ).decode()
        
        self._save_users()
        
    def deactivate_user(
        self,
        username: str,
        admin_token: str
    ) -> None:
        """Deactivate user account (admin only)"""
        # Verify admin permissions
        try:
            self.check_permission(admin_token, UserRole.ADMIN)
        except AuthError:
            raise PermissionError("Admin privileges required")
            
        if username not in self.users:
            raise AuthError("User not found")
            
        # Deactivate account
        self.users[username].is_active = False
        self._save_users()
        
    def get_user_info(
        self,
        username: str,
        admin_token: str
    ) -> Dict[str, Any]:
        """Get user account information (admin only)"""
        # Verify admin permissions
        try:
            self.check_permission(admin_token, UserRole.ADMIN)
        except AuthError:
            raise PermissionError("Admin privileges required")
            
        if username not in self.users:
            raise AuthError("User not found")
            
        user = self.users[username]
        return {
            "username": user.username,
            "role": user.role.value,
            "is_active": user.is_active,
            "last_login": user.last_login
        }

# Example usage
def create_auth_manager(config_path: Path) -> AuthenticationManager:
    """Create authentication manager instance"""
    with open(config_path) as f:
        config = json.load(f)
        
    return AuthenticationManager(
        config=config,
        user_db_path=Path(config.get("user_db_path", "users.json"))
    ) 