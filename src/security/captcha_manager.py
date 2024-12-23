from typing import Dict, Optional, Any, List
from enum import Enum
from dataclasses import dataclass
import aiohttp
import logging
import json
import time
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .rate_limiter.rate_limiter import RateLimiter

class CaptchaProvider(Enum):
    """Supported CAPTCHA providers"""
    RECAPTCHA = "recaptcha"
    HCAPTCHA = "hcaptcha"
    CLOUDFLARE_TURNSTILE = "cloudflare"
    CUSTOM = "custom"

@dataclass
class CaptchaConfig:
    """CAPTCHA configuration settings"""
    provider: CaptchaProvider
    site_key: str
    secret_key: str
    min_score: float = 0.5  # For providers that support risk scores
    timeout: int = 30  # seconds
    allowed_domains: List[str] = None
    require_verified_tokens: bool = True

class CaptchaError(Exception):
    """Base exception for CAPTCHA-related errors"""
    pass

class CaptchaProviderBase(ABC):
    """Base class for CAPTCHA providers"""
    
    def __init__(self, config: CaptchaConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
    @abstractmethod
    async def verify_token(self, token: str, remote_ip: Optional[str] = None) -> Dict[str, Any]:
        """Verify CAPTCHA token"""
        pass
    
    @abstractmethod
    def get_client_config(self) -> Dict[str, Any]:
        """Get client-side configuration"""
        pass

class ReCaptchaProvider(CaptchaProviderBase):
    """Google reCAPTCHA implementation"""
    
    VERIFY_URL = "https://www.google.com/recaptcha/api/siteverify"
    
    async def verify_token(self, token: str, remote_ip: Optional[str] = None) -> Dict[str, Any]:
        try:
            async with aiohttp.ClientSession() as session:
                payload = {
                    'secret': self.config.secret_key,
                    'response': token
                }
                if remote_ip:
                    payload['remoteip'] = remote_ip
                
                async with session.post(self.VERIFY_URL, data=payload) as response:
                    result = await response.json()
                    
                    if not result.get('success'):
                        raise CaptchaError(
                            f"reCAPTCHA verification failed: {result.get('error-codes')}"
                        )
                    
                    # Check score for v3
                    score = result.get('score')
                    if score is not None and score < self.config.min_score:
                        raise CaptchaError(f"reCAPTCHA score too low: {score}")
                    
                    return result
                    
        except Exception as e:
            self.logger.error(f"reCAPTCHA verification error: {str(e)}")
            raise CaptchaError(str(e))
    
    def get_client_config(self) -> Dict[str, Any]:
        return {
            'provider': 'recaptcha',
            'siteKey': self.config.site_key,
            'minScore': self.config.min_score
        }

class HCaptchaProvider(CaptchaProviderBase):
    """hCaptcha implementation"""
    
    VERIFY_URL = "https://api.hcaptcha.com/siteverify"
    
    async def verify_token(self, token: str, remote_ip: Optional[str] = None) -> Dict[str, Any]:
        try:
            async with aiohttp.ClientSession() as session:
                payload = {
                    'secret': self.config.secret_key,
                    'response': token,
                    'sitekey': self.config.site_key
                }
                if remote_ip:
                    payload['remoteip'] = remote_ip
                
                async with session.post(self.VERIFY_URL, data=payload) as response:
                    result = await response.json()
                    
                    if not result.get('success'):
                        raise CaptchaError(
                            f"hCaptcha verification failed: {result.get('error-codes')}"
                        )
                    
                    return result
                    
        except Exception as e:
            self.logger.error(f"hCaptcha verification error: {str(e)}")
            raise CaptchaError(str(e))
    
    def get_client_config(self) -> Dict[str, Any]:
        return {
            'provider': 'hcaptcha',
            'siteKey': self.config.site_key
        }

class CloudflareTurnstileProvider(CaptchaProviderBase):
    """Cloudflare Turnstile implementation"""
    
    VERIFY_URL = "https://challenges.cloudflare.com/turnstile/v0/siteverify"
    
    async def verify_token(self, token: str, remote_ip: Optional[str] = None) -> Dict[str, Any]:
        try:
            async with aiohttp.ClientSession() as session:
                payload = {
                    'secret': self.config.secret_key,
                    'response': token
                }
                if remote_ip:
                    payload['remoteip'] = remote_ip
                
                async with session.post(self.VERIFY_URL, data=payload) as response:
                    result = await response.json()
                    
                    if not result.get('success'):
                        raise CaptchaError(
                            f"Turnstile verification failed: {result.get('error-codes')}"
                        )
                    
                    return result
                    
        except Exception as e:
            self.logger.error(f"Turnstile verification error: {str(e)}")
            raise CaptchaError(str(e))
    
    def get_client_config(self) -> Dict[str, Any]:
        return {
            'provider': 'turnstile',
            'siteKey': self.config.site_key
        }

class CaptchaManager:
    """Manages CAPTCHA verification and rate limiting"""
    
    def __init__(
        self,
        config: CaptchaConfig,
        rate_limiter: Optional['RateLimiter'] = None
    ):
        self.config = config
        self.rate_limiter = rate_limiter
        self.logger = logging.getLogger(__name__)
        
        # Initialize provider
        self.provider = self._create_provider()
        
        # Track verified tokens
        self.verified_tokens = {}
        
    def _create_provider(self) -> CaptchaProviderBase:
        """Create CAPTCHA provider instance"""
        if self.config.provider == CaptchaProvider.RECAPTCHA:
            return ReCaptchaProvider(self.config)
        elif self.config.provider == CaptchaProvider.HCAPTCHA:
            return HCaptchaProvider(self.config)
        elif self.config.provider == CaptchaProvider.CLOUDFLARE_TURNSTILE:
            return CloudflareTurnstileProvider(self.config)
        else:
            raise ValueError(f"Unsupported CAPTCHA provider: {self.config.provider}")
    
    async def verify_request(
        self,
        token: str,
        remote_ip: Optional[str] = None,
        action: Optional[str] = None
    ) -> bool:
        """Verify CAPTCHA token for request"""
        try:
            # Check if rate limited
            if self.rate_limiter:
                if not await self.rate_limiter.check_rate_limit(remote_ip or 'unknown'):
                    raise CaptchaError("Rate limit exceeded")
            
            # Check if token was already verified
            if self.config.require_verified_tokens:
                if token in self.verified_tokens:
                    token_info = self.verified_tokens[token]
                    if time.time() - token_info['timestamp'] > self.config.timeout:
                        del self.verified_tokens[token]
                        raise CaptchaError("CAPTCHA token expired")
                    return True
            
            # Verify token with provider
            result = await self.provider.verify_token(token, remote_ip)
            
            # Check action if specified
            if action and result.get('action') != action:
                raise CaptchaError(f"Invalid action: {result.get('action')}")
            
            # Store verified token
            if self.config.require_verified_tokens:
                self.verified_tokens[token] = {
                    'timestamp': time.time(),
                    'ip': remote_ip,
                    'result': result
                }
            
            return True
            
        except Exception as e:
            self.logger.error(f"CAPTCHA verification failed: {str(e)}")
            return False
    
    def get_client_config(self) -> Dict[str, Any]:
        """Get client-side CAPTCHA configuration"""
        return {
            **self.provider.get_client_config(),
            'timeout': self.config.timeout,
            'allowedDomains': self.config.allowed_domains
        }
    
    def cleanup_expired_tokens(self) -> None:
        """Remove expired verified tokens"""
        current_time = time.time()
        expired = [
            token for token, info in self.verified_tokens.items()
            if current_time - info['timestamp'] > self.config.timeout
        ]
        for token in expired:
            del self.verified_tokens[token] 