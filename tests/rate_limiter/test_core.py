import pytest
from datetime import datetime
from rate_limiter import (
    RateLimiter,
    RateLimitType,
    RateLimit,
    RateLimitDecorator as rate_limit,
    RateLimitExceeded
)

class TestRateLimiter:
    def test_default_limits(self, rate_limiter):
        """Test default rate limits are properly configured"""
        assert RateLimitType.REQUESTS in rate_limiter.default_limits
        assert RateLimitType.PROCESSING in rate_limiter.default_limits
        assert RateLimitType.UPLOADS in rate_limiter.default_limits
        assert RateLimitType.API in rate_limiter.default_limits
        
        # Verify default values
        requests_limit = rate_limiter.default_limits[RateLimitType.REQUESTS]
        assert requests_limit.limit == 60
        assert requests_limit.window == 60
    
    def test_custom_limit(self, rate_limiter):
        """Test rate limiting with custom limit"""
        custom_limit = RateLimit(
            limit=3,
            window=30,
            type=RateLimitType.REQUESTS
        )
        
        # Test with custom limit
        for _ in range(3):
            assert rate_limiter.check_rate_limit(
                RateLimitType.REQUESTS,
                "test_scope",
                custom_limit
            )
        
        # Should exceed custom limit
        assert not rate_limiter.check_rate_limit(
            RateLimitType.REQUESTS,
            "test_scope",
            custom_limit
        )
    
    def test_different_scopes(self, rate_limiter):
        """Test rate limiting with different scopes"""
        # Should be tracked separately
        for _ in range(30):
            assert rate_limiter.check_rate_limit(RateLimitType.REQUESTS, "scope1")
            assert rate_limiter.check_rate_limit(RateLimitType.REQUESTS, "scope2")
        
        # Verify different scope counts
        info1 = rate_limiter.get_limit_info(RateLimitType.REQUESTS, "scope1")
        info2 = rate_limiter.get_limit_info(RateLimitType.REQUESTS, "scope2")
        
        assert info1["remaining"] == info2["remaining"]
        assert info1["remaining"] == 30  # Default is 60 requests per minute

    def test_multiple_limit_types(self, rate_limiter):
        """Test handling multiple rate limit types"""
        # Different limit types should be independent
        for _ in range(10):
            assert rate_limiter.check_rate_limit(RateLimitType.REQUESTS, "scope")
            assert rate_limiter.check_rate_limit(RateLimitType.PROCESSING, "scope")
            assert rate_limiter.check_rate_limit(RateLimitType.UPLOADS, "scope")
            assert rate_limiter.check_rate_limit(RateLimitType.API, "scope")

    def test_limit_info_format(self, rate_limiter):
        """Test format of limit info response"""
        info = rate_limiter.get_limit_info(RateLimitType.REQUESTS, "test")
        
        assert "limit" in info
        assert "remaining" in info
        assert "reset" in info
        assert "window" in info
        
        # Verify reset is valid ISO format
        datetime.fromisoformat(info["reset"])
        
        # Verify values are correct type
        assert isinstance(info["limit"], int)
        assert isinstance(info["remaining"], int)
        assert isinstance(info["window"], int)

class TestRateLimitDecorator:
    def test_decorator_within_limit(self, rate_limiter):
        """Test decorator allows calls within limit"""
        
        @rate_limit(rate_limiter, RateLimitType.REQUESTS)
        def test_func():
            return "success"
        
        # Should work multiple times within limit
        for _ in range(10):
            assert test_func() == "success"
    
    def test_decorator_exceeded(self, rate_limiter):
        """Test decorator blocks calls when limit exceeded"""
        
        @rate_limit(
            rate_limiter, 
            RateLimitType.REQUESTS,
            custom_limit=RateLimit(limit=1, window=60, type=RateLimitType.REQUESTS)
        )
        def test_func():
            return "success"
            
        # First call should succeed
        assert test_func() == "success"
        
        # Second call should raise RateLimitExceeded
        with pytest.raises(RateLimitExceeded):
            test_func()

    def test_decorator_with_scope(self, rate_limiter):
        """Test decorator with scope function"""
        
        def get_scope(*args, **kwargs):
            return kwargs.get('user_id', 'default')
        
        @rate_limit(rate_limiter, RateLimitType.REQUESTS, scope_func=get_scope)
        def test_func(user_id: str):
            return "success"
            
        # Different users should have separate limits
        for _ in range(30):
            assert test_func(user_id="user1") == "success"
            assert test_func(user_id="user2") == "success"

    def test_decorator_error_handling(self, rate_limiter):
        """Test decorator handles errors properly"""
        
        @rate_limit(rate_limiter, RateLimitType.REQUESTS)
        def failing_func():
            raise ValueError("Test error")
            
        # Should still count towards rate limit
        with pytest.raises(ValueError):
            failing_func()
            
        # Verify attempt was counted
        info = rate_limiter.get_limit_info(RateLimitType.REQUESTS, "global")
        assert info["remaining"] < rate_limiter.default_limits[RateLimitType.REQUESTS].limit