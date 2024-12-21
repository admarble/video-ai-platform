import smtplib
from retry import RetryConfig, RetryStrategy

# Predefined retry configurations for different scenarios
METRIC_COLLECTION_RETRY = RetryConfig(
    max_attempts=3,
    initial_delay=1.0,
    max_delay=5.0,
    strategy=RetryStrategy.EXPONENTIAL,
    exceptions=(ConnectionError, TimeoutError)
)

ALERT_NOTIFICATION_RETRY = RetryConfig(
    max_attempts=5,
    initial_delay=2.0,
    max_delay=30.0,
    strategy=RetryStrategy.EXPONENTIAL,
    exceptions=(ConnectionError, TimeoutError, smtplib.SMTPException)
)

MONITORING_SERVICE_RETRY = RetryConfig(
    max_attempts=3,
    initial_delay=5.0,
    max_delay=15.0,
    strategy=RetryStrategy.FIXED,
    exceptions=(ConnectionError, TimeoutError, RuntimeError)
) 