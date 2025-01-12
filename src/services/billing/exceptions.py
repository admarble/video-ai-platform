class BillingError(Exception):
    """Base exception for billing-related errors."""
    pass

class CustomerNotFoundError(BillingError):
    """Raised when a customer cannot be found."""
    pass

class PaymentError(BillingError):
    """Raised when there is an error processing a payment."""
    pass

class SubscriptionError(BillingError):
    """Raised when there is an error with a subscription."""
    pass

class UsageError(BillingError):
    """Raised when there is an error recording usage."""
    pass 