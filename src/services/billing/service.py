from typing import Optional, Dict, Any
import stripe
from datetime import datetime
from sqlalchemy.orm import Session
from .exceptions import BillingError, CustomerNotFoundError, PaymentError, SubscriptionError, UsageError
from ...models.billing import Customer, Usage
from ...config.settings import Settings

class BillingService:
    def __init__(self, db: Session, settings: Settings):
        """Initialize the billing service with database session and settings."""
        stripe.api_key = settings.STRIPE_SECRET_KEY
        self.db = db
        self.settings = settings
    
    async def create_customer(
        self,
        email: str,
        payment_method_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Create a new customer with optional payment method."""
        try:
            # Create Stripe customer
            stripe_customer = stripe.Customer.create(
                email=email,
                payment_method=payment_method_id,
                invoice_settings={
                    'default_payment_method': payment_method_id
                } if payment_method_id else None
            )
            
            # Create local customer record
            customer = Customer(
                email=email,
                stripe_customer_id=stripe_customer.id,
                subscription_status='active'
            )
            self.db.add(customer)
            self.db.commit()
            
            return {
                "customer_id": customer.id,
                "stripe_customer_id": stripe_customer.id,
                "email": email,
                "status": "active"
            }
            
        except stripe.error.StripeError as e:
            self.db.rollback()
            raise PaymentError(f"Failed to create customer: {str(e)}")
        except Exception as e:
            self.db.rollback()
            raise BillingError(f"Unexpected error creating customer: {str(e)}")
    
    async def create_subscription(
        self,
        customer_id: int,
        price_id: str
    ) -> Dict[str, Any]:
        """Create a new subscription for customer."""
        customer = self.db.query(Customer).get(customer_id)
        if not customer:
            raise CustomerNotFoundError(f"Customer {customer_id} not found")
            
        try:
            subscription = stripe.Subscription.create(
                customer=customer.stripe_customer_id,
                items=[{"price": price_id}],
                expand=['latest_invoice.payment_intent']
            )
            
            # Update customer subscription status
            customer.subscription_status = subscription.status
            self.db.commit()
            
            return {
                "subscription_id": subscription.id,
                "client_secret": subscription.latest_invoice.payment_intent.client_secret,
                "status": subscription.status
            }
            
        except stripe.error.StripeError as e:
            self.db.rollback()
            raise SubscriptionError(f"Failed to create subscription: {str(e)}")
        except Exception as e:
            self.db.rollback()
            raise BillingError(f"Unexpected error creating subscription: {str(e)}")
    
    async def record_usage(
        self,
        customer_id: int,
        minutes: float
    ) -> None:
        """Record usage for a customer."""
        customer = self.db.query(Customer).get(customer_id)
        if not customer:
            raise CustomerNotFoundError(f"Customer {customer_id} not found")
            
        try:
            usage = Usage(
                customer_id=customer_id,
                minutes_used=minutes,
                recorded_at=datetime.utcnow()
            )
            self.db.add(usage)
            self.db.commit()
            
        except Exception as e:
            self.db.rollback()
            raise UsageError(f"Failed to record usage: {str(e)}")
            
    async def get_customer_usage(
        self,
        customer_id: int
    ) -> Dict[str, float]:
        """Get total usage for a customer."""
        customer = self.db.query(Customer).get(customer_id)
        if not customer:
            raise CustomerNotFoundError(f"Customer {customer_id} not found")
            
        try:
            total_usage = self.db.query(Usage).filter(
                Usage.customer_id == customer_id
            ).with_entities(
                func.sum(Usage.minutes_used)
            ).scalar() or 0.0
            
            return {
                "total_minutes_used": total_usage
            }
            
        except Exception as e:
            raise UsageError(f"Failed to get usage: {str(e)}")
            
    async def update_subscription_status(
        self,
        customer_id: int,
        status: str
    ) -> None:
        """Update customer subscription status."""
        customer = self.db.query(Customer).get(customer_id)
        if not customer:
            raise CustomerNotFoundError(f"Customer {customer_id} not found")
            
        try:
            customer.subscription_status = status
            self.db.commit()
            
        except Exception as e:
            self.db.rollback()
            raise SubscriptionError(f"Failed to update subscription status: {str(e)}") 