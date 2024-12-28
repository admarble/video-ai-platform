from fastapi import APIRouter, Request, HTTPException, Depends
import stripe
from sqlalchemy.orm import Session
from ..config.settings import Settings
from ..services.billing import BillingService
from ..database import get_db

router = APIRouter(prefix="/webhooks", tags=["webhooks"])

async def handle_paid_invoice(billing_service: BillingService, invoice):
    """Handle successful payment webhook."""
    customer_id = invoice.customer
    # Update customer status to active if needed
    await billing_service.update_subscription_status(customer_id, "active")

async def handle_failed_payment(billing_service: BillingService, invoice):
    """Handle failed payment webhook."""
    customer_id = invoice.customer
    # Update customer status to past_due
    await billing_service.update_subscription_status(customer_id, "past_due")

async def handle_subscription_updated(billing_service: BillingService, subscription):
    """Handle subscription update webhook."""
    customer_id = subscription.customer
    # Update customer subscription status
    await billing_service.update_subscription_status(customer_id, subscription.status)

@router.post("/stripe")
async def stripe_webhook(
    request: Request,
    db: Session = Depends(get_db),
    settings: Settings = Depends(Settings)
):
    """Handle Stripe webhook events."""
    # Get the webhook secret from settings
    webhook_secret = settings.STRIPE_WEBHOOK_SECRET.get_secret_value()
    
    # Get the webhook data
    payload = await request.body()
    sig_header = request.headers.get("stripe-signature")
    
    try:
        # Verify the webhook signature
        event = stripe.Webhook.construct_event(
            payload,
            sig_header,
            webhook_secret
        )
        
        # Initialize billing service
        billing_service = BillingService(db, settings)
        
        # Handle different event types
        if event.type == "invoice.paid":
            await handle_paid_invoice(billing_service, event.data.object)
        elif event.type == "invoice.payment_failed":
            await handle_failed_payment(billing_service, event.data.object)
        elif event.type == "customer.subscription.updated":
            await handle_subscription_updated(billing_service, event.data.object)
            
        return {"status": "success"}
        
    except stripe.error.SignatureVerificationError:
        raise HTTPException(status_code=400, detail="Invalid signature")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)) 