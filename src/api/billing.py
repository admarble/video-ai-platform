from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import Dict, Optional
from ..services.billing import BillingService, BillingError, CustomerNotFoundError
from ..config.settings import Settings
from ..database import get_db
from pydantic import BaseModel, EmailStr

router = APIRouter(prefix="/billing", tags=["billing"])

# Request/Response Models
class CustomerCreate(BaseModel):
    email: EmailStr
    payment_method_id: Optional[str] = None

class SubscriptionCreate(BaseModel):
    customer_id: int
    price_id: str

class UsageRecord(BaseModel):
    customer_id: int
    minutes: float

# Endpoints
@router.post("/customers/", status_code=201)
async def create_customer(
    customer: CustomerCreate,
    db: Session = Depends(get_db),
    settings: Settings = Depends(Settings)
):
    """Create a new customer with optional payment method."""
    billing_service = BillingService(db, settings)
    try:
        return await billing_service.create_customer(
            customer.email,
            customer.payment_method_id
        )
    except BillingError as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/subscriptions/", status_code=201)
async def create_subscription(
    subscription: SubscriptionCreate,
    db: Session = Depends(get_db),
    settings: Settings = Depends(Settings)
):
    """Create a new subscription for a customer."""
    billing_service = BillingService(db, settings)
    try:
        return await billing_service.create_subscription(
            subscription.customer_id,
            subscription.price_id
        )
    except CustomerNotFoundError:
        raise HTTPException(status_code=404, detail="Customer not found")
    except BillingError as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/usage/", status_code=201)
async def record_usage(
    usage: UsageRecord,
    db: Session = Depends(get_db),
    settings: Settings = Depends(Settings)
):
    """Record usage for a customer."""
    billing_service = BillingService(db, settings)
    try:
        await billing_service.record_usage(
            usage.customer_id,
            usage.minutes
        )
        return {"status": "success"}
    except CustomerNotFoundError:
        raise HTTPException(status_code=404, detail="Customer not found")
    except BillingError as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/usage/{customer_id}")
async def get_customer_usage(
    customer_id: int,
    db: Session = Depends(get_db),
    settings: Settings = Depends(Settings)
):
    """Get total usage for a customer."""
    billing_service = BillingService(db, settings)
    try:
        return await billing_service.get_customer_usage(customer_id)
    except CustomerNotFoundError:
        raise HTTPException(status_code=404, detail="Customer not found")
    except BillingError as e:
        raise HTTPException(status_code=400, detail=str(e)) 