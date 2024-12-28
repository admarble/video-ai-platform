from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, func
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime

Base = declarative_base()

class Customer(Base):
    """Customer model for storing billing customer information."""
    __tablename__ = "customers"
    
    id = Column(Integer, primary_key=True)
    email = Column(String, unique=True, nullable=False)
    stripe_customer_id = Column(String, unique=True, nullable=False)
    subscription_status = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

class Usage(Base):
    """Usage model for tracking customer resource usage."""
    __tablename__ = "usage"
    
    id = Column(Integer, primary_key=True)
    customer_id = Column(Integer, ForeignKey("customers.id"), nullable=False)
    minutes_used = Column(Float, nullable=False)
    recorded_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    def __repr__(self):
        return f"<Usage(customer_id={self.customer_id}, minutes_used={self.minutes_used})>" 