from datetime import datetime
from sqlalchemy import Column, Integer, Float, String, DateTime
from .database import Base

class PredictionRecord(Base):
    __tablename__ = "prediction_history"
    id = Column(Integer, primary_key=True, index=True)
    machine_type = Column(String(1), nullable=False)
    failure_probability = Column(Float, nullable=False)
    predicted_failure = Column(Integer, nullable=False)
    risk_level = Column(String(20), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
