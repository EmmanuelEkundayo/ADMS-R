from sqlalchemy import Boolean, Column, ForeignKey, Integer, String, Float, DateTime
from sqlalchemy.orm import relationship
import datetime
from app.database import Base

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True)
    full_name = Column(String)
    hashed_password = Column(String)
    role = Column(String) # employee, manager, hr_admin
    department = Column(String)
    age = Column(Integer, nullable=True)
    gender = Column(String, nullable=True)
    ethnicity = Column(String, nullable=True)
    is_active = Column(Boolean, default=True)

    kpis = relationship("KPI", back_populates="owner")
    received_feedback = relationship("Feedback", foreign_keys="[Feedback.target_user_id]", back_populates="target_user")
    given_feedback = relationship("Feedback", foreign_keys="[Feedback.giver_user_id]", back_populates="giver_user")
    sessions = relationship("ExtensionSession", back_populates="user")
    predictions = relationship("PerformancePrediction", back_populates="user")

class KPI(Base):
    __tablename__ = "kpis"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    category = Column(String)
    target_value = Column(Float)
    current_value = Column(Float)
    owner_id = Column(Integer, ForeignKey("users.id"))

    owner = relationship("User", back_populates="kpis")

class Feedback(Base):
    __tablename__ = "feedback"

    id = Column(Integer, primary_key=True, index=True)
    giver_user_id = Column(Integer, ForeignKey("users.id"))
    target_user_id = Column(Integer, ForeignKey("users.id"))
    category = Column(String) # peer, manager, student
    qualitative_text = Column(String)
    sentiment_score = Column(Float) # calculated by NLP layer
    competency_leadership = Column(Float)
    competency_collaboration = Column(Float)
    competency_execution = Column(Float)
    sda_alignment = Column(Float)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    giver_user = relationship("User", foreign_keys=[giver_user_id], back_populates="given_feedback")
    target_user = relationship("User", foreign_keys=[target_user_id], back_populates="received_feedback")

class ExtensionSession(Base):
    __tablename__ = "extension_sessions"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    url = Column(String)
    title = Column(String)
    platform = Column(String, nullable=True)
    duration_minutes = Column(Float)
    interaction_count = Column(Integer)
    timestamp = Column(DateTime)
    
    user = relationship("User", back_populates="sessions")

class PerformancePrediction(Base):
    __tablename__ = "predictions"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    score = Column(Float)
    confidence = Column(Float)
    feature_kpi = Column(Float)
    feature_sentiment = Column(Float)
    feature_attendance = Column(Float)
    feature_research = Column(Float)
    shap_json = Column(String) # Serialized SHAP explainations
    lime_json = Column(String) # Serialized LIME explanations
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    user = relationship("User", back_populates="predictions")
