from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from typing import List
import json

from app.database import Base, engine, SessionLocal, get_db
from app.models.models import User, ExtensionSession, PerformancePrediction
from app.schemas.schemas import (
    ExtensionSessionCreate, ExtensionSessionResponse,
    PerformancePredictionResponse,
)
from app.ml.predictor import ml_service
from app.api.auth import router as auth_router, get_current_user

# Initialize DB tables
Base.metadata.create_all(bind=engine)

app = FastAPI(title="ADMS-R API", description="AI-Driven Multi-Source Rating Backend")

app.include_router(auth_router)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4300", "http://127.0.0.1:4300", "http://127.0.0.1:8081", "http://localhost:8081"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"message": "Welcome to the ADMS-R Backend API"}

@app.post("/api/extension/log-session", response_model=ExtensionSessionResponse)
def create_extension_session(
    session_data: ExtensionSessionCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):


    db_session = ExtensionSession(
        user_id=current_user.id,
        url=session_data.url,
        title=session_data.title,
        platform=session_data.platform,
        duration_minutes=session_data.duration_minutes,
        interaction_count=session_data.interaction_count,
        timestamp=session_data.timestamp,
    )
    db.add(db_session)
    db.commit()
    db.refresh(db_session)
    return db_session

@app.post("/api/ml/predict", response_model=PerformancePredictionResponse)
def predict_performance(
    kpi: float, sentiment: float, attendance: float, research: float,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    result = ml_service.predict_and_explain(kpi, sentiment, attendance, research)

    db_pred = PerformancePrediction(
        user_id=current_user.id,
        score=result["score"],
        confidence=result["confidence"],
        feature_kpi=kpi,
        feature_sentiment=sentiment,
        feature_attendance=attendance,
        feature_research=research,
        shap_json=result["shap_explanation"],
        lime_json=result["lime_explanation"],
    )
    db.add(db_pred)
    db.commit()
    db.refresh(db_pred)
    return db_pred

from app.schemas.schemas import FeedbackCreate, Feedback as FeedbackSchema
from app.models.models import Feedback
from app.ml.nlp import analyze_sentiment

@app.get("/api/employees")
def get_all_employees(db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    users = db.query(User).all()
    employees = []
    for u in users:
        # Get latest prediction metrics
        latest_pred = db.query(PerformancePrediction).filter(PerformancePrediction.user_id == u.id).order_by(PerformancePrediction.created_at.desc()).first()
        
        overallScore = latest_pred.score if latest_pred else 0.0
        kpiScore = latest_pred.feature_kpi if latest_pred else 0.0
        attendance = latest_pred.feature_attendance if latest_pred else 0.0
        sentimentScore = latest_pred.feature_sentiment if latest_pred else 0.0
        
        riskLevel = "high" if overallScore < 60 else "medium" if overallScore < 75 else "low"
        
        parts = u.full_name.split() if u.full_name else ["N", "A"]
        initials = "".join([p[0] for p in parts if p]).upper()[:2]
        
        employees.append({
            "id": u.id,
            "email": u.email,
            "name": u.full_name,
            "department": u.department or "Unknown",
            "role": u.role,
            "overallScore": round(overallScore),
            "kpiScore": round(kpiScore),
            "sentimentScore": round(sentimentScore, 2),
            "attendance": round(attendance),
            "initials": initials,
            "riskLevel": riskLevel,
            "gender": u.gender or "Unknown",
            "trend": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, round(overallScore)]
        })
    return employees

from pydantic import BaseModel as PydanticBaseModel

class AnalyzeRequest(PydanticBaseModel):
    text: str

@app.post("/api/feedback/analyze")
def analyze_feedback_text(
    req: AnalyzeRequest,
    current_user: User = Depends(get_current_user)
):
    """Analyze sentiment only — does NOT save to database."""
    result = analyze_sentiment(req.text)
    return result

@app.post("/api/feedback", response_model=FeedbackSchema)
def submit_feedback(
    feedback_in: FeedbackCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    # Role-based feedback type restrictions
    allowed = {
        "student": ["student"],
        "employee": ["peer", "subordinate"],
        "manager": ["supervisor"],
        "hr": ["peer", "supervisor", "subordinate", "student"],
    }
    user_allowed = allowed.get(current_user.role, [])
    if feedback_in.category not in user_allowed:
        raise HTTPException(status_code=400, detail=f"Your role ({current_user.role}) cannot submit '{feedback_in.category}' feedback. Allowed: {', '.join(user_allowed)}")
        
    # Analyze sentiment via Together API
    sentiment_result = analyze_sentiment(feedback_in.qualitative_text)
    
    db_feedback = Feedback(
        giver_user_id=current_user.id,
        target_user_id=feedback_in.target_user_id,
        category=feedback_in.category,
        qualitative_text=feedback_in.qualitative_text,
        sentiment_score=sentiment_result["score"],
        competency_leadership=feedback_in.competency_leadership,
        competency_collaboration=feedback_in.competency_collaboration,
        competency_execution=feedback_in.competency_execution,
        sda_alignment=feedback_in.sda_alignment
    )
    
    db.add(db_feedback)
    db.commit()
    db.refresh(db_feedback)
    return db_feedback
