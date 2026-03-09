from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime

class UserBase(BaseModel):
    email: str
    full_name: str
    role: str
    department: str
    age: Optional[int] = None
    gender: Optional[str] = None
    ethnicity: Optional[str] = None

class UserCreate(UserBase):
    password: str

class User(UserBase):
    id: int
    is_active: bool

    class Config:
        from_attributes = True

class Token(BaseModel):
    access_token: str
    token_type: str
    user: User

class TokenData(BaseModel):
    email: Optional[str] = None

class KPICreate(BaseModel):
    name: str
    category: str
    target_value: float
    current_value: float

class KPI(KPICreate):
    id: int
    owner_id: int

    class Config:
        from_attributes = True

class FeedbackCreate(BaseModel):
    target_user_id: int
    category: str
    qualitative_text: str
    competency_leadership: float
    competency_collaboration: float
    competency_execution: float
    sda_alignment: float

class Feedback(FeedbackCreate):
    id: int
    giver_user_id: int
    sentiment_score: float
    created_at: datetime

    class Config:
        from_attributes = True

class ExtensionSessionCreate(BaseModel):
    url: str
    title: str
    platform: Optional[str] = None
    duration_minutes: float
    interaction_count: int
    timestamp: str

class ExtensionSessionResponse(ExtensionSessionCreate):
    id: int
    user_id: int
    
    class Config:
        from_attributes = True

class PerformancePredictionBase(BaseModel):
    score: float
    confidence: float
    feature_kpi: float
    feature_sentiment: float
    feature_attendance: float
    feature_research: float
    shap_json: str
    lime_json: str

class PerformancePredictionResponse(PerformancePredictionBase):
    id: int
    user_id: int
    created_at: datetime

    class Config:
        from_attributes = True
