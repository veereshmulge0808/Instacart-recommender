from pydantic import BaseModel, Field
from typing import List, Optional

class RecommendationRequest(BaseModel):
    user_id: int = Field(..., description="Unique customer ID", example=1)
    order_dow: int = Field(default=1, ge=0, le=6, description="Day of week (0=Sunday, 6=Saturday)", example=1)
    order_hour: int = Field(default=10, ge=0, le=23, description="Hour of the day (0-23)", example=10)
    days_since_prior: float = Field(default=7.0, ge=0.0, description="Days elapsed since previous order", example=7.0)
    top_n: int = Field(default=10, ge=1, le=50, description="Max recommended items to return", example=5)

class ProductRecommendation(BaseModel):
    product_id: int
    product_name: str
    aisle: Optional[str] = None
    department: Optional[str] = None
    predicted_probability: float

class RecommendationResponse(BaseModel):
    user_id: int
    latency_ms: float
    threshold_used: float
    total_recommendations: int
    recommendations: List[ProductRecommendation]

class HealthCheckResponse(BaseModel):
    status: str
    model_version: str
    optimal_threshold: float