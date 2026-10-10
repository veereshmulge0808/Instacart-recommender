import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, status
from app.schemas import (
    RecommendationRequest, 
    RecommendationResponse, 
    HealthCheckResponse
)
from app.service import RecommenderEngine

# Global variable to hold engine instance
recommender: RecommenderEngine = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global recommender
    artifacts_path = os.getenv("ARTIFACTS_DIR", "./artifacts")
    print(f"Loading recommender artifacts from: {artifacts_path}")
    recommender = RecommenderEngine(artifacts_dir=artifacts_path)
    print("All artifacts successfully loaded into memory.")
    yield
    print("Shutting down application and freeing memory.")

app = FastAPI(
    title="Instacart Market Basket Recommendation API",
    description="Production-grade real-time inference microservice for repurchased grocery basket ranking.",
    version="1.0.0",
    lifespan=lifespan
)

@app.get(
    "/health", 
    response_model=HealthCheckResponse, 
    tags=["System"]
)
def health_check():
    """Service status and loaded model metadata."""
    if recommender is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, 
            detail="Model engine not loaded."
        )
    return {
        "status": "healthy",
        "model_version": recommender.model_version,
        "optimal_threshold": recommender.threshold
    }

@app.post(
    "/recommend", 
    response_model=RecommendationResponse, 
    tags=["Inference"]
)
def get_recommendations(request: RecommendationRequest):
    """Predict and rank products user is most likely to repurchase in next order."""
    if recommender is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, 
            detail="Model engine not ready."
        )
    
    result = recommender.predict(
        user_id=request.user_id,
        order_dow=request.order_dow,
        order_hour=request.order_hour,
        days_since_prior=request.days_since_prior,
        top_n=request.top_n
    )
    return result