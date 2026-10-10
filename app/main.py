import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, status
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
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

# Mount static folder
static_dir = os.path.join(os.path.dirname(__file__), "static")
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

@app.get("/", include_in_schema=False)
def serve_frontend():
    """Serves the interactive web UI dashboard."""
    index_file = os.path.join(static_dir, "index.html")
    if os.path.exists(index_file):
        return FileResponse(index_file)
    return {"message": "API is online. Visit /docs for documentation."}

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