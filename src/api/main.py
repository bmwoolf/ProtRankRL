"""
FastAPI application for ProtRankRL protein ranking API.

Provides a simple REST API for ranking proteins by their predicted activity.
"""

import logging
from typing import List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .ranker import ranker

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="ProtRankRL API",
    description="Protein ranking API using reinforcement learning",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)


class RankRequest(BaseModel):
    """Request model for protein ranking."""
    uniprot_ids: List[str] = Field(
        ..., 
        description="List of UniProt IDs to rank",
        min_length=1,
        max_length=1000
    )


class RankingEntry(BaseModel):
    """Individual protein ranking result."""
    uniprot_id: str
    rank: int
    score: float
    confidence: float
    has_activity: bool = None
    experimental_data: dict = None


class RankResponse(BaseModel):
    """Response model for protein ranking."""
    rankings: List[RankingEntry]
    metadata: dict


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    database: dict
    model: dict
    timestamp: float


@app.on_event("startup")
async def startup_event():
    """Initialize the API on startup."""
    logger.info("Starting ProtRankRL API")
    logger.info("Loading protein database and model...")
    
    # The ranker is initialized when imported, so just log success
    logger.info("API ready to serve requests")


@app.get("/", response_model=dict)
async def root():
    """Root endpoint with API information."""
    return {
        "message": "ProtRankRL Protein Ranking API",
        "version": "1.0.0",
        "endpoints": {
            "rank": "/rank - Rank proteins by activity",
            "health": "/health - Health check",
            "docs": "/docs - API documentation"
        }
    }


@app.post("/rank", response_model=RankResponse)
async def rank_proteins(request: RankRequest):
    """
    Rank proteins by their predicted activity scores.
    
    Takes a list of UniProt IDs and returns them ranked by predicted activity.
    """
    try:
        logger.info(f"Ranking {len(request.uniprot_ids)} proteins")
        
        result = ranker.rank(request.uniprot_ids)
        
        logger.info(f"Ranking completed in {result['metadata']['processing_time']:.3f}s")
        
        return RankResponse(**result)
        
    except Exception as e:
        logger.error(f"Error ranking proteins: {e}")
        raise HTTPException(status_code=500, detail=f"Ranking failed: {str(e)}")


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    try:
        health_status = ranker.get_health_status()
        return HealthResponse(**health_status)
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")


@app.get("/stats")
async def get_stats():
    """Get database and model statistics."""
    try:
        db_stats = ranker.db.get_database_stats()
        model_info = ranker.predictor.get_model_info()
        
        return {
            "database": db_stats,
            "model": model_info,
            "available_proteins": len(ranker.db.get_all_proteins())
        }
    except Exception as e:
        logger.error(f"Stats request failed: {e}")
        raise HTTPException(status_code=500, detail=f"Stats request failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 