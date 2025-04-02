"""KROD Api Server - Handles incoming requests and responses 
Provides HTTP interface to 

This module provides a RESTful API for the Krod application.

The API supports the following endpoints:
- /chat: Handles general chat requests
- /code: Handles code analysis requests
"""


from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any, Optional
import logging
import os
from dotenv import load_dotenv

from ..core.engine import KrodEngine
from ..core.config import load_config

# load environment variables
load_dotenv()

# initialize logger
logging.basicConfig(level=os.getenv("KROD_LOG_LEVEL", "INFO"))
logger = logging.getLogger("Krod.api")

# initialize fastapi app
app = FastAPI(
    title ="Krod API",
    description="Krod an AI research assistant for researchers, engineers, and students",
    version="0.1.0"
)

# add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # this will be changed in production to only allow specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# pydantic models for request/response validations
class QueryRequest(BaseModel):
    query: str
    context_id: Optional[str] = None

class QueryResponse(BaseModel):
    response: str
    context_id: Optional[str] = None
    domain: str
    security_level: str
    token_usage: int
    metadata: Dict[str, Any]

# global engine instance
engine = Optional[KrodEngine] = None

def get_engine():
    """get or initialize the krod engine"""
    global engine
    if engine is None:
        config = load_config()
        engine = KrodEngine(config)
        logger.info("Krod Engine initialized")
    return engine

@app.post("/api/query", response_model=QueryResponse)
async def process_query(request: QueryRequest, engine: KrodEngine = Depends(get_engine)) -> Dict[str, Any]:
    """
    Process a query through Krod.
    
    Args:
        request: QueryRequest containing the user's query and optional context_id

    Returns:
        Dictionary containing the response and metadata
    """

    try:
        # process the query
        result = engine.process_query(request.query, request.context_id)

        # fromat the response
        response = {
            "response": result["response"],
            "context_id": result.get("context_id"),
            "domain": result.get("domain", "general"),
            "security_level": result.get("security_level", "low"),
            "token_usage": result.get("token_usage", 0),
            "metadata": {
                "capabilities": result.get("capabilities", []),
                "common_sense": result.get("common_sense", {}),
                "security_warnings": result.get("security_warnings", []),
                "security_recommendations": result.get("security_recommendations", [])
            }
        }
        return response
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing query: {str(e)}"
        )
    
# health check endpoint
@app.get("/api/health")
async def health_check() -> Dict[str, str]:
    """Check API health status"""
    return {"status": "healthy"}

@app.get("/api/token-usage")
async def get_token_usage(engine: KrodEngine = Depends(get_engine)) -> Dict[str, int]:
    """Get the total token usage for the current session"""
    return engine.get_token_usage()

def start():
    """
    Start the API server using uvicorn
    """
    import uvicorn
    uvicorn.run(
        "krod.api:app",
        host=os.getenv("KROD_HOST", "0.0.0.0"),
        port=int(os.getenv("KROD_PORT", 8000)),
        reload=os.getenv("KROD_DEBUG", "False").lower() == "true",
    )


# run the API
if __name__ == "__main__":
    start()


