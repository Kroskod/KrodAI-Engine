"""KROD Api Server - Handles incoming requests and responses 
Provides HTTP interface to 

This module provides a RESTful API for the Krod application.

The API supports the following endpoints:
- /chat: Handles general chat requests
- /code: Handles code analysis requests
"""


from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Dict, Any, Optional, List
import logging
import os
from dotenv import load_dotenv
from contextlib import asynccontextmanager

import json

from krod.core.engine import KrodEngine
from krod.core.config import load_config

# load environment variables
load_dotenv()

# Environment settings
ENVIRONMENT = os.getenv("ENVIRONMENT", "development")
KROD_API_KEY = os.getenv("KROD_API_KEY")
FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:3000")

if not KROD_API_KEY:
    raise RuntimeError("KROD_API_KEY environment variable is not set")


log_dir = "logs"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# Configure logging
logging.basicConfig(
    level=os.getenv("KROD_LOG_LEVEL", "INFO"),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(log_dir, 'api.log')) if ENVIRONMENT == "production" else logging.NullHandler()
    ]
)
logger = logging.getLogger("Krod.api")




# # Pretty JSON response
# class PrettyJSONResponse(JSONResponse):
#     def render(self, content: Any) -> bytes:
#         return json.dumps(
#             content,
#             ensure_ascii=False,
#             allow_nan=False,
#             indent=2,
#             separators=(", ", ": "),
#         ).encode("utf-8")


# initialize fastapi app
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info(f"Starting Krod API server in {ENVIRONMENT} mode")
    yield
    # Shutdown
    logger.info("Shutting down Krod API server")

app = FastAPI(
    title="Krod API",
    description="Krod AI research assistant API",
    version="0.1.0",
    lifespan=lifespan
)

# add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[FRONTEND_URL] if ENVIRONMENT == "production" else ["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["X-API-Key", "Content-Type"],
)

# API key security
api_key_header = APIKeyHeader(name="X-API-Key")

async def verify_api_key(api_key: str = Depends(api_key_header)):
    if api_key != KROD_API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API key")
    return api_key

# Request/Response models
class QueryRequest(BaseModel):
    query: str
    context_id: Optional[str] = None
    conversation_history: Optional[List[Dict[str, str]]] = None


class QueryResponse(BaseModel):
    response: str
    context_id: Optional[str] = None
    domain: str
    security_level: str
    token_usage: int
    metadata: Dict[str, Any]

# Global engine instance
engine: Optional[KrodEngine] = None

def get_engine():
    """get or initialize the krod engine"""
    global engine
    if engine is None:
        config = load_config()
        engine = KrodEngine(config)
        logger.info("Krod Engine initialized")
    return engine

# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unhandled error: {str(exc)}")
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})

# API endpoints
@app.post("/api/query", response_model=QueryResponse)
async def process_query(
    request: QueryRequest,
    _: str = Depends(verify_api_key),
    engine: KrodEngine = Depends(get_engine)
) -> Dict[str, Any]:
    try:
        result = engine.process(
            request.query,
            request.context_id,
            request.conversation_history
        )
        return {
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
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

@app.get("/api/health")
async def health_check() -> Dict[str, str]:
    return {"status": "healthy"}

@app.get("/api/token-usage")
async def get_token_usage(
    api_key: str = Depends(verify_api_key),
    engine: KrodEngine = Depends(get_engine)
) -> Dict[str, int]:
    return engine.get_token_usage()

@app.get("/")
async def root():
    return {"message": "Krod API is running"}

def start():
    """
    Start the API server using uvicorn
    """
    import uvicorn
    uvicorn.run(
        "krod.api:app",
        host=os.getenv("KROD_HOST", "0.0.0.0"),
        port=int(os.getenv("KROD_PORT", 8000)),
        reload=ENVIRONMENT == "development",
    )

if __name__ == "__main__":
    start()