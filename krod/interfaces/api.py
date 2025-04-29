"""KROD Api Server - Handles incoming requests and responses 
Provides HTTP interface to 

This module provides a RESTful API for the Krod application.

The API supports the following endpoints:
- /chat: Handles general chat requests
- /code: Handles code analysis requests
"""


from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Dict, Any, Optional, List
import logging
import os
from dotenv import load_dotenv
from contextlib import asynccontextmanager
import asyncio
from threading import Lock
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

# initialize fastapi app
@asynccontextmanager
def lifespan(app: FastAPI):
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
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:3001",
        "https://krod.kroskod.com",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
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
    session_id: Optional[str] = None


class QueryResponse(BaseModel):
    response: str
    session_id: Optional[str] = None
    domain: str
    security_level: str
    token_usage: int
    metadata: Dict[str, Any]

# Global engine instance
engine: Optional[KrodEngine] = None
engine_lock = Lock()

def get_engine():
    """get or initialize the krod engine"""
    global engine
    with engine_lock:
        if engine is None or not hasattr(engine, 'ready') or not engine.ready:
            config = load_config()
            engine = KrodEngine(config)
            try:
                engine.initialize()
                logger.info("Krod Engine initialized successfully")
            except Exception as e:
                logger.error(f"Engine initialization error: {str(e)}")
                raise RuntimeError(f"Failed to initialize Krod Engine: {str(e)}")
    return engine

# Add timeout constant
REQUEST_TIMEOUT = 60  # 60 seconds

# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    logger.error(f"HTTP error: {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail}
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled error: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )

# API endpoints
@app.post("/api/query", response_model=QueryResponse)
async def process_query(
    request: QueryRequest,
    _: str = Depends(verify_api_key),
    engine: KrodEngine = Depends(get_engine)
) -> Dict[str, Any]:
    try:
        result = await asyncio.wait_for(
            engine.process(
                request.query,
                request.session_id
            ),
            timeout=REQUEST_TIMEOUT
        )
        
        return {
            "response": result["response"],
            "session_id": result["session_id"],  # Return session ID to client
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
    except asyncio.TimeoutError:
        logger.error("Request timed out")
        raise HTTPException(
            status_code=504,
            detail="Request processing timed out. Please try again."
        )
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing query: {str(e)}"
        )

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
        timeout_keep_alive=65,  # Keep-alive timeout
        workers=4  # Number of worker processes
    )

if __name__ == "__main__":
    start()