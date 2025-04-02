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



