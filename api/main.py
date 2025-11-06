"""
FinCrime-LLM FastAPI Application.

Production-ready API for financial crime detection using fine-tuned Mistral 7B.

Endpoints:
- /api/v1/sar - Generate Suspicious Activity Reports
- /api/v1/kyc - Perform KYC assessments
- /api/v1/transaction - Analyze transactions
- /api/v1/compliance - Compliance checks
- /health - Health check
"""

import logging
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from api.routers import sar, kyc, transaction, compliance
from api.utils.logging import setup_logging

# Load environment variables
load_dotenv()

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)

# Global model cache
MODEL_CACHE = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup/shutdown."""
    # Startup
    logger.info("Starting FinCrime-LLM API...")

    # Load model
    model_path = os.getenv("MODEL_PATH", "models/final")
    logger.info(f"Loading model from {model_path}")

    try:
        from inference.load_model import load_fincrime_model

        model, tokenizer = load_fincrime_model(model_path, load_in_4bit=True)
        MODEL_CACHE["model"] = model
        MODEL_CACHE["tokenizer"] = tokenizer
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        logger.warning("API will start without model. Endpoints will return errors.")

    yield

    # Shutdown
    logger.info("Shutting down FinCrime-LLM API...")
    MODEL_CACHE.clear()


# Rate limiter
limiter = Limiter(key_func=get_remote_address)

# Create FastAPI app
app = FastAPI(
    title="FinCrime-LLM API",
    description="AI-powered financial crime detection for African markets",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# Add rate limiter
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "detail": str(exc)},
    )


# Health check
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    model_loaded = "model" in MODEL_CACHE and "tokenizer" in MODEL_CACHE
    return {
        "status": "healthy" if model_loaded else "degraded",
        "model_loaded": model_loaded,
        "version": "1.0.0",
    }


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "FinCrime-LLM API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
    }


# Include routers
app.include_router(
    sar.router,
    prefix="/api/v1/sar",
    tags=["SAR"],
)

app.include_router(
    kyc.router,
    prefix="/api/v1/kyc",
    tags=["KYC"],
)

app.include_router(
    transaction.router,
    prefix="/api/v1/transaction",
    tags=["Transaction Analysis"],
)

app.include_router(
    compliance.router,
    prefix="/api/v1/compliance",
    tags=["Compliance"],
)


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("API_PORT", 8000))
    host = os.getenv("API_HOST", "0.0.0.0")

    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=os.getenv("ENVIRONMENT", "production") == "development",
        log_level=os.getenv("LOG_LEVEL", "info").lower(),
    )
