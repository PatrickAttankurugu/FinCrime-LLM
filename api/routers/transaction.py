"""Transaction analysis router."""

import logging
import uuid
from datetime import datetime

from fastapi import APIRouter, HTTPException, Request
from slowapi import Limiter
from slowapi.util import get_remote_address

from api.models.schemas import TransactionAnalysisRequest, TransactionAnalysisResponse
from inference.generate import generate_transaction_analysis

logger = logging.getLogger(__name__)
limiter = Limiter(key_func=get_remote_address)
router = APIRouter()


@router.post("/analyze", response_model=TransactionAnalysisResponse)
@limiter.limit("15/minute")
async def analyze_transaction_endpoint(request: Request, analysis_request: TransactionAnalysisRequest):
    """Analyze transactions for suspicious patterns."""
    try:
        model = request.app.MODEL_CACHE.get("model")
        tokenizer = request.app.MODEL_CACHE.get("tokenizer")

        if not model or not tokenizer:
            raise HTTPException(status_code=503, detail="Model not loaded")

        transaction_data = analysis_request.dict()
        analysis_content = generate_transaction_analysis(model, tokenizer, transaction_data)

        return TransactionAnalysisResponse(
            analysis_id=f"TXN-{uuid.uuid4().hex[:12].upper()}",
            analysis_content=analysis_content,
            generated_at=datetime.utcnow(),
        )
    except Exception as e:
        logger.error(f"Error analyzing transactions: {e}")
        raise HTTPException(status_code=500, detail=str(e))
