"""SAR (Suspicious Activity Report) generation router."""

import logging
import uuid
from datetime import datetime

from fastapi import APIRouter, HTTPException, Request
from slowapi import Limiter
from slowapi.util import get_remote_address

from api.models.schemas import SARRequest, SARResponse, ErrorResponse
from inference.generate import generate_sar

logger = logging.getLogger(__name__)
limiter = Limiter(key_func=get_remote_address)

router = APIRouter()


@router.post("/generate", response_model=SARResponse, responses={500: {"model": ErrorResponse}})
@limiter.limit("10/minute")
async def generate_sar_endpoint(request: Request, sar_request: SARRequest):
    """
    Generate a Suspicious Activity Report.

    Rate limit: 10 requests per minute per IP.
    """
    try:
        # Get model from cache
        model = request.app.MODEL_CACHE.get("model")
        tokenizer = request.app.MODEL_CACHE.get("tokenizer")

        if not model or not tokenizer:
            raise HTTPException(status_code=503, detail="Model not loaded")

        # Prepare transaction data
        transaction_data = {
            "country": sar_request.country,
            "subject_name": sar_request.subject_name,
            "institution": sar_request.institution,
            "total_amount": sar_request.total_amount,
            "currency": sar_request.currency,
            "transactions": sar_request.transactions,
            "summary": sar_request.summary or "",
        }

        # Generate SAR
        logger.info(f"Generating SAR for {sar_request.subject_name}")
        sar_content = generate_sar(model, tokenizer, transaction_data)

        # Create response
        report_id = f"SAR-{uuid.uuid4().hex[:12].upper()}"

        return SARResponse(
            report_id=report_id,
            sar_content=sar_content,
            generated_at=datetime.utcnow(),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating SAR: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def sar_health():
    """SAR endpoint health check."""
    return {"status": "healthy", "endpoint": "sar"}
