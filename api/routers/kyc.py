"""KYC assessment router."""

import logging
import uuid
from datetime import datetime

from fastapi import APIRouter, HTTPException, Request
from slowapi import Limiter
from slowapi.util import get_remote_address

from api.models.schemas import KYCRequest, KYCResponse
from inference.generate import generate_kyc_assessment

logger = logging.getLogger(__name__)
limiter = Limiter(key_func=get_remote_address)
router = APIRouter()


@router.post("/assess", response_model=KYCResponse)
@limiter.limit("10/minute")
async def assess_kyc_endpoint(request: Request, kyc_request: KYCRequest):
    """Generate KYC risk assessment. Rate limit: 10/min."""
    try:
        model = request.app.MODEL_CACHE.get("model")
        tokenizer = request.app.MODEL_CACHE.get("tokenizer")

        if not model or not tokenizer:
            raise HTTPException(status_code=503, detail="Model not loaded")

        customer_data = kyc_request.dict()
        kyc_content = generate_kyc_assessment(model, tokenizer, customer_data)

        return KYCResponse(
            assessment_id=f"KYC-{uuid.uuid4().hex[:12].upper()}",
            kyc_content=kyc_content,
            generated_at=datetime.utcnow(),
        )
    except Exception as e:
        logger.error(f"Error in KYC assessment: {e}")
        raise HTTPException(status_code=500, detail=str(e))
