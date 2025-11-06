"""Pydantic schemas for API requests and responses."""

from datetime import datetime
from typing import Dict, List, Optional

from pydantic import BaseModel, Field


# SAR Schemas
class SARRequest(BaseModel):
    """Request schema for SAR generation."""

    country: str = Field(..., description="Country code or name")
    subject_name: str = Field(..., description="Subject under investigation")
    institution: str = Field(..., description="Reporting institution")
    total_amount: float = Field(..., description="Total transaction amount")
    currency: str = Field(default="USD", description="Currency code")
    transactions: str = Field(..., description="Transaction details")
    summary: Optional[str] = Field(None, description="Transaction summary")


class SARResponse(BaseModel):
    """Response schema for SAR generation."""

    report_id: str
    sar_content: str
    generated_at: datetime
    model_version: str = "1.0.0"


# KYC Schemas
class KYCRequest(BaseModel):
    """Request schema for KYC assessment."""

    name: str = Field(..., description="Customer name")
    customer_type: str = Field(..., description="Individual or Entity")
    country: str = Field(..., description="Country")
    occupation: Optional[str] = Field(None, description="Occupation or business type")
    source_of_funds: Optional[str] = Field(None, description="Source of funds")
    expected_volume: Optional[str] = Field(None, description="Expected transaction volume")
    additional_info: Optional[str] = Field(None, description="Additional information")


class KYCResponse(BaseModel):
    """Response schema for KYC assessment."""

    assessment_id: str
    kyc_content: str
    risk_rating: Optional[str] = None
    generated_at: datetime
    model_version: str = "1.0.0"


# Transaction Analysis Schemas
class TransactionAnalysisRequest(BaseModel):
    """Request schema for transaction analysis."""

    transactions: str = Field(..., description="Transaction data to analyze")
    description: Optional[str] = Field(None, description="Context description")
    focus_areas: Optional[List[str]] = Field(None, description="Specific focus areas")


class TransactionAnalysisResponse(BaseModel):
    """Response schema for transaction analysis."""

    analysis_id: str
    analysis_content: str
    red_flags: Optional[List[str]] = None
    generated_at: datetime
    model_version: str = "1.0.0"


# Compliance Schemas
class ComplianceCheckRequest(BaseModel):
    """Request schema for compliance check."""

    check_type: str = Field(..., description="Type of compliance check")
    entity_name: str = Field(..., description="Entity to check")
    country: str = Field(..., description="Country")
    additional_data: Optional[Dict] = Field(None, description="Additional data")


class ComplianceCheckResponse(BaseModel):
    """Response schema for compliance check."""

    check_id: str
    result: str
    compliance_status: str
    generated_at: datetime


# Error Response
class ErrorResponse(BaseModel):
    """Error response schema."""

    error: str
    detail: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)
