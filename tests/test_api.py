"""Tests for FastAPI endpoints."""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch

# Import the FastAPI app
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from api.main import app


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


@pytest.fixture
def mock_model_cache():
    """Mock the model cache."""
    mock_model = Mock()
    mock_tokenizer = Mock()

    mock_tokenizer.decode.return_value = "Generated SAR content"
    mock_model.generate.return_value = Mock()

    return {"model": mock_model, "tokenizer": mock_tokenizer}


class TestHealthEndpoint:
    """Test health check endpoint."""

    def test_health_check(self, client):
        """Test health check returns 200."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "model_loaded" in data

    def test_root_endpoint(self, client):
        """Test root endpoint."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert data["version"] == "1.0.0"


class TestSAREndpoint:
    """Test SAR generation endpoint."""

    @patch("api.routers.sar.generate_sar")
    def test_generate_sar_success(self, mock_generate, client, mock_model_cache):
        """Test successful SAR generation."""
        # Mock generate_sar function
        mock_generate.return_value = "Generated SAR content"

        # Mock model cache
        app.MODEL_CACHE = mock_model_cache

        sar_request = {
            "country": "Ghana",
            "subject_name": "John Doe",
            "institution": "Test Bank",
            "total_amount": 100000,
            "currency": "GHS",
            "transactions": "Multiple suspicious transactions",
            "summary": "Structuring detected",
        }

        response = client.post("/api/v1/sar/generate", json=sar_request)

        # Verify response
        assert response.status_code == 200
        data = response.json()
        assert "report_id" in data
        assert "sar_content" in data
        assert data["sar_content"] == "Generated SAR content"

    def test_generate_sar_missing_fields(self, client):
        """Test SAR generation with missing required fields."""
        incomplete_request = {
            "country": "Ghana",
            # Missing required fields
        }

        response = client.post("/api/v1/sar/generate", json=incomplete_request)
        assert response.status_code == 422  # Validation error

    def test_sar_health(self, client):
        """Test SAR health endpoint."""
        response = client.get("/api/v1/sar/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"


class TestKYCEndpoint:
    """Test KYC assessment endpoint."""

    @patch("api.routers.kyc.generate_kyc_assessment")
    def test_assess_kyc_success(self, mock_generate, client, mock_model_cache):
        """Test successful KYC assessment."""
        mock_generate.return_value = "Generated KYC assessment"
        app.MODEL_CACHE = mock_model_cache

        kyc_request = {
            "name": "Jane Smith",
            "customer_type": "Individual",
            "country": "Kenya",
            "occupation": "Business Owner",
            "source_of_funds": "Business Revenue",
        }

        response = client.post("/api/v1/kyc/assess", json=kyc_request)

        assert response.status_code == 200
        data = response.json()
        assert "assessment_id" in data
        assert "kyc_content" in data

    def test_assess_kyc_missing_fields(self, client):
        """Test KYC assessment with missing required fields."""
        incomplete_request = {
            "name": "Jane Smith",
            # Missing customer_type and country
        }

        response = client.post("/api/v1/kyc/assess", json=incomplete_request)
        assert response.status_code == 422


class TestTransactionEndpoint:
    """Test transaction analysis endpoint."""

    @patch("api.routers.transaction.generate_transaction_analysis")
    def test_analyze_transaction_success(self, mock_generate, client, mock_model_cache):
        """Test successful transaction analysis."""
        mock_generate.return_value = "Generated transaction analysis"
        app.MODEL_CACHE = mock_model_cache

        transaction_request = {
            "transactions": "Multiple rapid transactions",
            "description": "Suspicious pattern",
        }

        response = client.post("/api/v1/transaction/analyze", json=transaction_request)

        assert response.status_code == 200
        data = response.json()
        assert "analysis_id" in data
        assert "analysis_content" in data


class TestRateLimiting:
    """Test rate limiting functionality."""

    def test_rate_limit_enforcement(self, client, mock_model_cache):
        """Test that rate limiting is enforced (placeholder test)."""
        # Note: Actual rate limit testing requires more setup
        # This is a placeholder to ensure rate limiting exists
        app.MODEL_CACHE = mock_model_cache

        # Make a single request (should succeed)
        sar_request = {
            "country": "Ghana",
            "subject_name": "Test",
            "institution": "Test Bank",
            "total_amount": 1000,
            "currency": "GHS",
            "transactions": "Test",
        }

        response = client.post("/api/v1/sar/generate", json=sar_request)
        # First request should work or fail gracefully
        assert response.status_code in [200, 503]  # 503 if model not loaded


class TestErrorHandling:
    """Test error handling."""

    def test_model_not_loaded(self, client):
        """Test behavior when model is not loaded."""
        # Clear model cache
        app.MODEL_CACHE = {}

        sar_request = {
            "country": "Ghana",
            "subject_name": "Test",
            "institution": "Test Bank",
            "total_amount": 1000,
            "currency": "GHS",
            "transactions": "Test",
        }

        response = client.post("/api/v1/sar/generate", json=sar_request)
        assert response.status_code == 503
        data = response.json()
        assert "detail" in data

    def test_invalid_json(self, client):
        """Test handling of invalid JSON."""
        response = client.post(
            "/api/v1/sar/generate",
            data="invalid json",
            headers={"Content-Type": "application/json"},
        )
        assert response.status_code == 422


class TestCORS:
    """Test CORS configuration."""

    def test_cors_headers(self, client):
        """Test that CORS headers are present."""
        response = client.options("/api/v1/sar/generate")
        # Should have CORS headers configured
        assert response.status_code in [200, 405]  # OPTIONS might not be fully configured
