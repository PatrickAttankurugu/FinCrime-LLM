"""Pytest configuration and fixtures for FinCrime-LLM tests."""

import os
import sys
from pathlib import Path

import pytest

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture(scope="session")
def test_data_dir():
    """Return path to test data directory."""
    return PROJECT_ROOT / "tests" / "test_data"


@pytest.fixture(scope="session")
def sample_sar_data():
    """Sample SAR data for testing."""
    return {
        "report_id": "SAR-TEST-001",
        "country": "Ghana",
        "typology": "money_laundering",
        "currency": "GHS",
        "reporting_institution": "Test Bank Ltd",
        "subject_name": "John Doe",
        "subject_type": "individual",
        "transaction_summary": "Multiple suspicious transactions",
        "detailed_narrative": "Customer made multiple cash deposits...",
        "transaction_details": [
            {
                "date": "2024-01-15",
                "amount": 50000,
                "counterparty": "Shell Company A",
            },
            {
                "date": "2024-01-20",
                "amount": 50000,
                "counterparty": "Offshore Account B",
            },
        ],
        "red_flags": [
            "Rapid in-and-out transactions",
            "Structuring behavior",
            "Unusual business activity",
        ],
        "cdd_findings": "Customer unable to explain source of funds",
        "regulatory_references": "Ghana AML Act 2020",
        "recommendation": "Report to FIU for investigation",
        "report_date": "2024-01-25",
        "total_amount": 100000,
    }


@pytest.fixture(scope="session")
def sample_kyc_data():
    """Sample KYC data for testing."""
    return {
        "name": "Jane Smith",
        "customer_type": "Individual",
        "country": "Kenya",
        "occupation": "Import/Export Business",
        "source_of_funds": "Business Revenue",
        "expected_volume": "100,000 - 500,000 KES",
        "additional_info": "Customer deals with international suppliers",
    }


@pytest.fixture(scope="session")
def sample_transaction_data():
    """Sample transaction data for testing."""
    return {
        "transactions": (
            "Account: 123456789\n"
            "2024-01-10: Deposit 10,000 USD\n"
            "2024-01-11: Withdrawal 9,500 USD\n"
            "2024-01-12: Deposit 15,000 USD\n"
            "2024-01-13: Withdrawal 14,500 USD"
        ),
        "description": "Rapid in-and-out transactions",
    }


@pytest.fixture
def mock_model():
    """Mock model for testing without loading actual model."""

    class MockModel:
        def __init__(self):
            self.device = "cpu"

        def generate(self, **kwargs):
            # Return mock tensor
            import torch

            return torch.tensor([[1, 2, 3, 4, 5]])

    return MockModel()


@pytest.fixture
def mock_tokenizer():
    """Mock tokenizer for testing."""

    class MockTokenizer:
        def __init__(self):
            self.pad_token_id = 0
            self.eos_token_id = 2

        def __call__(self, text, **kwargs):
            # Return mock tokenized output
            import torch

            return {
                "input_ids": torch.tensor([[1, 2, 3]]),
                "attention_mask": torch.tensor([[1, 1, 1]]),
            }

        def decode(self, tokens, **kwargs):
            return "Mock generated response"

    return MockTokenizer()


@pytest.fixture(scope="session", autouse=True)
def setup_test_env():
    """Set up test environment variables."""
    os.environ["TESTING"] = "true"
    os.environ["LOG_LEVEL"] = "ERROR"
    yield
    os.environ.pop("TESTING", None)
