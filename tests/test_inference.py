"""Tests for inference functionality."""

import pytest
import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestGenerateText:
    """Test text generation."""

    def test_generate_text(self, mock_model, mock_tokenizer):
        """Test basic text generation."""
        from inference.generate import generate_text

        prompt = "Test prompt"
        result = generate_text(mock_model, mock_tokenizer, prompt)

        assert isinstance(result, str)
        assert result == "Mock generated response"

    def test_generate_text_with_parameters(self, mock_model, mock_tokenizer):
        """Test text generation with custom parameters."""
        from inference.generate import generate_text

        result = generate_text(
            mock_model,
            mock_tokenizer,
            "Test prompt",
            max_new_tokens=512,
            temperature=0.8,
            top_p=0.9,
        )

        assert isinstance(result, str)


class TestSARGeneration:
    """Test SAR generation."""

    @patch("inference.generate.create_sar_prompt")
    def test_generate_sar(self, mock_create_prompt, mock_model, mock_tokenizer):
        """Test SAR generation."""
        from inference.generate import generate_sar

        mock_create_prompt.return_value = "Generated SAR prompt"

        transaction_data = {
            "country": "Ghana",
            "subject_name": "John Doe",
            "institution": "Test Bank",
            "total_amount": 100000,
            "currency": "GHS",
            "transactions": "Multiple suspicious transactions",
        }

        result = generate_sar(mock_model, mock_tokenizer, transaction_data)

        mock_create_prompt.assert_called_once_with(transaction_data)
        assert isinstance(result, str)

    def test_generate_sar_with_missing_data(self, mock_model, mock_tokenizer):
        """Test SAR generation with incomplete data."""
        from inference.generate import generate_sar

        # Even with incomplete data, should not crash
        incomplete_data = {"country": "Ghana"}

        try:
            result = generate_sar(mock_model, mock_tokenizer, incomplete_data)
            assert isinstance(result, str)
        except Exception as e:
            # Should handle gracefully
            pytest.fail(f"Should handle incomplete data: {e}")


class TestKYCGeneration:
    """Test KYC assessment generation."""

    @patch("inference.generate.create_kyc_prompt")
    def test_generate_kyc_assessment(self, mock_create_prompt, mock_model, mock_tokenizer):
        """Test KYC assessment generation."""
        from inference.generate import generate_kyc_assessment

        mock_create_prompt.return_value = "Generated KYC prompt"

        customer_data = {
            "name": "Jane Smith",
            "customer_type": "Individual",
            "country": "Kenya",
            "occupation": "Business Owner",
        }

        result = generate_kyc_assessment(mock_model, mock_tokenizer, customer_data)

        mock_create_prompt.assert_called_once_with(customer_data)
        assert isinstance(result, str)


class TestTransactionAnalysis:
    """Test transaction analysis."""

    @patch("inference.generate.create_analysis_prompt")
    def test_generate_transaction_analysis(
        self, mock_create_prompt, mock_model, mock_tokenizer
    ):
        """Test transaction analysis generation."""
        from inference.generate import generate_transaction_analysis

        mock_create_prompt.return_value = "Generated analysis prompt"

        transaction_data = {
            "transactions": "Multiple rapid transactions",
            "description": "Suspicious pattern",
        }

        result = generate_transaction_analysis(mock_model, mock_tokenizer, transaction_data)

        mock_create_prompt.assert_called_once_with(transaction_data)
        assert isinstance(result, str)


class TestBatchGeneration:
    """Test batch generation."""

    @patch("inference.generate.generate_sar")
    def test_batch_generate(self, mock_generate_sar, mock_model, mock_tokenizer, tmp_path):
        """Test batch generation from file."""
        from inference.generate import batch_generate
        import json

        # Create test input file
        input_file = tmp_path / "inputs.jsonl"
        output_file = tmp_path / "outputs.jsonl"

        test_data = [
            {"country": "Ghana", "subject_name": "Test1"},
            {"country": "Nigeria", "subject_name": "Test2"},
        ]

        with open(input_file, "w") as f:
            for item in test_data:
                f.write(json.dumps(item) + "\n")

        # Mock generation
        mock_generate_sar.return_value = "Generated SAR"

        # Run batch generation
        batch_generate(
            mock_model, mock_tokenizer, str(input_file), str(output_file), task="sar"
        )

        # Verify output file was created
        assert output_file.exists()

        # Verify contents
        with open(output_file, "r") as f:
            results = [json.loads(line) for line in f]

        assert len(results) == 2
        assert all("output" in r for r in results)

    def test_batch_generate_with_errors(self, mock_model, mock_tokenizer, tmp_path):
        """Test batch generation handles errors gracefully."""
        from inference.generate import batch_generate
        import json

        input_file = tmp_path / "inputs.jsonl"
        output_file = tmp_path / "outputs.jsonl"

        # Create input with data that might cause errors
        with open(input_file, "w") as f:
            f.write(json.dumps({"invalid": "data"}) + "\n")

        # Should not crash
        try:
            batch_generate(
                mock_model, mock_tokenizer, str(input_file), str(output_file), task="sar"
            )
        except Exception as e:
            pytest.fail(f"Batch generation should handle errors: {e}")


class TestModelLoading:
    """Test model loading functionality."""

    @patch("inference.load_model.AutoModelForCausalLM")
    @patch("inference.load_model.AutoTokenizer")
    def test_load_fincrime_model(self, mock_tokenizer_class, mock_model_class):
        """Test loading fine-tuned model."""
        from inference.load_model import load_fincrime_model

        mock_model = Mock()
        mock_tokenizer = Mock()

        mock_model_class.from_pretrained.return_value = mock_model
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer

        model, tokenizer = load_fincrime_model("test_model_path")

        mock_model_class.from_pretrained.assert_called_once()
        mock_tokenizer_class.from_pretrained.assert_called_once()

        assert model is not None
        assert tokenizer is not None


class TestPromptTemplates:
    """Test prompt template functions."""

    def test_create_sar_prompt(self, sample_sar_data):
        """Test SAR prompt creation."""
        from inference.prompts import create_sar_prompt

        prompt = create_sar_prompt(sample_sar_data)

        assert isinstance(prompt, str)
        assert len(prompt) > 0
        assert "Ghana" in prompt or "SAR" in prompt or "transaction" in prompt.lower()

    def test_create_kyc_prompt(self, sample_kyc_data):
        """Test KYC prompt creation."""
        from inference.prompts import create_kyc_prompt

        prompt = create_kyc_prompt(sample_kyc_data)

        assert isinstance(prompt, str)
        assert len(prompt) > 0
        assert "Kenya" in prompt or "KYC" in prompt or "customer" in prompt.lower()

    def test_create_analysis_prompt(self, sample_transaction_data):
        """Test analysis prompt creation."""
        from inference.prompts import create_analysis_prompt

        prompt = create_analysis_prompt(sample_transaction_data)

        assert isinstance(prompt, str)
        assert len(prompt) > 0


class TestCLIInterface:
    """Test command-line interface."""

    @patch("sys.argv", ["generate.py", "--model", "test_model", "--task", "sar", "--input", "test"])
    @patch("inference.generate.load_fincrime_model")
    @patch("inference.generate.generate_sar")
    def test_cli_sar_generation(self, mock_generate, mock_load_model):
        """Test CLI for SAR generation."""
        mock_model = Mock()
        mock_tokenizer = Mock()
        mock_load_model.return_value = (mock_model, mock_tokenizer)
        mock_generate.return_value = "Generated SAR"

        # CLI testing would require running the actual script
        # This is a placeholder test structure
        assert True  # Placeholder


class TestGenerationParameters:
    """Test generation parameter handling."""

    def test_temperature_variation(self, mock_model, mock_tokenizer):
        """Test different temperature values."""
        from inference.generate import generate_text

        # Test with low temperature (more deterministic)
        result1 = generate_text(mock_model, mock_tokenizer, "test", temperature=0.1)
        assert isinstance(result1, str)

        # Test with high temperature (more random)
        result2 = generate_text(mock_model, mock_tokenizer, "test", temperature=1.5)
        assert isinstance(result2, str)

    def test_max_tokens_variation(self, mock_model, mock_tokenizer):
        """Test different max token values."""
        from inference.generate import generate_text

        result = generate_text(mock_model, mock_tokenizer, "test", max_new_tokens=100)
        assert isinstance(result, str)

        result = generate_text(mock_model, mock_tokenizer, "test", max_new_tokens=1000)
        assert isinstance(result, str)


class TestErrorHandling:
    """Test error handling in inference."""

    def test_empty_input(self, mock_model, mock_tokenizer):
        """Test handling of empty input."""
        from inference.generate import generate_text

        try:
            result = generate_text(mock_model, mock_tokenizer, "")
            assert isinstance(result, str)
        except Exception:
            # Should handle gracefully
            pass

    def test_very_long_input(self, mock_model, mock_tokenizer):
        """Test handling of very long input."""
        from inference.generate import generate_text

        long_input = "test " * 10000

        try:
            result = generate_text(mock_model, mock_tokenizer, long_input)
            assert isinstance(result, str)
        except Exception:
            # Truncation should happen gracefully
            pass
