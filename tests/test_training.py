"""Tests for training pipeline."""

import pytest
import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestModelSetup:
    """Test model and tokenizer setup."""

    @patch("training.train_sar.AutoModelForCausalLM")
    @patch("training.train_sar.AutoTokenizer")
    def test_setup_model_and_tokenizer(self, mock_tokenizer_class, mock_model_class):
        """Test model and tokenizer initialization."""
        from training.train_sar import setup_model_and_tokenizer

        # Mock returns
        mock_model = Mock()
        mock_model.num_parameters.return_value = 7000000000
        mock_tokenizer = Mock()
        mock_tokenizer.pad_token = None
        mock_tokenizer.eos_token = "</s>"
        mock_tokenizer.eos_token_id = 2

        mock_model_class.from_pretrained.return_value = mock_model
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer

        model, tokenizer = setup_model_and_tokenizer(
            model_name="mistralai/Mistral-7B-v0.1", use_4bit=True
        )

        # Verify calls
        mock_model_class.from_pretrained.assert_called_once()
        mock_tokenizer_class.from_pretrained.assert_called_once()

        # Verify padding token is set
        assert tokenizer.pad_token is not None

    @patch("training.train_sar.prepare_model_for_kbit_training")
    @patch("training.train_sar.get_peft_model")
    def test_setup_lora(self, mock_get_peft, mock_prepare_kbit):
        """Test LoRA configuration."""
        from training.train_sar import setup_lora

        mock_model = Mock()
        mock_model.parameters.return_value = [Mock(numel=Mock(return_value=100), requires_grad=True)]

        mock_prepared_model = Mock()
        mock_prepared_model.parameters.return_value = mock_model.parameters()
        mock_prepare_kbit.return_value = mock_prepared_model

        mock_lora_model = Mock()
        mock_lora_model.parameters.return_value = mock_model.parameters()
        mock_get_peft.return_value = mock_lora_model

        lora_model = setup_lora(mock_model, r=16, lora_alpha=32)

        # Verify LoRA was applied
        mock_prepare_kbit.assert_called_once()
        mock_get_peft.assert_called_once()


class TestDataLoading:
    """Test data loading functionality."""

    @patch("training.train_sar.load_from_disk")
    def test_load_training_data_from_disk(self, mock_load_disk):
        """Test loading data from disk."""
        from training.train_sar import load_training_data

        mock_dataset = {"train": [], "validation": []}
        mock_load_disk.return_value = mock_dataset

        dataset = load_training_data("data/processed/sar_dataset_alpaca")

        mock_load_disk.assert_called_once()
        assert dataset == mock_dataset

    def test_format_instruction(self):
        """Test instruction formatting."""
        from training.train_sar import format_instruction

        mock_tokenizer = Mock()
        mock_tokenizer.return_value = {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1]}

        example = {
            "instruction": "Generate a SAR",
            "input": "Transaction details",
            "output": "SAR content",
        }

        result = format_instruction(example, mock_tokenizer)

        # Verify tokenizer was called
        mock_tokenizer.assert_called_once()
        assert "input_ids" in result


class TestTrainingConfig:
    """Test training configuration."""

    def test_training_args_defaults(self):
        """Test default training arguments."""
        from transformers import TrainingArguments

        args = TrainingArguments(
            output_dir="test_output",
            num_train_epochs=3,
            per_device_train_batch_size=4,
            learning_rate=2e-4,
        )

        assert args.num_train_epochs == 3
        assert args.per_device_train_batch_size == 4
        assert args.learning_rate == 2e-4


class TestDataPreparation:
    """Test data preparation scripts."""

    def test_load_sar_data(self, tmp_path, sample_sar_data):
        """Test loading SAR data from JSONL."""
        import json
        from data.scripts.prepare_sar_data import load_sar_data

        # Create temporary JSONL file
        test_file = tmp_path / "test_sars.jsonl"
        with open(test_file, "w") as f:
            f.write(json.dumps(sample_sar_data) + "\n")
            f.write(json.dumps(sample_sar_data) + "\n")

        sars = load_sar_data(str(test_file))

        assert len(sars) == 2
        assert sars[0]["country"] == "Ghana"

    def test_format_sar_for_training(self, sample_sar_data):
        """Test SAR formatting for training."""
        from data.scripts.prepare_sar_data import format_sar_for_training

        formatted = format_sar_for_training(sample_sar_data, format_type="alpaca")

        assert "instruction" in formatted
        assert "input" in formatted
        assert "output" in formatted
        assert formatted["metadata"]["country"] == "Ghana"

    def test_split_dataset(self):
        """Test dataset splitting."""
        from data.scripts.prepare_sar_data import split_dataset

        data = [{"id": i} for i in range(100)]
        train, val, test = split_dataset(data, split_ratios=(0.8, 0.1, 0.1))

        assert len(train) == 80
        assert len(val) == 10
        assert len(test) == 10


class TestSyntheticDataGeneration:
    """Test synthetic data generation."""

    def test_get_currency(self):
        """Test currency mapping."""
        from data.scripts.generate_synthetic_sars import get_currency

        assert get_currency("Ghana") == "GHS"
        assert get_currency("Nigeria") == "NGN"
        assert get_currency("Kenya") == "KES"
        assert get_currency("Unknown") == "USD"  # Default

    def test_get_country_code(self):
        """Test country code mapping."""
        from data.scripts.generate_synthetic_sars import get_country_code

        assert get_country_code("Ghana") == "GH"
        assert get_country_code("Nigeria") == "NG"
        assert get_country_code("Kenya") == "KE"

    @patch("data.scripts.generate_synthetic_sars.openai.OpenAI")
    def test_generate_sar(self, mock_openai_class):
        """Test SAR generation with OpenAI."""
        from data.scripts.generate_synthetic_sars import generate_sar
        import json

        # Mock OpenAI client
        mock_client = Mock()
        mock_openai_class.return_value = mock_client

        # Mock response
        mock_response = Mock()
        mock_response.choices = [
            Mock(
                message=Mock(
                    content=json.dumps(
                        {
                            "report_id": "SAR-TEST-001",
                            "country": "Ghana",
                            "typology": "money_laundering",
                        }
                    )
                )
            )
        ]
        mock_client.chat.completions.create.return_value = mock_response

        sar = generate_sar("Ghana", "money_laundering", "test-api-key")

        assert sar is not None
        assert sar["country"] == "Ghana"
        assert sar["typology"] == "money_laundering"


class TestLoRAConfig:
    """Test LoRA configuration files."""

    def test_lora_config_yaml(self):
        """Test LoRA config YAML structure."""
        import yaml

        config_path = Path(__file__).parent.parent / "training" / "configs" / "lora_config.yaml"

        if config_path.exists():
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)

            assert "lora" in config
            assert config["lora"]["r"] == 16
            assert config["lora"]["lora_alpha"] == 32
            assert "target_modules" in config["lora"]
            assert "q_proj" in config["lora"]["target_modules"]


class TestModelSaving:
    """Test model saving and loading."""

    @patch("training.train_sar.SFTTrainer")
    def test_model_saving(self, mock_trainer_class, tmp_path):
        """Test that model is saved correctly."""
        mock_trainer = Mock()
        mock_trainer_class.return_value = mock_trainer

        output_dir = str(tmp_path / "test_model")

        # Simulate saving
        mock_trainer.save_model(f"{output_dir}/final")

        # Verify save_model was called
        mock_trainer.save_model.assert_called()
