"""Model loading utilities for inference."""

import logging
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)


def load_fincrime_model(model_path: str, device: str = "auto", load_in_4bit: bool = False):
    """
    Load FinCrime-LLM model for inference.

    Args:
        model_path: Path to model directory
        device: Device to load model on
        load_in_4bit: Use 4-bit quantization for inference

    Returns:
        Tuple of (model, tokenizer)
    """
    logger.info(f"Loading model from {model_path}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Load model
    if load_in_4bit:
        from transformers import BitsAndBytesConfig

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=bnb_config,
            device_map=device,
            trust_remote_code=True,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map=device,
            trust_remote_code=True,
        )

    model.eval()

    logger.info("Model loaded successfully")
    return model, tokenizer
