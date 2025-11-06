"""
Training utilities for FinCrime-LLM.

Common functions for model training, data loading, and monitoring.
"""

import logging
from typing import Dict, Optional

import torch
import yaml
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)


def load_config(config_path: str) -> Dict:
    """Load YAML configuration file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def count_parameters(model) -> Dict[str, int]:
    """Count model parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total": total,
        "trainable": trainable,
        "trainable_percent": (trainable / total) * 100 if total > 0 else 0,
    }


def print_model_info(model):
    """Print detailed model information."""
    params = count_parameters(model)
    logger.info(f"Total parameters: {params['total']:,}")
    logger.info(f"Trainable parameters: {params['trainable']:,}")
    logger.info(f"Trainable: {params['trainable_percent']:.2f}%")


def merge_and_save_model(
    base_model_name: str,
    adapter_path: str,
    output_path: str,
    push_to_hub: bool = False,
    hub_model_id: Optional[str] = None,
):
    """
    Merge LoRA adapters with base model and save.

    Args:
        base_model_name: Base model identifier
        adapter_path: Path to LoRA adapters
        output_path: Output path for merged model
        push_to_hub: Push to HuggingFace Hub
        hub_model_id: Hub model ID
    """
    logger.info("Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    logger.info("Loading LoRA adapters...")
    model = PeftModel.from_pretrained(base_model, adapter_path)

    logger.info("Merging adapters...")
    model = model.merge_and_unload()

    logger.info(f"Saving merged model to {output_path}")
    model.save_pretrained(output_path)

    # Save tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    tokenizer.save_pretrained(output_path)

    if push_to_hub and hub_model_id:
        logger.info(f"Pushing to hub: {hub_model_id}")
        model.push_to_hub(hub_model_id)
        tokenizer.push_to_hub(hub_model_id)

    logger.info("Merge complete!")
