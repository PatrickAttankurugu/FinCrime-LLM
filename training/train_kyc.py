#!/usr/bin/env python3
"""
Fine-tune Mistral 7B for KYC assessment using QLoRA.

Usage:
    python train_kyc.py --data data/processed/kyc_dataset_alpaca --output models/kyc-mistral-7b
"""

import argparse
import sys

# Import main training function from train_sar
from train_sar import train

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Main entry point for KYC training."""
    parser = argparse.ArgumentParser(description="Fine-tune Mistral 7B for KYC assessment")

    parser.add_argument("--data", type=str, required=True, help="Path to KYC training data")
    parser.add_argument("--output", type=str, default="models/kyc-mistral-7b", help="Output directory")
    parser.add_argument("--model", type=str, default="mistralai/Mistral-7B-v0.1", help="Base model")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--learning-rate", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--lora-r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora-alpha", type=int, default=32, help="LoRA alpha")
    parser.add_argument("--no-wandb", action="store_true", help="Disable WandB")
    parser.add_argument("--push-to-hub", action="store_true", help="Push to HuggingFace Hub")
    parser.add_argument("--hub-model-id", type=str, help="Hub model ID")

    args = parser.parse_args()

    logger.info("Starting KYC model training...")

    # Use the same training function as SAR
    train(
        data_path=args.data,
        output_dir=args.output,
        model_name=args.model,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        use_wandb=not args.no_wandb,
        push_to_hub=args.push_to_hub,
        hub_model_id=args.hub_model_id,
    )


if __name__ == "__main__":
    main()
