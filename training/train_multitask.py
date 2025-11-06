#!/usr/bin/env python3
"""
Multi-task fine-tuning for FinCrime-LLM.

Trains on combined SAR, KYC, and transaction analysis tasks.

Usage:
    python train_multitask.py --sar-data data/processed/sar_dataset_alpaca \
                               --kyc-data data/processed/kyc_dataset_alpaca \
                               --output models/fincrime-mistral-7b
"""

import argparse
import logging
from pathlib import Path

from datasets import DatasetDict, concatenate_datasets, load_from_disk
from train_sar import setup_lora, setup_model_and_tokenizer
from transformers import TrainingArguments
from trl import SFTTrainer
import wandb

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_and_combine_datasets(dataset_paths: list) -> DatasetDict:
    """Load and combine multiple datasets."""
    logger.info(f"Loading {len(dataset_paths)} datasets...")

    combined_train = []
    combined_val = []
    combined_test = []

    for path in dataset_paths:
        if not Path(path).exists():
            logger.warning(f"Dataset not found: {path}, skipping...")
            continue

        dataset = load_from_disk(path)
        combined_train.append(dataset["train"])

        if "validation" in dataset:
            combined_val.append(dataset["validation"])
        if "test" in dataset:
            combined_test.append(dataset["test"])

    # Concatenate datasets
    dataset_dict = {}
    if combined_train:
        dataset_dict["train"] = concatenate_datasets(combined_train)
    if combined_val:
        dataset_dict["validation"] = concatenate_datasets(combined_val)
    if combined_test:
        dataset_dict["test"] = concatenate_datasets(combined_test)

    logger.info(f"Combined dataset: {DatasetDict(dataset_dict)}")
    return DatasetDict(dataset_dict)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Multi-task training for FinCrime-LLM")

    parser.add_argument("--sar-data", type=str, help="Path to SAR dataset")
    parser.add_argument("--kyc-data", type=str, help="Path to KYC dataset")
    parser.add_argument("--transaction-data", type=str, help="Path to transaction dataset")
    parser.add_argument("--output", type=str, default="models/fincrime-mistral-7b", help="Output dir")
    parser.add_argument("--model", type=str, default="mistralai/Mistral-7B-v0.1", help="Base model")
    parser.add_argument("--epochs", type=int, default=3, help="Epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--no-wandb", action="store_true", help="Disable WandB")

    args = parser.parse_args()

    # Collect dataset paths
    dataset_paths = []
    if args.sar_data:
        dataset_paths.append(args.sar_data)
    if args.kyc_data:
        dataset_paths.append(args.kyc_data)
    if args.transaction_data:
        dataset_paths.append(args.transaction_data)

    if not dataset_paths:
        logger.error("No datasets specified!")
        return

    # Initialize WandB
    if not args.no_wandb:
        wandb.init(project="fincrime-llm", name="multitask-training")

    # Load and combine datasets
    dataset = load_and_combine_datasets(dataset_paths)

    # Setup model
    model, tokenizer = setup_model_and_tokenizer(args.model)
    model = setup_lora(model)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=2e-4,
        bf16=True,
        logging_steps=10,
        save_strategy="steps",
        save_steps=100,
        evaluation_strategy="steps",
        eval_steps=100,
        report_to="wandb" if not args.no_wandb else "none",
    )

    # Train
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset.get("validation"),
        tokenizer=tokenizer,
        max_seq_length=2048,
        formatting_func=lambda x: f"### Instruction:\n{x['instruction']}\n\n### Input:\n{x['input']}\n\n### Response:\n{x['output']}",
    )

    logger.info("Starting multi-task training...")
    trainer.train()
    trainer.save_model(f"{args.output}/final")

    logger.info("Training complete!")


if __name__ == "__main__":
    main()
