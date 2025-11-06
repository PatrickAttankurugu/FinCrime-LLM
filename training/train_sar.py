#!/usr/bin/env python3
"""
Fine-tune Mistral 7B for SAR generation using QLoRA.

This script implements efficient fine-tuning using:
- 4-bit quantization (bitsandbytes)
- LoRA adapters (PEFT)
- WandB logging
- HuggingFace Hub integration

Usage:
    python train_sar.py --config training/configs/lora_config.yaml
    python train_sar.py --data data/processed/sar_dataset_alpaca --output models/sar-mistral-7b
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Optional

import torch
import wandb
from datasets import load_dataset, load_from_disk
from dotenv import load_dotenv
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import SFTTrainer

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("training.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)


def setup_model_and_tokenizer(
    model_name: str = "mistralai/Mistral-7B-v0.1",
    use_4bit: bool = True,
) -> tuple:
    """
    Load and configure model and tokenizer with quantization.

    Args:
        model_name: HuggingFace model identifier
        use_4bit: Use 4-bit quantization

    Returns:
        Tuple of (model, tokenizer)
    """
    logger.info(f"Loading model: {model_name}")

    # Quantization config
    if use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    else:
        bnb_config = None

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        use_cache=False,  # Required for gradient checkpointing
    )

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
    )

    # Set padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model.config.pad_token_id = tokenizer.pad_token_id

    logger.info(f"Model loaded. Parameters: {model.num_parameters():,}")

    return model, tokenizer


def setup_lora(
    model,
    r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    target_modules: Optional[list] = None,
) -> any:
    """
    Configure LoRA adapters.

    Args:
        model: Base model
        r: LoRA rank
        lora_alpha: LoRA alpha parameter
        lora_dropout: Dropout rate
        target_modules: Modules to apply LoRA to

    Returns:
        PEFT model with LoRA
    """
    logger.info("Configuring LoRA...")

    if target_modules is None:
        # Default target modules for Mistral
        target_modules = [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]

    # Prepare model for k-bit training
    model = prepare_model_for_kbit_training(model)

    # LoRA configuration
    lora_config = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # Apply LoRA
    model = get_peft_model(model, lora_config)

    # Print trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Trainable parameters: {trainable_params:,} ({trainable_params/total_params*100:.2f}%)")

    return model


def format_instruction(example: dict, tokenizer) -> dict:
    """Format instruction for training."""
    # Create instruction prompt
    if "instruction" in example:
        # Alpaca format
        prompt = f"""### Instruction:
{example['instruction']}

### Input:
{example['input']}

### Response:
{example['output']}"""
    else:
        # Handle other formats
        prompt = example.get("text", "")

    # Tokenize
    return tokenizer(
        prompt,
        truncation=True,
        max_length=2048,
        padding="max_length",
    )


def load_training_data(data_path: str):
    """
    Load training dataset.

    Args:
        data_path: Path to dataset (directory or HF dataset name)

    Returns:
        Dataset object
    """
    logger.info(f"Loading training data from {data_path}")

    data_path_obj = Path(data_path)

    if data_path_obj.exists() and data_path_obj.is_dir():
        # Load from disk
        dataset = load_from_disk(data_path)
        logger.info(f"Loaded dataset from disk: {dataset}")
    else:
        # Try loading from HuggingFace Hub
        try:
            dataset = load_dataset(data_path)
            logger.info(f"Loaded dataset from HuggingFace Hub: {dataset}")
        except Exception as e:
            logger.error(f"Failed to load dataset: {e}")
            sys.exit(1)

    return dataset


def train(
    data_path: str,
    output_dir: str,
    model_name: str = "mistralai/Mistral-7B-v0.1",
    num_epochs: int = 3,
    batch_size: int = 4,
    gradient_accumulation_steps: int = 4,
    learning_rate: float = 2e-4,
    max_seq_length: int = 2048,
    lora_r: int = 16,
    lora_alpha: int = 32,
    use_wandb: bool = True,
    push_to_hub: bool = False,
    hub_model_id: Optional[str] = None,
) -> None:
    """
    Main training function.

    Args:
        data_path: Path to training data
        output_dir: Output directory for checkpoints
        model_name: Base model name
        num_epochs: Number of training epochs
        batch_size: Per-device batch size
        gradient_accumulation_steps: Gradient accumulation steps
        learning_rate: Learning rate
        max_seq_length: Maximum sequence length
        lora_r: LoRA rank
        lora_alpha: LoRA alpha
        use_wandb: Enable WandB logging
        push_to_hub: Push to HuggingFace Hub
        hub_model_id: HuggingFace Hub model ID
    """
    # Initialize WandB
    if use_wandb:
        wandb.init(
            project="fincrime-llm",
            name=f"sar-mistral-{lora_r}r-{lora_alpha}alpha",
            config={
                "model": model_name,
                "lora_r": lora_r,
                "lora_alpha": lora_alpha,
                "learning_rate": learning_rate,
                "batch_size": batch_size,
                "epochs": num_epochs,
            },
        )

    # Load model and tokenizer
    model, tokenizer = setup_model_and_tokenizer(model_name)

    # Setup LoRA
    model = setup_lora(
        model,
        r=lora_r,
        lora_alpha=lora_alpha,
    )

    # Load dataset
    dataset = load_training_data(data_path)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        logging_steps=10,
        save_strategy="steps",
        save_steps=100,
        evaluation_strategy="steps",
        eval_steps=100,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        bf16=True,  # Use bfloat16 precision
        gradient_checkpointing=True,
        optim="paged_adamw_8bit",
        report_to="wandb" if use_wandb else "none",
        push_to_hub=push_to_hub,
        hub_model_id=hub_model_id,
        hub_strategy="end" if push_to_hub else "never",
    )

    # Initialize trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset.get("validation", None),
        tokenizer=tokenizer,
        max_seq_length=max_seq_length,
        dataset_text_field="text" if "text" in dataset["train"].column_names else None,
        formatting_func=(
            lambda x: f"### Instruction:\n{x['instruction']}\n\n### Input:\n{x['input']}\n\n### Response:\n{x['output']}"
            if "instruction" in dataset["train"].column_names
            else None
        ),
    )

    # Train
    logger.info("Starting training...")
    trainer.train()

    # Save final model
    logger.info(f"Saving model to {output_dir}/final")
    trainer.save_model(f"{output_dir}/final")
    tokenizer.save_pretrained(f"{output_dir}/final")

    logger.info("Training complete!")

    if use_wandb:
        wandb.finish()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Fine-tune Mistral 7B for SAR generation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to training data (directory or HF dataset)",
    )

    parser.add_argument(
        "--output",
        type=str,
        default="models/sar-mistral-7b",
        help="Output directory",
    )

    parser.add_argument(
        "--model",
        type=str,
        default="mistralai/Mistral-7B-v0.1",
        help="Base model name",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Per-device batch size",
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-4,
        help="Learning rate",
    )

    parser.add_argument(
        "--lora-r",
        type=int,
        default=16,
        help="LoRA rank",
    )

    parser.add_argument(
        "--lora-alpha",
        type=int,
        default=32,
        help="LoRA alpha",
    )

    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable WandB logging",
    )

    parser.add_argument(
        "--push-to-hub",
        action="store_true",
        help="Push model to HuggingFace Hub",
    )

    parser.add_argument(
        "--hub-model-id",
        type=str,
        help="HuggingFace Hub model ID",
    )

    args = parser.parse_args()

    # Validate HuggingFace token if pushing to hub
    if args.push_to_hub and not os.getenv("HUGGINGFACE_TOKEN"):
        logger.error("HUGGINGFACE_TOKEN not found in environment")
        sys.exit(1)

    # Train model
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
