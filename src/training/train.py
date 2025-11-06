#!/usr/bin/env python3
"""
FinCrime-LLM Training Script

This script fine-tunes Mistral 7B for financial crime detection using LoRA.
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import Optional

import torch
import yaml
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    BitsAndBytesConfig,
    set_seed,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("logs/training.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """Load training configuration from YAML file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    logger.info(f"Loaded configuration from {config_path}")
    return config


def format_instruction(example: dict, instruction_template: str) -> dict:
    """Format examples according to instruction template."""
    text = instruction_template.format(
        instruction=example.get("instruction", ""),
        input=example.get("input", ""),
        output=example.get("output", ""),
    )
    return {"text": text}


def prepare_model_and_tokenizer(config: dict):
    """Prepare model and tokenizer with quantization and LoRA."""
    model_config = config["model"]
    lora_config = config["lora"]
    quant_config = config["quantization"]

    # Configure quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=quant_config["load_in_4bit"],
        bnb_4bit_compute_dtype=getattr(torch, quant_config["bnb_4bit_compute_dtype"]),
        bnb_4bit_use_double_quant=quant_config["bnb_4bit_use_double_quant"],
        bnb_4bit_quant_type=quant_config["bnb_4bit_quant_type"],
    )

    # Load tokenizer
    logger.info(f"Loading tokenizer: {model_config['name']}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_config["name"],
        trust_remote_code=model_config.get("trust_remote_code", True),
    )

    # Set pad token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Load model
    logger.info(f"Loading model: {model_config['name']}")
    model = AutoModelForCausalLM.from_pretrained(
        model_config["name"],
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=model_config.get("trust_remote_code", True),
        torch_dtype=getattr(torch, model_config.get("torch_dtype", "bfloat16")),
    )

    # Prepare model for k-bit training
    model = prepare_model_for_kbit_training(model)

    # Configure LoRA
    peft_config = LoraConfig(
        r=lora_config["r"],
        lora_alpha=lora_config["lora_alpha"],
        target_modules=lora_config["target_modules"],
        lora_dropout=lora_config["lora_dropout"],
        bias=lora_config["bias"],
        task_type=lora_config["task_type"],
    )

    # Add LoRA adapters
    logger.info("Adding LoRA adapters to model")
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    return model, tokenizer


def prepare_datasets(config: dict, tokenizer):
    """Load and prepare training and validation datasets."""
    data_config = config["data"]

    # Load datasets
    logger.info("Loading datasets...")
    data_files = {
        "train": data_config["train_file"],
        "validation": data_config["validation_file"],
    }

    try:
        dataset = load_dataset("json", data_files=data_files)
        logger.info(f"Train dataset size: {len(dataset['train'])}")
        logger.info(f"Validation dataset size: {len(dataset['validation'])}")
    except FileNotFoundError as e:
        logger.error(f"Dataset file not found: {e}")
        logger.info("Please prepare your datasets first using src/data/prepare_dataset.py")
        sys.exit(1)

    # Format datasets
    instruction_template = data_config["instruction_template"]

    def format_func(example):
        return format_instruction(example, instruction_template)

    dataset = dataset.map(format_func, remove_columns=dataset["train"].column_names)

    return dataset["train"], dataset["validation"]


def main():
    parser = argparse.ArgumentParser(description="Train FinCrime-LLM")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/training_config.yaml",
        help="Path to training configuration file",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint to resume training from",
    )
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # Set seed for reproducibility
    set_seed(config.get("seed", 42))

    # Prepare model and tokenizer
    model, tokenizer = prepare_model_and_tokenizer(config)

    # Prepare datasets
    train_dataset, eval_dataset = prepare_datasets(config, tokenizer)

    # Configure training arguments
    training_config = config["training"]
    training_args = TrainingArguments(
        output_dir=training_config["output_dir"],
        num_train_epochs=training_config["num_train_epochs"],
        per_device_train_batch_size=training_config["per_device_train_batch_size"],
        per_device_eval_batch_size=training_config["per_device_eval_batch_size"],
        gradient_accumulation_steps=training_config["gradient_accumulation_steps"],
        gradient_checkpointing=training_config["gradient_checkpointing"],
        learning_rate=training_config["learning_rate"],
        weight_decay=training_config["weight_decay"],
        warmup_ratio=training_config["warmup_ratio"],
        lr_scheduler_type=training_config["lr_scheduler_type"],
        optim=training_config["optim"],
        fp16=training_config["fp16"],
        bf16=training_config["bf16"],
        logging_steps=training_config["logging_steps"],
        eval_steps=training_config["eval_steps"],
        save_steps=training_config["save_steps"],
        save_total_limit=training_config["save_total_limit"],
        evaluation_strategy=training_config["evaluation_strategy"],
        save_strategy=training_config["save_strategy"],
        load_best_model_at_end=training_config["load_best_model_at_end"],
        metric_for_best_model=training_config["metric_for_best_model"],
        report_to=training_config["report_to"],
        ddp_find_unused_parameters=training_config.get("ddp_find_unused_parameters", False),
        group_by_length=training_config.get("group_by_length", True),
        dataloader_num_workers=training_config.get("dataloader_num_workers", 4),
    )

    # Create trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        dataset_text_field="text",
        max_seq_length=config["data"]["max_seq_length"],
        packing=False,
    )

    # Train
    logger.info("Starting training...")
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    # Save final model
    final_model_path = "models/final/fincrime-mistral-7b"
    logger.info(f"Saving final model to {final_model_path}")
    trainer.model.save_pretrained(final_model_path)
    tokenizer.save_pretrained(final_model_path)

    logger.info("Training completed successfully!")


if __name__ == "__main__":
    main()
