#!/usr/bin/env python3
"""
Prepare SAR data for fine-tuning Mistral 7B.

This script processes raw SAR data and formats it for instruction-tuning,
creating train/validation/test splits with proper prompt templates.

Usage:
    python prepare_sar_data.py --input data/raw/synthetic_sars.jsonl --output data/processed/
    python prepare_sar_data.py --input data/raw/sars.jsonl --format alpaca --split 0.8/0.1/0.1
"""

import argparse
import json
import logging
import random
from pathlib import Path
from typing import Dict, List, Tuple

from datasets import Dataset, DatasetDict
from tqdm import tqdm

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Instruction templates for SAR generation
SAR_INSTRUCTION_TEMPLATES = [
    "Generate a suspicious activity report based on the following transaction details:",
    "Create a detailed SAR for the following suspicious financial activity:",
    "Write a comprehensive suspicious activity report for:",
    "Prepare an SAR documenting the following suspicious transactions:",
    "Analyze and document the following suspicious activity in SAR format:",
]

# Instruction templates for SAR analysis
ANALYSIS_INSTRUCTION_TEMPLATES = [
    "Analyze the following transaction data and identify suspicious patterns:",
    "Review these transactions and highlight potential red flags:",
    "Assess the following financial activity for signs of money laundering:",
    "Examine these transactions and explain what makes them suspicious:",
    "Evaluate the following financial behavior and identify compliance concerns:",
]


def load_sar_data(input_path: str) -> List[Dict]:
    """
    Load SAR data from JSONL file.

    Args:
        input_path: Path to input JSONL file

    Returns:
        List of SAR dictionaries
    """
    logger.info(f"Loading SAR data from {input_path}")
    sars = []

    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    sar = json.loads(line)
                    sars.append(sar)
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse line: {e}")

    logger.info(f"Loaded {len(sars)} SARs")
    return sars


def format_sar_for_training(sar: Dict, format_type: str = "alpaca") -> Dict:
    """
    Format a SAR into instruction-tuning format.

    Args:
        sar: SAR dictionary
        format_type: Format type ("alpaca" or "chatml")

    Returns:
        Formatted training example
    """
    # Create input from transaction details
    transactions_str = "\n".join(
        [
            f"- {t.get('date', 'N/A')}: {t.get('amount', 'N/A')} "
            f"{sar.get('currency', '')} to/from {t.get('counterparty', 'N/A')}"
            for t in sar.get("transaction_details", [])
        ]
    )

    input_text = f"""
Country: {sar.get('country', 'N/A')}
Subject: {sar.get('subject_name', 'N/A')} ({sar.get('subject_type', 'N/A')})
Reporting Institution: {sar.get('reporting_institution', 'N/A')}
Total Amount: {sar.get('total_amount', 'N/A')} {sar.get('currency', '')}

Transaction Details:
{transactions_str}

Summary: {sar.get('transaction_summary', 'N/A')}
""".strip()

    # Create output (the complete SAR narrative)
    output_text = f"""
SUSPICIOUS ACTIVITY REPORT

Report ID: {sar.get('report_id', 'N/A')}
Report Date: {sar.get('report_date', 'N/A')}
Typology: {sar.get('typology', 'N/A')}

DETAILED NARRATIVE:
{sar.get('detailed_narrative', 'N/A')}

RED FLAGS IDENTIFIED:
{chr(10).join([f"- {flag}" for flag in sar.get('red_flags', [])])}

CUSTOMER DUE DILIGENCE FINDINGS:
{sar.get('cdd_findings', 'N/A')}

REGULATORY REFERENCES:
{sar.get('regulatory_references', 'N/A')}

RECOMMENDATION:
{sar.get('recommendation', 'N/A')}
""".strip()

    # Select random instruction template
    instruction = random.choice(SAR_INSTRUCTION_TEMPLATES)

    if format_type == "alpaca":
        return {
            "instruction": instruction,
            "input": input_text,
            "output": output_text,
            "metadata": {
                "country": sar.get("country"),
                "typology": sar.get("typology"),
                "report_id": sar.get("report_id"),
            },
        }
    elif format_type == "chatml":
        return {
            "messages": [
                {"role": "system", "content": "You are an expert financial crime analyst."},
                {"role": "user", "content": f"{instruction}\n\n{input_text}"},
                {"role": "assistant", "content": output_text},
            ],
            "metadata": {
                "country": sar.get("country"),
                "typology": sar.get("typology"),
                "report_id": sar.get("report_id"),
            },
        }
    else:
        raise ValueError(f"Unknown format type: {format_type}")


def create_analysis_examples(sars: List[Dict], format_type: str = "alpaca") -> List[Dict]:
    """
    Create analysis-focused training examples.

    Args:
        sars: List of SAR dictionaries
        format_type: Format type

    Returns:
        List of analysis training examples
    """
    examples = []

    for sar in sars:
        transactions_str = "\n".join(
            [
                f"- {t.get('date', 'N/A')}: {t.get('amount', 'N/A')} "
                f"{sar.get('currency', '')} to/from {t.get('counterparty', 'N/A')}"
                for t in sar.get("transaction_details", [])
            ]
        )

        input_text = f"""
Subject: {sar.get('subject_name', 'N/A')}
Country: {sar.get('country', 'N/A')}
Transactions:
{transactions_str}
""".strip()

        # Focus on red flags and analysis
        output_text = f"""
SUSPICIOUS PATTERNS IDENTIFIED:

Red Flags:
{chr(10).join([f"{i+1}. {flag}" for i, flag in enumerate(sar.get('red_flags', []))])}

Analysis:
{sar.get('detailed_narrative', 'N/A')[:500]}...

Risk Level: {sar.get('typology', 'Unknown').upper()}
Recommendation: {sar.get('recommendation', 'N/A')}
""".strip()

        instruction = random.choice(ANALYSIS_INSTRUCTION_TEMPLATES)

        if format_type == "alpaca":
            examples.append(
                {
                    "instruction": instruction,
                    "input": input_text,
                    "output": output_text,
                    "metadata": {
                        "country": sar.get("country"),
                        "typology": sar.get("typology"),
                    },
                }
            )

    return examples


def split_dataset(
    data: List[Dict], split_ratios: Tuple[float, float, float] = (0.8, 0.1, 0.1)
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Split data into train/validation/test sets.

    Args:
        data: List of training examples
        split_ratios: Tuple of (train, val, test) ratios

    Returns:
        Tuple of (train_data, val_data, test_data)
    """
    # Shuffle data
    random.shuffle(data)

    total = len(data)
    train_size = int(total * split_ratios[0])
    val_size = int(total * split_ratios[1])

    train_data = data[:train_size]
    val_data = data[train_size : train_size + val_size]
    test_data = data[train_size + val_size :]

    logger.info(f"Split sizes - Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")

    return train_data, val_data, test_data


def save_dataset(
    train_data: List[Dict],
    val_data: List[Dict],
    test_data: List[Dict],
    output_dir: str,
    format_type: str = "alpaca",
) -> None:
    """
    Save datasets to disk.

    Args:
        train_data: Training examples
        val_data: Validation examples
        test_data: Test examples
        output_dir: Output directory
        format_type: Format type
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save as JSONL
    for split_name, split_data in [
        ("train", train_data),
        ("validation", val_data),
        ("test", test_data),
    ]:
        output_file = output_path / f"{split_name}_{format_type}.jsonl"
        with open(output_file, "w", encoding="utf-8") as f:
            for example in split_data:
                f.write(json.dumps(example, ensure_ascii=False) + "\n")
        logger.info(f"Saved {len(split_data)} examples to {output_file}")

    # Also save as HuggingFace dataset
    dataset_dict = DatasetDict(
        {
            "train": Dataset.from_list(train_data),
            "validation": Dataset.from_list(val_data),
            "test": Dataset.from_list(test_data),
        }
    )

    hf_output_path = output_path / f"sar_dataset_{format_type}"
    dataset_dict.save_to_disk(str(hf_output_path))
    logger.info(f"Saved HuggingFace dataset to {hf_output_path}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Prepare SAR data for fine-tuning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input JSONL file with raw SARs",
    )

    parser.add_argument(
        "--output",
        type=str,
        default="data/processed/",
        help="Output directory (default: data/processed/)",
    )

    parser.add_argument(
        "--format",
        type=str,
        choices=["alpaca", "chatml"],
        default="alpaca",
        help="Output format (default: alpaca)",
    )

    parser.add_argument(
        "--split",
        type=str,
        default="0.8/0.1/0.1",
        help="Train/val/test split ratios (default: 0.8/0.1/0.1)",
    )

    parser.add_argument(
        "--include-analysis",
        action="store_true",
        help="Include analysis-focused examples",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )

    args = parser.parse_args()

    # Set random seed
    random.seed(args.seed)

    # Parse split ratios
    split_ratios = tuple(map(float, args.split.split("/")))
    if len(split_ratios) != 3 or sum(split_ratios) != 1.0:
        logger.error("Split ratios must sum to 1.0")
        return

    # Load data
    sars = load_sar_data(args.input)

    if not sars:
        logger.error("No SARs loaded")
        return

    # Format data
    logger.info(f"Formatting SARs in {args.format} format...")
    formatted_data = []

    for sar in tqdm(sars, desc="Formatting SARs"):
        formatted_data.append(format_sar_for_training(sar, args.format))

    # Add analysis examples if requested
    if args.include_analysis:
        logger.info("Creating analysis examples...")
        analysis_examples = create_analysis_examples(sars, args.format)
        formatted_data.extend(analysis_examples)
        logger.info(f"Added {len(analysis_examples)} analysis examples")

    # Split dataset
    train_data, val_data, test_data = split_dataset(formatted_data, split_ratios)

    # Save dataset
    save_dataset(train_data, val_data, test_data, args.output, args.format)

    # Print summary
    logger.info(f"\n{'='*60}")
    logger.info(f"Dataset Preparation Complete!")
    logger.info(f"{'='*60}")
    logger.info(f"Total examples: {len(formatted_data)}")
    logger.info(f"Training examples: {len(train_data)}")
    logger.info(f"Validation examples: {len(val_data)}")
    logger.info(f"Test examples: {len(test_data)}")
    logger.info(f"Format: {args.format}")
    logger.info(f"Output directory: {args.output}")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    main()
