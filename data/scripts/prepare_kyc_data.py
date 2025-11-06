#!/usr/bin/env python3
"""
Prepare KYC (Know Your Customer) data for fine-tuning.

This script processes KYC assessment data and formats it for instruction-tuning.

Usage:
    python prepare_kyc_data.py --input data/raw/kyc_data.jsonl --output data/processed/
"""

import argparse
import json
import logging
import random
from pathlib import Path
from typing import Dict, List

from datasets import Dataset, DatasetDict
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

KYC_INSTRUCTION_TEMPLATES = [
    "Perform a KYC risk assessment for the following customer:",
    "Evaluate the following customer profile for KYC compliance:",
    "Conduct a customer due diligence assessment based on:",
    "Assess the risk level of this customer profile:",
    "Review the following customer information and provide a KYC assessment:",
]


def load_kyc_data(input_path: str) -> List[Dict]:
    """Load KYC data from JSONL file."""
    logger.info(f"Loading KYC data from {input_path}")
    data = []

    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse line: {e}")

    logger.info(f"Loaded {len(data)} KYC records")
    return data


def format_kyc_for_training(kyc: Dict, format_type: str = "alpaca") -> Dict:
    """Format KYC data for training."""
    input_text = f"""
Customer Name: {kyc.get('customer_name', 'N/A')}
Customer Type: {kyc.get('customer_type', 'N/A')}
Country: {kyc.get('country', 'N/A')}
Occupation/Business: {kyc.get('occupation', 'N/A')}
Source of Funds: {kyc.get('source_of_funds', 'N/A')}
Expected Transaction Volume: {kyc.get('expected_volume', 'N/A')}
Account Purpose: {kyc.get('account_purpose', 'N/A')}

Additional Information:
{kyc.get('additional_info', 'N/A')}
""".strip()

    output_text = f"""
KYC RISK ASSESSMENT

Risk Rating: {kyc.get('risk_rating', 'N/A')}
Risk Score: {kyc.get('risk_score', 'N/A')}/100

RISK FACTORS:
{chr(10).join([f"- {factor}" for factor in kyc.get('risk_factors', [])])}

DUE DILIGENCE LEVEL: {kyc.get('dd_level', 'Standard')}

REQUIRED DOCUMENTATION:
{chr(10).join([f"- {doc}" for doc in kyc.get('required_docs', [])])}

MONITORING RECOMMENDATIONS:
{kyc.get('monitoring_recommendations', 'N/A')}

APPROVAL STATUS: {kyc.get('approval_status', 'N/A')}
NOTES: {kyc.get('notes', 'N/A')}
""".strip()

    instruction = random.choice(KYC_INSTRUCTION_TEMPLATES)

    if format_type == "alpaca":
        return {
            "instruction": instruction,
            "input": input_text,
            "output": output_text,
        }
    else:
        return {
            "messages": [
                {"role": "system", "content": "You are a KYC compliance specialist."},
                {"role": "user", "content": f"{instruction}\n\n{input_text}"},
                {"role": "assistant", "content": output_text},
            ],
        }


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Prepare KYC data for fine-tuning")

    parser.add_argument("--input", type=str, required=True, help="Input JSONL file")
    parser.add_argument(
        "--output",
        type=str,
        default="data/processed/",
        help="Output directory",
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["alpaca", "chatml"],
        default="alpaca",
        help="Output format",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()
    random.seed(args.seed)

    # Load and process data
    kyc_data = load_kyc_data(args.input)
    formatted_data = [format_kyc_for_training(kyc, args.format) for kyc in tqdm(kyc_data)]

    # Split data
    random.shuffle(formatted_data)
    train_size = int(len(formatted_data) * 0.8)
    val_size = int(len(formatted_data) * 0.1)

    train_data = formatted_data[:train_size]
    val_data = formatted_data[train_size : train_size + val_size]
    test_data = formatted_data[train_size + val_size :]

    # Save datasets
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    for split_name, split_data in [
        ("train", train_data),
        ("validation", val_data),
        ("test", test_data),
    ]:
        output_file = output_path / f"kyc_{split_name}_{args.format}.jsonl"
        with open(output_file, "w", encoding="utf-8") as f:
            for example in split_data:
                f.write(json.dumps(example, ensure_ascii=False) + "\n")
        logger.info(f"Saved {len(split_data)} examples to {output_file}")

    logger.info(f"Dataset preparation complete! Total: {len(formatted_data)} examples")


if __name__ == "__main__":
    main()
