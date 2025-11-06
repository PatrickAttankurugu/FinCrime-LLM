#!/usr/bin/env python3
"""
Data Preprocessing Script for FinCrime-LLM

Prepares financial crime datasets for training by:
- Cleaning and validating data
- Splitting into train/validation/test sets
- Formatting according to instruction template
- Anonymizing sensitive information
"""

import argparse
import json
import logging
import re
from pathlib import Path
from typing import List, Dict, Optional
import random

import pandas as pd
from sklearn.model_selection import train_test_split

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class FinCrimeDataProcessor:
    """Process and prepare financial crime datasets for training."""

    def __init__(
        self,
        input_dir: str,
        output_dir: str,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        seed: int = 42,
    ):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.seed = seed

        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)

        random.seed(seed)

    def anonymize_text(self, text: str) -> str:
        """
        Anonymize sensitive information in text.

        Replaces:
        - Account numbers
        - Names (simplified)
        - Phone numbers
        - Email addresses
        """
        # Anonymize account numbers (simplified pattern)
        text = re.sub(r"\b\d{10,16}\b", "[ACCOUNT_NUMBER]", text)

        # Anonymize phone numbers
        text = re.sub(
            r"\b(\+?\d{1,3}[-.\s]?)?(\(?\d{3}\)?[-.\s]?)?\d{3}[-.\s]?\d{4}\b",
            "[PHONE_NUMBER]",
            text,
        )

        # Anonymize email addresses
        text = re.sub(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", "[EMAIL]", text)

        # Anonymize currency amounts (keep structure but replace specific amounts)
        # This is optional - you may want to keep amounts for training
        # text = re.sub(r'\$[\d,]+\.?\d*', '[AMOUNT]', text)

        return text

    def validate_example(self, example: Dict) -> bool:
        """Validate that an example has required fields."""
        required_fields = ["instruction", "input", "output"]

        for field in required_fields:
            if field not in example or not example[field]:
                logger.warning(f"Missing or empty required field: {field}")
                return False

        return True

    def clean_example(self, example: Dict, anonymize: bool = True) -> Dict:
        """Clean and preprocess a single example."""
        cleaned = example.copy()

        # Strip whitespace
        for key in ["instruction", "input", "output"]:
            if key in cleaned:
                cleaned[key] = cleaned[key].strip()

        # Anonymize if requested
        if anonymize:
            for key in ["input", "output"]:
                if key in cleaned:
                    cleaned[key] = self.anonymize_text(cleaned[key])

        return cleaned

    def load_data(self, file_pattern: str = "*.json") -> List[Dict]:
        """Load all JSON/JSONL files from input directory."""
        all_data = []

        # Load JSON files
        for json_file in self.input_dir.glob("*.json"):
            logger.info(f"Loading {json_file}")
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        all_data.extend(data)
                    else:
                        all_data.append(data)
            except Exception as e:
                logger.error(f"Error loading {json_file}: {e}")

        # Load JSONL files
        for jsonl_file in self.input_dir.glob("*.jsonl"):
            logger.info(f"Loading {jsonl_file}")
            try:
                with open(jsonl_file, "r", encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            all_data.append(json.loads(line))
            except Exception as e:
                logger.error(f"Error loading {jsonl_file}: {e}")

        logger.info(f"Loaded {len(all_data)} examples")
        return all_data

    def process_data(self, data: List[Dict], anonymize: bool = True) -> List[Dict]:
        """Process and clean all data examples."""
        processed_data = []

        for i, example in enumerate(data):
            # Validate
            if not self.validate_example(example):
                logger.warning(f"Skipping invalid example at index {i}")
                continue

            # Clean
            cleaned = self.clean_example(example, anonymize=anonymize)
            processed_data.append(cleaned)

        logger.info(f"Processed {len(processed_data)} valid examples")
        return processed_data

    def split_data(
        self, data: List[Dict]
    ) -> tuple[List[Dict], List[Dict], List[Dict]]:
        """Split data into train, validation, and test sets."""
        # First split: separate test set
        train_val, test = train_test_split(
            data, test_size=self.test_ratio, random_state=self.seed
        )

        # Second split: separate validation from train
        val_size = self.val_ratio / (self.train_ratio + self.val_ratio)
        train, val = train_test_split(
            train_val, test_size=val_size, random_state=self.seed
        )

        logger.info(f"Split sizes - Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")
        return train, val, test

    def save_dataset(self, data: List[Dict], filename: str):
        """Save dataset to JSONL format."""
        output_path = self.output_dir.parent / "datasets" / filename
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            for example in data:
                f.write(json.dumps(example, ensure_ascii=False) + "\n")

        logger.info(f"Saved {len(data)} examples to {output_path}")

    def create_sample_data(self):
        """Create sample financial crime dataset for demonstration."""
        sample_data = [
            {
                "instruction": "Analyze the following transaction for potential money laundering indicators.",
                "input": "Customer John Doe made 12 cash deposits of $9,500 each over 3 weeks to account ending in 1234. Customer is a retail worker with reported income of $35,000/year.",
                "output": "HIGH RISK - Multiple red flags detected:\n1. Structuring: Multiple deposits just under $10,000 reporting threshold\n2. Inconsistent with income: Total deposits ($114,000) far exceed annual income\n3. Pattern: Systematic deposits suggest deliberate structuring\n\nRecommendation: File SAR (Suspicious Activity Report) and escalate to compliance team.",
            },
            {
                "instruction": "Evaluate this transaction against Nigerian CBN AML regulations.",
                "input": "International wire transfer of $45,000 from Nigeria to offshore account in Seychelles. Stated purpose: 'Business consulting fees'. No supporting documentation provided.",
                "output": "COMPLIANCE REVIEW REQUIRED:\n1. CBN regulations require documentation for transfers >$10,000\n2. High-risk jurisdiction (Seychelles) - enhanced due diligence needed\n3. Missing: Contract, invoice, business relationship proof\n\nActions required:\n- Request supporting documentation\n- Verify business relationship\n- Conduct enhanced due diligence\n- Consider holding transfer pending review",
            },
            {
                "instruction": "Assess the fraud risk in this scenario.",
                "input": "New account opened online with scanned ID. Within 24 hours: received $50,000 transfer, then immediately initiated 8 outbound transfers to different accounts in various countries.",
                "output": "CRITICAL FRAUD INDICATORS:\n1. Rapid cycling: Funds in/out within 24 hours suggests money mule activity\n2. New account: No established relationship\n3. Multiple beneficiaries: Scattering funds is typical fraud pattern\n4. International transfers: Higher risk, harder to reverse\n\nImmediate actions:\n- Freeze account and pending transfers\n- Enhanced identity verification\n- Contact customer via registered phone\n- Report to fraud team and law enforcement if confirmed",
            },
        ]

        output_path = self.input_dir / "sample_data.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(sample_data, f, indent=2, ensure_ascii=False)

        logger.info(f"Created sample data at {output_path}")

    def run(self, anonymize: bool = True, create_sample: bool = False):
        """Run the complete data processing pipeline."""
        logger.info("Starting data processing pipeline...")

        # Create sample data if requested
        if create_sample:
            logger.info("Creating sample data...")
            self.create_sample_data()

        # Load data
        data = self.load_data()

        if not data:
            logger.warning("No data found. Use --create-sample to generate sample data.")
            return

        # Process data
        processed_data = self.process_data(data, anonymize=anonymize)

        if not processed_data:
            logger.error("No valid data after processing")
            return

        # Split data
        train, val, test = self.split_data(processed_data)

        # Save datasets
        self.save_dataset(train, "train.jsonl")
        self.save_dataset(val, "validation.jsonl")
        self.save_dataset(test, "test.jsonl")

        logger.info("Data processing complete!")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare financial crime datasets for training"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="data/raw",
        help="Input directory containing raw data files",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/processed",
        help="Output directory for processed data",
    )
    parser.add_argument(
        "--train-ratio", type=float, default=0.8, help="Training set ratio"
    )
    parser.add_argument(
        "--val-ratio", type=float, default=0.1, help="Validation set ratio"
    )
    parser.add_argument(
        "--test-ratio", type=float, default=0.1, help="Test set ratio"
    )
    parser.add_argument(
        "--no-anonymize",
        action="store_true",
        help="Skip anonymization of sensitive data",
    )
    parser.add_argument(
        "--create-sample",
        action="store_true",
        help="Create sample dataset for demonstration",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    # Validate ratios
    total_ratio = args.train_ratio + args.val_ratio + args.test_ratio
    if abs(total_ratio - 1.0) > 0.01:
        logger.error(f"Ratios must sum to 1.0 (got {total_ratio})")
        return

    # Create processor
    processor = FinCrimeDataProcessor(
        input_dir=args.input,
        output_dir=args.output,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )

    # Run processing
    processor.run(anonymize=not args.no_anonymize, create_sample=args.create_sample)


if __name__ == "__main__":
    main()
