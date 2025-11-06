#!/usr/bin/env python3
"""
Validate training datasets for quality and consistency.

Usage:
    python validate_dataset.py --input data/processed/train_alpaca.jsonl
    python validate_dataset.py --input data/processed/ --recursive
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def validate_alpaca_format(data: List[Dict]) -> Tuple[bool, List[str]]:
    """Validate Alpaca format data."""
    errors = []
    required_fields = ["instruction", "input", "output"]

    for idx, example in enumerate(data):
        # Check required fields
        for field in required_fields:
            if field not in example:
                errors.append(f"Example {idx}: Missing required field '{field}'")
            elif not isinstance(example[field], str):
                errors.append(f"Example {idx}: Field '{field}' must be a string")
            elif len(example[field].strip()) == 0:
                errors.append(f"Example {idx}: Field '{field}' is empty")

    return len(errors) == 0, errors


def validate_chatml_format(data: List[Dict]) -> Tuple[bool, List[str]]:
    """Validate ChatML format data."""
    errors = []

    for idx, example in enumerate(data):
        if "messages" not in example:
            errors.append(f"Example {idx}: Missing 'messages' field")
            continue

        messages = example["messages"]
        if not isinstance(messages, list):
            errors.append(f"Example {idx}: 'messages' must be a list")
            continue

        for msg_idx, msg in enumerate(messages):
            if "role" not in msg or "content" not in msg:
                errors.append(f"Example {idx}, Message {msg_idx}: Missing 'role' or 'content'")

    return len(errors) == 0, errors


def compute_statistics(data: List[Dict]) -> Dict:
    """Compute dataset statistics."""
    stats = {
        "total_examples": len(data),
        "instruction_lengths": [],
        "input_lengths": [],
        "output_lengths": [],
    }

    for example in data:
        if "instruction" in example:
            stats["instruction_lengths"].append(len(example["instruction"]))
        if "input" in example:
            stats["input_lengths"].append(len(example["input"]))
        if "output" in example:
            stats["output_lengths"].append(len(example["output"]))

    # Calculate summary stats
    for field in ["instruction_lengths", "input_lengths", "output_lengths"]:
        if stats[field]:
            stats[f"{field}_mean"] = sum(stats[field]) / len(stats[field])
            stats[f"{field}_max"] = max(stats[field])
            stats[f"{field}_min"] = min(stats[field])

    return stats


def validate_file(file_path: str) -> Dict:
    """Validate a single dataset file."""
    logger.info(f"Validating {file_path}")

    try:
        # Load data
        data = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))

        if not data:
            return {"status": "error", "message": "No data found in file"}

        # Detect format
        if "instruction" in data[0]:
            format_type = "alpaca"
            is_valid, errors = validate_alpaca_format(data)
        elif "messages" in data[0]:
            format_type = "chatml"
            is_valid, errors = validate_chatml_format(data)
        else:
            return {"status": "error", "message": "Unknown format"}

        # Compute statistics
        stats = compute_statistics(data)

        return {
            "status": "valid" if is_valid else "invalid",
            "format": format_type,
            "errors": errors,
            "statistics": stats,
        }

    except Exception as e:
        return {"status": "error", "message": str(e)}


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Validate training datasets")

    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input file or directory",
    )

    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively validate all .jsonl files in directory",
    )

    parser.add_argument(
        "--output-report",
        type=str,
        help="Save validation report to file",
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    results = {}

    # Collect files to validate
    if input_path.is_file():
        files = [input_path]
    elif input_path.is_dir():
        pattern = "**/*.jsonl" if args.recursive else "*.jsonl"
        files = list(input_path.glob(pattern))
    else:
        logger.error(f"Invalid input path: {input_path}")
        return

    # Validate each file
    for file_path in files:
        result = validate_file(str(file_path))
        results[str(file_path)] = result

    # Print summary
    logger.info(f"\n{'='*80}")
    logger.info("VALIDATION SUMMARY")
    logger.info(f"{'='*80}")

    for file_path, result in results.items():
        logger.info(f"\nFile: {file_path}")
        logger.info(f"Status: {result['status']}")

        if result["status"] == "valid":
            stats = result["statistics"]
            logger.info(f"Format: {result['format']}")
            logger.info(f"Total examples: {stats['total_examples']}")
            logger.info(f"Avg output length: {stats.get('output_lengths_mean', 0):.1f} chars")

        elif result["status"] == "invalid":
            logger.warning(f"Errors found: {len(result['errors'])}")
            for error in result["errors"][:5]:  # Show first 5 errors
                logger.warning(f"  - {error}")
            if len(result["errors"]) > 5:
                logger.warning(f"  ... and {len(result['errors']) - 5} more errors")

        else:
            logger.error(f"Error: {result.get('message', 'Unknown error')}")

    # Save report if requested
    if args.output_report:
        with open(args.output_report, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        logger.info(f"\nValidation report saved to {args.output_report}")

    logger.info(f"{'='*80}")


if __name__ == "__main__":
    main()
