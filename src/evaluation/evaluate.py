#!/usr/bin/env python3
"""
Evaluation script for FinCrime-LLM

Evaluate model performance on test dataset.
"""

import argparse
import json
import logging
from pathlib import Path
from typing import List, Dict

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Evaluate FinCrime-LLM on test data."""

    def __init__(self, model_path: str, test_data_path: str):
        """Initialize evaluator with model and test data."""
        logger.info(f"Loading model from {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        self.model.eval()

        logger.info(f"Loading test data from {test_data_path}")
        self.test_data = load_dataset("json", data_files=test_data_path)["train"]

    def generate_response(self, instruction: str, input_text: str) -> str:
        """Generate model response."""
        prompt = f"""### Instruction:
{instruction}

### Input:
{input_text}

### Response:
"""

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        if "### Response:" in response:
            response = response.split("### Response:")[1].strip()

        return response

    def evaluate(self, output_path: str = "logs/evaluation_results.json"):
        """Run evaluation on test set."""
        results = []

        logger.info(f"Evaluating on {len(self.test_data)} examples...")

        for example in tqdm(self.test_data):
            instruction = example.get("instruction", "")
            input_text = example.get("input", "")
            expected_output = example.get("output", "")

            generated_output = self.generate_response(instruction, input_text)

            results.append(
                {
                    "instruction": instruction,
                    "input": input_text,
                    "expected": expected_output,
                    "generated": generated_output,
                }
            )

        # Save results
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        logger.info(f"Evaluation complete. Results saved to {output_path}")

        # Print sample outputs
        logger.info("\nSample Outputs:")
        for i, result in enumerate(results[:3]):
            print(f"\n--- Example {i+1} ---")
            print(f"Instruction: {result['instruction'][:100]}...")
            print(f"Generated: {result['generated'][:200]}...")

        return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate FinCrime-LLM")
    parser.add_argument(
        "--model",
        type=str,
        default="models/final/fincrime-mistral-7b",
        help="Path to trained model",
    )
    parser.add_argument(
        "--test-data",
        type=str,
        default="data/datasets/test.jsonl",
        help="Path to test dataset",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="logs/evaluation_results.json",
        help="Output path for results",
    )

    args = parser.parse_args()

    evaluator = ModelEvaluator(args.model, args.test_data)
    evaluator.evaluate(args.output)


if __name__ == "__main__":
    main()
