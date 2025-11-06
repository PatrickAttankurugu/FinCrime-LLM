#!/usr/bin/env python3
"""
Inference script for FinCrime-LLM

Load trained model and run inference on financial crime detection tasks.
"""

import argparse
import logging
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FinCrimeLLM:
    """FinCrime-LLM inference wrapper."""

    def __init__(self, model_path: str, device: str = "auto"):
        """Initialize the model for inference."""
        logger.info(f"Loading model from {model_path}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map=device,
        )
        self.model.eval()

        logger.info("Model loaded successfully")

    def generate(
        self,
        instruction: str,
        input_text: str,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
    ) -> str:
        """Generate response for given instruction and input."""

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
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract only the response part
        if "### Response:" in response:
            response = response.split("### Response:")[1].strip()

        return response

    def analyze_transaction(self, transaction_details: str) -> str:
        """Analyze a transaction for potential financial crime."""
        return self.generate(
            instruction="Analyze the following transaction for potential money laundering or fraud indicators.",
            input_text=transaction_details,
        )

    def check_compliance(self, scenario: str, regulations: str = "General AML") -> str:
        """Check compliance against specified regulations."""
        return self.generate(
            instruction=f"Evaluate this scenario against {regulations} regulations and compliance requirements.",
            input_text=scenario,
        )


def main():
    parser = argparse.ArgumentParser(description="Run inference with FinCrime-LLM")
    parser.add_argument(
        "--model-path",
        type=str,
        default="models/final/fincrime-mistral-7b",
        help="Path to trained model",
    )
    parser.add_argument(
        "--instruction",
        type=str,
        required=True,
        help="Instruction for the model",
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input text to analyze",
    )
    parser.add_argument(
        "--max-tokens", type=int, default=512, help="Maximum tokens to generate"
    )
    parser.add_argument(
        "--temperature", type=float, default=0.7, help="Sampling temperature"
    )

    args = parser.parse_args()

    # Load model
    model = FinCrimeLLM(args.model_path)

    # Generate response
    response = model.generate(
        instruction=args.instruction,
        input_text=args.input,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
    )

    print("\n" + "=" * 80)
    print("RESPONSE:")
    print("=" * 80)
    print(response)
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
