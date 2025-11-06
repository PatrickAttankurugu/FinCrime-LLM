#!/usr/bin/env python3
"""
Generate SARs and KYC assessments using fine-tuned model.

Usage:
    python generate.py --model models/sar-mistral-7b/final --task sar --input "Transaction data..."
    python generate.py --model models/fincrime-mistral-7b/final --batch --input-file inputs.jsonl
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import torch
from load_model import load_fincrime_model
from prompts import create_sar_prompt, create_kyc_prompt, create_analysis_prompt
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_text(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 1024,
    temperature: float = 0.7,
    top_p: float = 0.95,
    top_k: int = 50,
    repetition_penalty: float = 1.1,
) -> str:
    """
    Generate text using the model.

    Args:
        model: Loaded model
        tokenizer: Tokenizer
        prompt: Input prompt
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        top_p: Nucleus sampling parameter
        top_k: Top-k sampling parameter
        repetition_penalty: Repetition penalty

    Returns:
        Generated text
    """
    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    # Decode output
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract only the generated part (after the prompt)
    if "### Response:" in generated_text:
        response = generated_text.split("### Response:")[-1].strip()
    else:
        response = generated_text[len(prompt):].strip()

    return response


def generate_sar(model, tokenizer, transaction_data: Dict, **gen_kwargs) -> str:
    """Generate a SAR from transaction data."""
    prompt = create_sar_prompt(transaction_data)
    return generate_text(model, tokenizer, prompt, **gen_kwargs)


def generate_kyc_assessment(model, tokenizer, customer_data: Dict, **gen_kwargs) -> str:
    """Generate a KYC assessment from customer data."""
    prompt = create_kyc_prompt(customer_data)
    return generate_text(model, tokenizer, prompt, **gen_kwargs)


def generate_transaction_analysis(model, tokenizer, transaction_data: Dict, **gen_kwargs) -> str:
    """Analyze transactions for suspicious patterns."""
    prompt = create_analysis_prompt(transaction_data)
    return generate_text(model, tokenizer, prompt, **gen_kwargs)


def batch_generate(
    model,
    tokenizer,
    input_file: str,
    output_file: str,
    task: str = "sar",
    **gen_kwargs,
) -> None:
    """
    Generate outputs for a batch of inputs.

    Args:
        model: Loaded model
        tokenizer: Tokenizer
        input_file: Input JSONL file
        output_file: Output JSONL file
        task: Task type (sar, kyc, analysis)
        **gen_kwargs: Generation parameters
    """
    logger.info(f"Processing batch from {input_file}")

    # Task-specific generation function
    task_funcs = {
        "sar": generate_sar,
        "kyc": generate_kyc_assessment,
        "analysis": generate_transaction_analysis,
    }

    if task not in task_funcs:
        raise ValueError(f"Unknown task: {task}")

    gen_func = task_funcs[task]

    # Load inputs
    inputs = []
    with open(input_file, "r") as f:
        for line in f:
            if line.strip():
                inputs.append(json.loads(line))

    # Generate outputs
    results = []
    for input_data in tqdm(inputs, desc=f"Generating {task}"):
        try:
            output = gen_func(model, tokenizer, input_data, **gen_kwargs)
            results.append({"input": input_data, "output": output, "task": task})
        except Exception as e:
            logger.error(f"Error generating for input: {e}")
            results.append({"input": input_data, "error": str(e)})

    # Save results
    with open(output_file, "w") as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

    logger.info(f"Saved {len(results)} results to {output_file}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Generate outputs using FinCrime-LLM")

    parser.add_argument("--model", type=str, required=True, help="Path to model")
    parser.add_argument("--task", type=str, choices=["sar", "kyc", "analysis"], default="sar")
    parser.add_argument("--input", type=str, help="Input text or JSON")
    parser.add_argument("--input-file", type=str, help="Input JSONL file for batch processing")
    parser.add_argument("--output-file", type=str, help="Output file for batch processing")
    parser.add_argument("--max-tokens", type=int, default=1024, help="Max new tokens")
    parser.add_argument("--temperature", type=float, default=0.7, help="Temperature")
    parser.add_argument("--top-p", type=float, default=0.95, help="Top-p")
    parser.add_argument("--top-k", type=int, default=50, help="Top-k")

    args = parser.parse_args()

    # Load model
    logger.info(f"Loading model from {args.model}")
    model, tokenizer = load_fincrime_model(args.model)

    gen_kwargs = {
        "max_new_tokens": args.max_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
    }

    # Batch or single generation
    if args.input_file:
        if not args.output_file:
            args.output_file = "outputs.jsonl"
        batch_generate(
            model,
            tokenizer,
            args.input_file,
            args.output_file,
            args.task,
            **gen_kwargs,
        )
    elif args.input:
        # Parse input
        try:
            input_data = json.loads(args.input)
        except:
            input_data = {"text": args.input}

        # Generate
        if args.task == "sar":
            output = generate_sar(model, tokenizer, input_data, **gen_kwargs)
        elif args.task == "kyc":
            output = generate_kyc_assessment(model, tokenizer, input_data, **gen_kwargs)
        else:
            output = generate_transaction_analysis(model, tokenizer, input_data, **gen_kwargs)

        print("\n" + "="*80)
        print(f"{args.task.upper()} OUTPUT:")
        print("="*80)
        print(output)
        print("="*80)
    else:
        parser.error("Either --input or --input-file required")


if __name__ == "__main__":
    main()
