#!/usr/bin/env python3
"""
Synthetic SAR (Suspicious Activity Report) Generator for African Financial Contexts.

This script uses OpenAI GPT-4 to generate realistic suspicious activity reports
covering various African countries and financial crime typologies.

Usage:
    python generate_synthetic_sars.py --count 100 --output data/raw/synthetic_sars.jsonl
    python generate_synthetic_sars.py --country Ghana --typology "money_laundering" --count 50
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import openai
from dotenv import load_dotenv
from tqdm import tqdm

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# African countries coverage
AFRICAN_COUNTRIES = [
    "Ghana",
    "Nigeria",
    "Kenya",
    "South Africa",
    "Tanzania",
    "Uganda",
    "Rwanda",
    "Zambia",
    "Botswana",
    "Ethiopia",
]

# Financial crime typologies
CRIME_TYPOLOGIES = [
    "money_laundering",
    "terrorist_financing",
    "fraud",
    "corruption",
    "trade_based_laundering",
    "cash_smuggling",
    "cybercrime",
    "ponzi_schemes",
    "shell_company_abuse",
    "invoice_manipulation",
]

# SAR generation prompt template
SAR_PROMPT_TEMPLATE = """Generate a realistic and detailed Suspicious Activity Report (SAR) for {country}.

Context:
- Country: {country}
- Crime Typology: {typology}
- Local Currency: {currency}
- Include local payment systems (e.g., M-Pesa for Kenya, MTN Mobile Money for Ghana/Uganda)

Requirements:
1. Create a realistic scenario involving suspicious financial activity
2. Include specific transaction details (amounts, dates, account numbers)
3. Reference local banks, mobile money providers, or financial institutions
4. Describe behavioral red flags and suspicious patterns
5. Include customer due diligence (CDD) findings
6. Reference relevant regulatory frameworks (e.g., GIABA, ESAAMLG, FIC Act)
7. Use realistic African names and business entities
8. Include recommendation for further investigation

Format the output as a structured JSON with these fields:
- report_id: unique identifier (e.g., SAR-{country_code}-YYYYMMDD-XXXX)
- country: {country}
- typology: {typology}
- currency: local currency code
- reporting_institution: name of bank/FI
- subject_name: individual or entity under suspicion
- subject_type: "individual" or "entity"
- transaction_summary: brief overview
- detailed_narrative: comprehensive description of suspicious activity
- transaction_details: list of suspicious transactions with amounts, dates, counterparties
- red_flags: list of specific suspicious indicators
- cdd_findings: customer due diligence results
- regulatory_references: applicable laws/regulations
- recommendation: suggested next steps
- report_date: date in YYYY-MM-DD format
- total_amount: total value of suspicious transactions

Generate only the JSON object, no additional text."""


def get_currency(country: str) -> str:
    """Get the local currency for a given African country."""
    currency_map = {
        "Ghana": "GHS",
        "Nigeria": "NGN",
        "Kenya": "KES",
        "South Africa": "ZAR",
        "Tanzania": "TZS",
        "Uganda": "UGX",
        "Rwanda": "RWF",
        "Zambia": "ZMW",
        "Botswana": "BWP",
        "Ethiopia": "ETB",
    }
    return currency_map.get(country, "USD")


def get_country_code(country: str) -> str:
    """Get the ISO country code."""
    code_map = {
        "Ghana": "GH",
        "Nigeria": "NG",
        "Kenya": "KE",
        "South Africa": "ZA",
        "Tanzania": "TZ",
        "Uganda": "UG",
        "Rwanda": "RW",
        "Zambia": "ZM",
        "Botswana": "BW",
        "Ethiopia": "ET",
    }
    return code_map.get(country, "XX")


def generate_sar(
    country: str,
    typology: str,
    api_key: str,
    model: str = "gpt-4",
    max_retries: int = 3,
) -> Optional[Dict]:
    """
    Generate a single synthetic SAR using OpenAI API.

    Args:
        country: African country for the SAR
        typology: Type of financial crime
        api_key: OpenAI API key
        model: OpenAI model to use
        max_retries: Maximum number of retry attempts

    Returns:
        Dictionary containing SAR data or None if generation failed
    """
    client = openai.OpenAI(api_key=api_key)
    currency = get_currency(country)
    country_code = get_country_code(country)

    prompt = SAR_PROMPT_TEMPLATE.format(
        country=country,
        typology=typology.replace("_", " ").title(),
        currency=currency,
        country_code=country_code,
    )

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are an expert in African financial crime compliance "
                            "and anti-money laundering. Generate realistic, detailed "
                            "suspicious activity reports."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.9,  # Higher temperature for diversity
                max_tokens=2000,
            )

            # Parse the response
            content = response.choices[0].message.content.strip()

            # Try to extract JSON if wrapped in markdown
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()

            sar_data = json.loads(content)
            logger.debug(f"Successfully generated SAR for {country} - {typology}")
            return sar_data

        except json.JSONDecodeError as e:
            logger.warning(
                f"Attempt {attempt + 1}/{max_retries}: Failed to parse JSON: {e}"
            )
            if attempt == max_retries - 1:
                logger.error(f"Failed to generate valid SAR after {max_retries} attempts")
                return None

        except Exception as e:
            logger.error(f"Attempt {attempt + 1}/{max_retries}: Error generating SAR: {e}")
            if attempt == max_retries - 1:
                return None

    return None


def generate_dataset(
    count: int,
    output_path: str,
    country: Optional[str] = None,
    typology: Optional[str] = None,
    model: str = "gpt-4",
) -> None:
    """
    Generate a dataset of synthetic SARs.

    Args:
        count: Number of SARs to generate
        output_path: Path to save the dataset (JSONL format)
        country: Specific country (None for random selection)
        typology: Specific typology (None for random selection)
        model: OpenAI model to use
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.error("OPENAI_API_KEY not found in environment variables")
        sys.exit(1)

    # Ensure output directory exists
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    generated_sars = []
    failed_count = 0

    logger.info(f"Starting generation of {count} synthetic SARs...")
    logger.info(f"Output file: {output_path}")
    if country:
        logger.info(f"Country filter: {country}")
    if typology:
        logger.info(f"Typology filter: {typology}")

    with open(output_path, "w", encoding="utf-8") as f:
        for i in tqdm(range(count), desc="Generating SARs"):
            # Select country and typology
            selected_country = country or AFRICAN_COUNTRIES[i % len(AFRICAN_COUNTRIES)]
            selected_typology = typology or CRIME_TYPOLOGIES[i % len(CRIME_TYPOLOGIES)]

            # Generate SAR
            sar = generate_sar(
                country=selected_country,
                typology=selected_typology,
                api_key=api_key,
                model=model,
            )

            if sar:
                # Add metadata
                sar["generation_metadata"] = {
                    "model": model,
                    "index": i,
                    "requested_country": selected_country,
                    "requested_typology": selected_typology,
                }

                # Write to JSONL
                f.write(json.dumps(sar, ensure_ascii=False) + "\n")
                f.flush()  # Ensure data is written immediately
                generated_sars.append(sar)
            else:
                failed_count += 1
                logger.warning(f"Failed to generate SAR {i+1}/{count}")

    # Summary
    success_count = len(generated_sars)
    logger.info(f"\n{'='*60}")
    logger.info(f"Generation Complete!")
    logger.info(f"{'='*60}")
    logger.info(f"Total requested: {count}")
    logger.info(f"Successfully generated: {success_count}")
    logger.info(f"Failed: {failed_count}")
    logger.info(f"Success rate: {success_count/count*100:.1f}%")
    logger.info(f"Output saved to: {output_path}")
    logger.info(f"{'='*60}")


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="Generate synthetic SARs for African financial crime detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate 100 SARs across all countries and typologies
  python generate_synthetic_sars.py --count 100 --output data/raw/sars.jsonl

  # Generate 50 SARs for Ghana only
  python generate_synthetic_sars.py --count 50 --country Ghana --output data/raw/ghana_sars.jsonl

  # Generate money laundering SARs
  python generate_synthetic_sars.py --count 30 --typology money_laundering --output data/raw/ml_sars.jsonl

  # Use GPT-3.5-turbo (faster, cheaper)
  python generate_synthetic_sars.py --count 200 --model gpt-3.5-turbo --output data/raw/sars.jsonl
        """,
    )

    parser.add_argument(
        "--count",
        type=int,
        default=100,
        help="Number of SARs to generate (default: 100)",
    )

    parser.add_argument(
        "--output",
        type=str,
        default="data/raw/synthetic_sars.jsonl",
        help="Output file path (default: data/raw/synthetic_sars.jsonl)",
    )

    parser.add_argument(
        "--country",
        type=str,
        choices=AFRICAN_COUNTRIES,
        help="Generate SARs for a specific country only",
    )

    parser.add_argument(
        "--typology",
        type=str,
        choices=CRIME_TYPOLOGIES,
        help="Generate SARs for a specific crime typology only",
    )

    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4",
        choices=["gpt-4", "gpt-4-turbo-preview", "gpt-3.5-turbo"],
        help="OpenAI model to use (default: gpt-4)",
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    if args.verbose:
        logger.setLevel(logging.DEBUG)

    # Validate inputs
    if args.count <= 0:
        logger.error("Count must be a positive integer")
        sys.exit(1)

    # Generate dataset
    generate_dataset(
        count=args.count,
        output_path=args.output,
        country=args.country,
        typology=args.typology,
        model=args.model,
    )


if __name__ == "__main__":
    main()
