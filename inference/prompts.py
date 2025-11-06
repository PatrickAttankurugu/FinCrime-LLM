"""Prompt templates for different tasks."""

from typing import Dict


def create_sar_prompt(transaction_data: Dict) -> str:
    """Create prompt for SAR generation."""
    return f"""### Instruction:
Generate a detailed Suspicious Activity Report (SAR) for the following transaction information.

### Input:
Country: {transaction_data.get('country', 'N/A')}
Subject: {transaction_data.get('subject_name', 'N/A')}
Institution: {transaction_data.get('institution', 'N/A')}
Total Amount: {transaction_data.get('total_amount', 'N/A')} {transaction_data.get('currency', '')}

Transactions:
{transaction_data.get('transactions', 'N/A')}

Summary: {transaction_data.get('summary', 'N/A')}

### Response:
"""


def create_kyc_prompt(customer_data: Dict) -> str:
    """Create prompt for KYC assessment."""
    return f"""### Instruction:
Perform a comprehensive KYC risk assessment for the following customer profile.

### Input:
Customer Name: {customer_data.get('name', 'N/A')}
Customer Type: {customer_data.get('type', 'N/A')}
Country: {customer_data.get('country', 'N/A')}
Occupation/Business: {customer_data.get('occupation', 'N/A')}
Source of Funds: {customer_data.get('source_of_funds', 'N/A')}
Expected Volume: {customer_data.get('expected_volume', 'N/A')}

Additional Information:
{customer_data.get('additional_info', 'N/A')}

### Response:
"""


def create_analysis_prompt(transaction_data: Dict) -> str:
    """Create prompt for transaction analysis."""
    return f"""### Instruction:
Analyze the following transactions and identify any suspicious patterns or red flags.

### Input:
{transaction_data.get('description', 'N/A')}

Transactions:
{transaction_data.get('transactions', 'N/A')}

### Response:
"""
