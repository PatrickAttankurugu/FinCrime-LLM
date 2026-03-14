#!/bin/bash
# Quick start script for FinCrime-LLM

set -e

echo "=========================================="
echo "FinCrime-LLM Quick Start"
echo "=========================================="

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Create necessary directories
mkdir -p data/raw data/processed data/datasets
mkdir -p models/checkpoints models/final
mkdir -p logs

echo "=========================================="
echo "Setup complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Copy .env.example to .env and add your API keys"
echo "2. Generate synthetic data: python data/scripts/generate_synthetic_sars.py --count 500 --output data/raw/synthetic_sars.jsonl"
echo "3. Prepare data: python data/scripts/prepare_sar_data.py --input data/raw/synthetic_sars.jsonl --output data/processed/"
echo "4. Start training: python training/train_sar.py --data data/processed/sar_dataset_alpaca --output models/sar-mistral-7b"
echo ""
echo "For more information, see README.md"
