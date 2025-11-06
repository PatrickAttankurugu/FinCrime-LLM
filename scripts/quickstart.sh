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

# Create sample data
echo "Creating sample dataset..."
python src/data/prepare_dataset.py --create-sample

# Run data preprocessing
echo "Processing dataset..."
python src/data/prepare_dataset.py --input data/raw --output data/processed

echo "=========================================="
echo "Setup complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Review the generated dataset in data/datasets/"
echo "2. Adjust training config in configs/training_config.yaml"
echo "3. Start training: python src/training/train.py"
echo ""
echo "For more information, see README.md"
