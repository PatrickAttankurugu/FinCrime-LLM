# 🔍 FinCrime-LLM

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-Apache%202.0-green.svg)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](CONTRIBUTING.md)

**AI-Powered Financial Crime Detection for African Markets**

Fine-tuned Mistral 7B model for generating Suspicious Activity Reports (SARs), KYC assessments, and transaction analysis tailored to African financial contexts.

## 🌟 Features

- **🚨 SAR Generation**: Automatically generate comprehensive Suspicious Activity Reports
- **👤 KYC Assessment**: Perform risk-based customer due diligence
- **💰 Transaction Analysis**: Detect suspicious patterns and red flags
- **🌍 African Focus**: Covers Ghana, Nigeria, Kenya, South Africa, and more
- **⚡ Production-Ready**: FastAPI backend, Streamlit demo, Docker support
- **🎯 QLoRA Training**: Efficient 4-bit quantized training on consumer GPUs
- **📊 WandB Integration**: Track experiments and model performance
- **🔒 Compliance-First**: Built with regulatory requirements in mind

## Project Structure

```
FinCrime-LLM/
├── data/
│   ├── raw/              # Raw financial crime datasets
│   ├── processed/        # Cleaned and preprocessed data
│   └── datasets/         # Training/validation/test splits
├── models/
│   ├── checkpoints/      # Model checkpoints during training
│   └── final/           # Final trained models
├── src/
│   ├── training/        # Training scripts and utilities
│   ├── data/            # Data preprocessing and loading
│   ├── evaluation/      # Model evaluation scripts
│   └── utils/           # Helper functions and utilities
├── configs/             # Configuration files for training
├── notebooks/           # Jupyter notebooks for exploration
├── scripts/             # Utility scripts
└── logs/               # Training logs and metrics
```

## Requirements

- Python 3.9+
- CUDA-compatible GPU (recommended: 24GB+ VRAM for full fine-tuning)
- See `requirements.txt` for complete dependency list

## Installation

```bash
# Clone the repository
git clone https://github.com/PatrickAttankurugu/FinCrime-LLM.git
cd FinCrime-LLM

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### 1. Prepare Your Data

Place your financial crime datasets in the `data/raw/` directory. Expected format:

```json
{
  "instruction": "Analyze this transaction for potential money laundering",
  "input": "Transaction details...",
  "output": "Analysis and risk assessment..."
}
```

### 2. Preprocess Data

```bash
python src/data/prepare_dataset.py --input data/raw --output data/processed
```

### 3. Configure Training

Edit `configs/training_config.yaml` to set your training parameters.

### 4. Start Training

```bash
python src/training/train.py --config configs/training_config.yaml
```

### 5. Evaluate Model

```bash
python src/evaluation/evaluate.py --model models/final/fincrime-mistral-7b
```

## Training Approach

This project uses:

- **Base Model**: Mistral 7B (mistralai/Mistral-7B-v0.1)
- **Fine-tuning Method**: LoRA (Low-Rank Adaptation) for efficient training
- **Training Data**: Curated financial crime datasets from African contexts
- **Optimization**: Mixed precision training (FP16/BF16)
- **Framework**: Hugging Face Transformers + PEFT

## Use Cases

### Anti-Money Laundering
```python
from src.inference import FinCrimeLLM

model = FinCrimeLLM.load("models/final/fincrime-mistral-7b")
result = model.analyze(
    "Customer made 15 cash deposits under $10,000 over 2 weeks"
)
print(result.risk_level, result.explanation)
```

### Compliance Checking
```python
result = model.check_compliance(
    transaction_data,
    regulations=["Nigeria_CBN_AML_2022", "Kenya_CMA_Guidelines"]
)
```

## Dataset Guidelines

When preparing financial crime datasets:

1. **Privacy**: Ensure all data is anonymized and compliant with data protection laws
2. **Diversity**: Include examples from multiple African countries and contexts
3. **Balance**: Maintain balance between positive and negative examples
4. **Quality**: Verify accuracy of labels and annotations
5. **Context**: Include relevant local regulations and practices

## Regulatory Compliance

This model is designed to assist with compliance but should not be used as the sole decision-making tool. Always:

- Have human oversight for critical decisions
- Regularly audit model outputs
- Stay updated with regulatory changes
- Consult legal experts for regulatory interpretation

## Contributing

We welcome contributions! Please see `CONTRIBUTING.md` for guidelines.

## License

[To be determined - Add appropriate license]

## Disclaimer

This model is for research and assistance purposes. Financial institutions must ensure compliance with all applicable laws and regulations. The model outputs should be reviewed by qualified compliance professionals.

## Contact

For questions or collaboration opportunities, please open an issue or contact the maintainers.

## Acknowledgments

- Mistral AI for the base model
- African financial regulatory bodies for guidance
- Contributing institutions and researchers

---

**Note**: This project is under active development. Features and documentation will be updated regularly.
