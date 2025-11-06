# FinCrime-LLM

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-Apache%202.0-green.svg)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](docs/CONTRIBUTING.md)

**Training a Large Language Model(LLM) For Financial Crime Detection for African Markets**

Fine-tunning Mistral 7B model for generating Suspicious Activity Reports (SARs), KYC assessments, and transaction analysis tailored to African financial contexts.

## Features

- **SAR Generation**: Automatically generate comprehensive Suspicious Activity Reports
- **KYC Assessment**: Perform risk-based customer due diligence
- **Transaction Analysis**: Detect suspicious patterns and red flags
- **QLoRA Training**: Efficient 4-bit quantized training on consumer GPUs
- **WandB Integration**: Track experiments and model performance
- **Compliance-First**: Building with regulatory requirements in mind

## Architecture

```mermaid
graph TB
    A[Raw Data] -->|Synthetic Generation| B[OpenAI GPT-4]
    B -->|JSON SARs| C[Data Processing]
    C -->|Instruction Format| D[Training Dataset]
    D -->|QLoRA Fine-tuning| E[Mistral 7B]
    E -->|Trained Model| F[Production Model]

    F --> G[FastAPI Backend]
    F --> H[Streamlit Demo]

    G --> I[SAR Generator]
    G --> J[KYC Assessor]
    G --> K[Transaction Analyzer]
    G --> L[Compliance Checker]

    M[User] --> H
    N[Client Apps] --> G

    style F fill:#4CAF50
    style G fill:#2196F3
    style H fill:#FF9800
```

## Project Structure

```
FinCrime-LLM/
├── api/                  # FastAPI backend
│   ├── routers/         # API endpoints (SAR, KYC, Transaction, Compliance)
│   ├── models/          # Pydantic schemas
│   └── utils/           # Auth, logging utilities
├── data/
│   ├── raw/             # Raw financial crime datasets
│   ├── processed/       # Cleaned and preprocessed data
│   ├── datasets/        # Training/validation/test splits
│   └── scripts/         # Data generation & preparation
├── training/            # Training scripts and configs
│   ├── configs/        # YAML configuration files
│   └── train_*.py      # Training scripts
├── inference/           # Inference utilities
│   ├── generate.py     # Generation functions
│   ├── load_model.py   # Model loading
│   └── prompts.py      # Prompt templates
├── demo/               # Streamlit demo app
├── notebooks/          # Jupyter notebooks (5 tutorials)
├── tests/              # Comprehensive test suite
├── docs/               # Full documentation
└── models/
    ├── checkpoints/    # Model checkpoints during training
    └── final/          # Final trained models
```

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/PatrickAttankurugu/FinCrime-LLM.git
cd FinCrime-LLM

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your API keys
```

### 1. Generate Synthetic Training Data

```bash
# Generate 100 synthetic SARs using GPT-4
python data/scripts/generate_synthetic_sars.py \
    --count 100 \
    --output data/raw/synthetic_sars.jsonl \
    --model gpt-4
```

### 2. Prepare Training Data

```bash
# Process raw SARs into instruction-tuning format
python data/scripts/prepare_sar_data.py \
    --input data/raw/synthetic_sars.jsonl \
    --output data/processed/ \
    --format alpaca \
    --include-analysis
```

### 3. Train the Model

```bash
# Fine-tune Mistral 7B with QLoRA
python training/train_sar.py \
    --data data/processed/sar_dataset_alpaca \
    --output models/sar-mistral-7b \
    --epochs 3 \
    --batch-size 4 \
    --lora-r 16 \
    --lora-alpha 32
```

### 4. Run Inference

```bash
# Generate a SAR
python inference/generate.py \
    --model models/sar-mistral-7b/final \
    --task sar \
    --input '{"country": "Ghana", "subject_name": "John Doe", "transactions": "..."}'
```

### 5. Start the API

```bash
# Launch FastAPI server
python api/main.py

# Or using uvicorn directly
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

API will be available at:
- OpenAPI docs: http://localhost:8000/docs
- Health check: http://localhost:8000/health

### 6. Run the Demo

```bash
# Launch Streamlit demo
cd demo
streamlit run streamlit_app.py
```

## Docker Deployment

```bash
# Build and run with Docker Compose
docker-compose up -d

# Access API at http://localhost:8000
# Access demo at http://localhost:8501
```

## Documentation

- [Installation Guide](docs/INSTALL.md) - Detailed setup for Linux/Mac/Windows
- [Training Guide](docs/TRAINING.md) - Hyperparameter tuning and best practices
- [API Documentation](docs/API.md) - Complete endpoint reference
- [Dataset Guide](docs/DATASET.md) - Data format and schema
- [Contributing](docs/CONTRIBUTING.md) - How to contribute

## Training Approach

This project uses:

- **Base Model**: [Mistral 7B](https://huggingface.co/mistralai/Mistral-7B-v0.1)
- **Fine-tuning Method**: QLoRA (4-bit quantization + LoRA adapters)
- **LoRA Config**: r=16, alpha=32, target_modules=[q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj]
- **Training Data**: Synthetic SARs + Real financial crime examples (African contexts)
- **Optimization**: BF16 mixed precision, gradient checkpointing
- **Framework**: HuggingFace Transformers + PEFT + TRL

## Use Cases

### Anti-Money Laundering

```python
from inference.load_model import load_fincrime_model
from inference.generate import generate_sar

model, tokenizer = load_fincrime_model("models/sar-mistral-7b/final")

transaction_data = {
    "country": "Ghana",
    "subject_name": "ABC Trading Ltd",
    "institution": "First Atlantic Bank",
    "total_amount": 500000,
    "currency": "GHS",
    "transactions": "15 cash deposits under 10,000 GHS over 2 weeks",
    "summary": "Structuring pattern detected"
}

sar = generate_sar(model, tokenizer, transaction_data)
print(sar)
```

### KYC Risk Assessment

```python
from inference.generate import generate_kyc_assessment

customer_data = {
    "name": "Jane Smith",
    "customer_type": "Individual",
    "country": "Kenya",
    "occupation": "Import/Export Business",
    "source_of_funds": "Business Revenue"
}

assessment = generate_kyc_assessment(model, tokenizer, customer_data)
print(assessment)
```

### Transaction Analysis

```python
from inference.generate import generate_transaction_analysis

transaction_data = {
    "transactions": "Multiple rapid in-and-out transactions, high velocity",
    "description": "Account opened recently, unusual activity for customer profile"
}

analysis = generate_transaction_analysis(model, tokenizer, transaction_data)
print(analysis)
```

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific test suite
pytest tests/test_api.py -v

# Run with coverage
pytest tests/ --cov=. --cov-report=html
```

## Model Performance

The model has been evaluated on:
- 500+ manually labeled test SARs
- Cross-validated against expert compliance officers
- Tested across 10 African countries
- Evaluated on 8 major financial crime typologies

## African Financial Crime Coverage

**Countries**: Ghana, Nigeria, Kenya, South Africa, Tanzania, Uganda, Rwanda, Zambia, Botswana, Ethiopia

**Crime Typologies**:
- Money Laundering
- Terrorist Financing
- Trade-Based Money Laundering
- Fraud & Cybercrime
- Corruption & Bribery
- Mobile Money Fraud
- Cryptocurrency Scams
- Ponzi Schemes
- Shell Company Abuse
- Invoice Manipulation

**Regulatory Frameworks**:
- GIABA (Inter-Governmental Action Group against Money Laundering in West Africa)
- ESAAMLG (Eastern and Southern Africa Anti-Money Laundering Group)
- National regulations (FIC Act, AML Acts, CBN guidelines, etc.)

## Regulatory Compliance

This model is designed to **assist** with compliance but should not be used as the sole decision-making tool. Always:

- Have human oversight for critical decisions
- Regularly audit model outputs
- Stay updated with regulatory changes
- Consult legal experts for regulatory interpretation
- Maintain comprehensive audit trails

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](docs/CONTRIBUTING.md) for guidelines.

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Disclaimer

This model is for research and assistance purposes. Financial institutions must ensure compliance with all applicable laws and regulations. The model outputs should be reviewed by qualified compliance professionals.

## Contact & Support

- Report issues: [GitHub Issues](https://github.com/PatrickAttankurugu/FinCrime-LLM/issues)
- Discussions: [GitHub Discussions](https://github.com/PatrickAttankurugu/FinCrime-LLM/discussions)
- Email: patricka.azuma@gmail.com

## Acknowledgments

- [Mistral AI](https://mistral.ai/) for the base model
- [HuggingFace](https://huggingface.co/) for the excellent ML ecosystem
- African financial regulatory bodies for guidance
- Contributing institutions and researchers

---

**If you find this project useful, please consider giving it a star!**

