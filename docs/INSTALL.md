# Installation Guide

## Prerequisites

- Python 3.11 or higher
- CUDA-compatible GPU with 24GB+ VRAM (recommended)
- 50GB+ free disk space
- Git

## Installation Steps

### 1. Clone Repository

```bash
git clone https://github.com/PatrickAttankurugu/FinCrime-LLM.git
cd FinCrime-LLM
```

### 2. Create Virtual Environment

```bash
python3.11 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Configure Environment

```bash
cp .env.example .env
# Edit .env and add your API keys
```

### 5. Verify Installation

```bash
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

## Docker Installation

```bash
# Build image
docker build -t fincrime-llm .

# Run with docker-compose
docker-compose up -d
```

## GPU Setup

For NVIDIA GPUs:
```bash
# Install CUDA 11.8
# Install cuDNN 8.x
# Verify: nvidia-smi
```

## Troubleshooting

See PROJECT_SETUP.md for common issues and solutions.
