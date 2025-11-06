# Training Guide

## Quick Start

```bash
# Generate training data
python data/scripts/generate_synthetic_sars.py --count 500 --output data/raw/sars.jsonl

# Prepare for training
python data/scripts/prepare_sar_data.py --input data/raw/sars.jsonl --output data/processed/

# Train model
python training/train_sar.py --data data/processed/sar_dataset_alpaca --output models/sar-mistral-7b
```

## Training Configuration

Edit `training/configs/lora_config.yaml` and `training/configs/training_args.yaml`.

Key parameters:
- `lora_r`: 16 (higher = more parameters)
- `lora_alpha`: 32
- `batch_size`: 4 (adjust based on GPU)
- `learning_rate`: 2e-4
- `epochs`: 3

## Monitoring

Training metrics are logged to:
- WandB (if configured)
- TensorBoard: `tensorboard --logdir models/sar-mistral-7b/`
- Local logs: `training.log`

See PROJECT_SETUP.md for detailed instructions.
