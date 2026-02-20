# Setup and Installation Guide

## Prerequisites

- Python 3.8 or higher
- pip package manager
- (Optional) CUDA-capable GPU for faster training

## Installation

### 1. Create Virtual Environment (Recommended)

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

This will install all required packages including:
- PyTorch and torchaudio
- Transformers (Whisper model)
- Speech processing libraries (librosa, soundfile)
- Phonetic processing (epitran, panphon)
- Evaluation metrics (jiwer)
- Training utilities (MLflow, tensorboard)

### 3. Verify Installation

Run the tests to ensure everything is installed correctly:

```bash
python -m pytest tests/ -v
```

Expected output: All tests should pass.

## Quick Test Run

Test the training pipeline with synthetic data (no download required):

```bash
python scripts/train.py --debug
```

This will:
- Use synthetic data (no real dataset download)
- Run for a few iterations to verify the pipeline works
- Create a test model in `./models/` directory

## Data Preparation

### Download CommonVoice Dataset

For real experiments, download the CommonVoice dataset:

```bash
# Download all languages (may take several hours and require significant disk space)
python scripts/download_data.py --languages en es fr de ar sw ta bn --output-dir ./data

# For quick testing, limit samples:
python scripts/download_data.py --max-samples 1000 --output-dir ./data
```

The script will download data to `./data/` directory and log progress.

## Running Tests

### All Tests

```bash
python -m pytest tests/ -v
```

### Specific Test Files

```bash
# Test model components
python -m pytest tests/test_model.py -v

# Test data loading
python -m pytest tests/test_data.py -v

# Test training
python -m pytest tests/test_training.py -v
```

### With Coverage Report

```bash
python -m pytest tests/ --cov=cross_lingual_phonetic_adapter_speech_recognition --cov-report=html
```

View coverage report: `open htmlcov/index.html`

## Training

### Full Training

```bash
python scripts/train.py --config configs/default.yaml
```

### Ablation Study (Baseline without Adapters)

```bash
python scripts/train.py --config configs/ablation.yaml
```

### Custom Configuration

Edit `configs/default.yaml` or create a new config file, then:

```bash
python scripts/train.py --config configs/your_config.yaml
```

## Evaluation

Evaluate a trained model:

```bash
python scripts/evaluate.py \
    --checkpoint models/best_model.pt \
    --config configs/default.yaml \
    --split test
```

Results are saved to `./results/` directory.

## Inference

Run inference on a single audio file:

```bash
python scripts/predict.py \
    --audio path/to/your/audio.wav \
    --language en \
    --checkpoint models/best_model.pt
```

## Troubleshooting

### ImportError: No module named 'torch'

Ensure dependencies are installed:
```bash
pip install -r requirements.txt
```

### CUDA Out of Memory

Reduce batch size in config file:
```yaml
data:
  train_batch_size: 8  # Reduce from 16
  eval_batch_size: 16  # Reduce from 32
```

Or use CPU:
```bash
python scripts/train.py --device cpu
```

### epitran/panphon Not Available

The system will use fallback synthetic phonetic features if these libraries fail to load. For full functionality, ensure they're installed:
```bash
pip install epitran panphon
```

### Tests Failing

1. Ensure all dependencies are installed
2. Check Python version (3.8+)
3. Try running with verbose output: `python -m pytest tests/ -vv`

## Directory Structure After Setup

```
.
├── data/                    # Downloaded CommonVoice data
├── logs/                    # Training logs
├── models/                  # Saved model checkpoints
│   └── best_model.pt       # Best model from training
├── results/                 # Evaluation results
│   ├── evaluation_results_test.json
│   ├── evaluation_results_test.csv
│   └── training_history.json
├── configs/                 # Configuration files
├── scripts/                 # Training/evaluation scripts
├── src/                     # Source code
└── tests/                   # Test suite
```

## Next Steps

1. Download data: `python scripts/download_data.py`
2. Run training: `python scripts/train.py`
3. Evaluate model: `python scripts/evaluate.py`
4. Check results in `./results/` directory
