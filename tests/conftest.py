"""Pytest configuration and fixtures."""

import pytest
import torch
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


@pytest.fixture
def device() -> str:
    """Get device for testing.

    Returns:
        Device string ('cpu' or 'cuda')
    """
    return "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture
def sample_audio() -> torch.Tensor:
    """Generate sample audio tensor.

    Returns:
        Audio tensor (1, 16000)
    """
    # 1 second of random audio at 16kHz
    return torch.randn(1, 16000)


@pytest.fixture
def sample_batch() -> dict:
    """Generate sample batch for testing.

    Returns:
        Dictionary with batch data
    """
    batch_size = 4
    seq_len = 1500
    phonetic_len = 50
    feature_dim = 24

    return {
        'audio': torch.randn(batch_size, 16000),
        'text': ['hello world', 'test sample', 'speech recognition', 'phonetic adapter'],
        'language': ['en', 'es', 'fr', 'de'],
        'ipa': ['həˈloʊ wɜrld', 'tɛst ˈsæmpəl', 'spiʧ ˌrɛkəɡˈnɪʃən', 'fəˈnɛtɪk əˈdæptər'],
        'phonetic_features': torch.randn(batch_size, phonetic_len, feature_dim),
        'complexity': torch.tensor([1, 2, 3, 2]),
    }


@pytest.fixture
def test_config() -> dict:
    """Get test configuration.

    Returns:
        Test configuration dictionary
    """
    return {
        'model': {
            'base_model': 'openai/whisper-tiny',
            'num_adapters': 4,
            'phonetic_dim': 128,
            'adapter_dim': 32,
            'adapter_layers': [3, 6],
            'freeze_base': True,
        },
        'training': {
            'num_epochs': 2,
            'learning_rate': 0.001,
            'weight_decay': 0.01,
            'max_grad_norm': 1.0,
            'gradient_accumulation_steps': 1,
            'mixed_precision': False,
            'optimizer': 'adamw',
            'scheduler': 'cosine',
            'scheduler_params': {
                'T_max': 2,
                'eta_min': 0.0001,
            },
        },
        'curriculum': {
            'enabled': False,
            'stages': [],
        },
        'loss_weights': {
            'asr_loss': 1.0,
            'phonetic_contrastive_loss': 0.5,
            'adapter_regularization': 0.01,
        },
        'early_stopping': {
            'enabled': False,
            'patience': 2,
            'min_delta': 0.001,
        },
        'checkpoint': {
            'save_dir': './test_models',
            'save_best_only': True,
        },
        'logging': {
            'log_interval': 10,
            'use_mlflow': False,
        },
        'seed': 42,
        'deterministic': True,
    }


@pytest.fixture
def temp_dir(tmp_path) -> Path:
    """Create temporary directory for tests.

    Args:
        tmp_path: Pytest temporary path fixture

    Returns:
        Path to temporary directory
    """
    return tmp_path
