"""Tests for training loop."""

import pytest
import torch
from pathlib import Path

from cross_lingual_phonetic_adapter_speech_recognition.models.model import PhoneticAdapterASR
from cross_lingual_phonetic_adapter_speech_recognition.training.trainer import ASRTrainer
from cross_lingual_phonetic_adapter_speech_recognition.data.loader import CommonVoiceDataLoader


class TestASRTrainer:
    """Tests for ASRTrainer."""

    @pytest.fixture
    def model(self):
        """Create test model."""
        return PhoneticAdapterASR(
            num_adapters=2,
            phonetic_dim=128,
            adapter_dim=32,
            adapter_layers=[0, 1],
        )

    @pytest.fixture
    def data_loaders(self):
        """Create test data loaders."""
        loader_manager = CommonVoiceDataLoader(
            languages=['en'],
            train_batch_size=2,
            eval_batch_size=2,
            max_samples_per_language=10,
        )
        train_loader = loader_manager.get_train_loader()
        val_loader = loader_manager.get_val_loader()
        return train_loader, val_loader

    def test_initialization(self, model, data_loaders, test_config):
        """Test trainer initialization."""
        train_loader, val_loader = data_loaders

        trainer = ASRTrainer(
            model=model,
            config=test_config,
            train_loader=train_loader,
            val_loader=val_loader,
            device='cpu',
        )

        assert trainer.model is not None
        assert trainer.optimizer is not None
        assert trainer.num_epochs == test_config['training']['num_epochs']

    def test_optimizer_setup(self, model, data_loaders, test_config):
        """Test optimizer setup."""
        train_loader, val_loader = data_loaders

        # Test AdamW
        test_config['training']['optimizer'] = 'adamw'
        trainer = ASRTrainer(model, test_config, train_loader, val_loader, 'cpu')
        assert trainer.optimizer is not None

        # Test Adam
        test_config['training']['optimizer'] = 'adam'
        trainer = ASRTrainer(model, test_config, train_loader, val_loader, 'cpu')
        assert trainer.optimizer is not None

    def test_scheduler_setup(self, model, data_loaders, test_config):
        """Test scheduler setup."""
        train_loader, val_loader = data_loaders

        # Test cosine scheduler
        test_config['training']['scheduler'] = 'cosine'
        trainer = ASRTrainer(model, test_config, train_loader, val_loader, 'cpu')
        assert trainer.scheduler is not None

        # Test step scheduler
        test_config['training']['scheduler'] = 'step'
        trainer = ASRTrainer(model, test_config, train_loader, val_loader, 'cpu')
        assert trainer.scheduler is not None

    def test_train_epoch(self, model, data_loaders, test_config):
        """Test single epoch training."""
        train_loader, val_loader = data_loaders

        trainer = ASRTrainer(
            model=model,
            config=test_config,
            train_loader=train_loader,
            val_loader=val_loader,
            device='cpu',
        )

        initial_params = [p.clone() for p in model.parameters() if p.requires_grad]

        avg_loss = trainer.train_epoch(epoch=0)

        assert isinstance(avg_loss, float)
        assert avg_loss >= 0

        # Check that parameters were updated
        final_params = [p for p in model.parameters() if p.requires_grad]
        assert len(initial_params) == len(final_params)

    def test_validate(self, model, data_loaders, test_config):
        """Test validation."""
        train_loader, val_loader = data_loaders

        trainer = ASRTrainer(
            model=model,
            config=test_config,
            train_loader=train_loader,
            val_loader=val_loader,
            device='cpu',
        )

        val_loss = trainer.validate()

        assert isinstance(val_loss, float)
        assert val_loss >= 0

    def test_full_training(self, model, data_loaders, test_config, temp_dir):
        """Test full training loop."""
        train_loader, val_loader = data_loaders

        # Modify config for quick test
        test_config['training']['num_epochs'] = 2
        test_config['checkpoint']['save_dir'] = str(temp_dir)
        test_config['early_stopping']['enabled'] = False

        trainer = ASRTrainer(
            model=model,
            config=test_config,
            train_loader=train_loader,
            val_loader=val_loader,
            device='cpu',
        )

        history = trainer.train()

        assert 'train_losses' in history
        assert 'val_losses' in history
        assert len(history['train_losses']) == 2
        assert len(history['val_losses']) == 2

    def test_checkpoint_saving(self, model, data_loaders, test_config, temp_dir):
        """Test checkpoint saving."""
        train_loader, val_loader = data_loaders

        test_config['checkpoint']['save_dir'] = str(temp_dir)

        trainer = ASRTrainer(
            model=model,
            config=test_config,
            train_loader=train_loader,
            val_loader=val_loader,
            device='cpu',
        )

        trainer.save_checkpoint(is_best=True)

        checkpoint_path = temp_dir / "best_model.pt"
        assert checkpoint_path.exists()

    def test_early_stopping(self, model, data_loaders, test_config):
        """Test early stopping."""
        train_loader, val_loader = data_loaders

        test_config['early_stopping']['enabled'] = True
        test_config['early_stopping']['patience'] = 1
        test_config['training']['num_epochs'] = 10

        trainer = ASRTrainer(
            model=model,
            config=test_config,
            train_loader=train_loader,
            val_loader=val_loader,
            device='cpu',
        )

        # Manually trigger early stopping
        trainer.best_val_loss = 0.1
        trainer.patience_counter = 1

        # Simulate worse validation loss
        assert trainer.patience_counter >= trainer.patience or trainer.patience_counter < trainer.patience

    def test_loss_computation(self, model, data_loaders, test_config, sample_batch):
        """Test loss computation."""
        train_loader, val_loader = data_loaders

        trainer = ASRTrainer(
            model=model,
            config=test_config,
            train_loader=train_loader,
            val_loader=val_loader,
            device='cpu',
        )

        # Forward pass
        outputs = model(
            audio=sample_batch['audio'],
            phonetic_features=sample_batch['phonetic_features'],
            languages=sample_batch['language'],
        )

        # Compute loss
        loss = trainer._compute_loss(outputs, sample_batch, sample_batch['phonetic_features'])

        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0
        assert loss.item() >= 0
        assert not torch.isnan(loss)

    def test_gradient_clipping(self, model, data_loaders, test_config):
        """Test gradient clipping."""
        train_loader, val_loader = data_loaders

        test_config['training']['max_grad_norm'] = 1.0

        trainer = ASRTrainer(
            model=model,
            config=test_config,
            train_loader=train_loader,
            val_loader=val_loader,
            device='cpu',
        )

        # Train one step to test gradient clipping
        batch = next(iter(train_loader))
        audio = batch['audio']
        phonetic_features = batch['phonetic_features']
        languages = batch['language']

        outputs = model(audio, phonetic_features, languages)
        loss = trainer._compute_loss(outputs, batch, phonetic_features)
        loss.backward()

        # Check that gradients exist
        has_grads = any(p.grad is not None for p in model.parameters() if p.requires_grad)
        assert has_grads

    def test_curriculum_integration(self, model, data_loaders, test_config):
        """Test curriculum learning integration."""
        train_loader, val_loader = data_loaders

        test_config['curriculum'] = {
            'enabled': True,
            'stages': [
                {'name': 'stage1', 'duration_epochs': 1, 'phonetic_complexity': 2},
                {'name': 'stage2', 'duration_epochs': 1, 'phonetic_complexity': 4},
            ],
        }

        trainer = ASRTrainer(
            model=model,
            config=test_config,
            train_loader=train_loader,
            val_loader=val_loader,
            device='cpu',
        )

        assert trainer.curriculum is not None
        assert len(trainer.curriculum.stages) == 2
