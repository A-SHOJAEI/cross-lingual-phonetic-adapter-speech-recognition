"""Tests for model components."""

import pytest
import torch

from cross_lingual_phonetic_adapter_speech_recognition.models.model import PhoneticAdapterASR
from cross_lingual_phonetic_adapter_speech_recognition.models.components import (
    PhoneticAdapterLayer,
    PhoneticContrastiveLoss,
    CurriculumScheduler,
    AdapterRegularization,
)


class TestPhoneticAdapterLayer:
    """Tests for PhoneticAdapterLayer."""

    def test_initialization(self):
        """Test adapter layer initialization."""
        adapter = PhoneticAdapterLayer(
            hidden_dim=384,
            adapter_dim=64,
            phonetic_dim=128,
            num_languages=8,
        )
        assert adapter.hidden_dim == 384
        assert adapter.adapter_dim == 64
        assert len(adapter.language_adapters) == 8

    def test_forward(self):
        """Test forward pass."""
        adapter = PhoneticAdapterLayer(
            hidden_dim=384,
            adapter_dim=64,
            phonetic_dim=128,
            num_languages=8,
        )

        batch_size = 2
        seq_len = 100
        hidden_states = torch.randn(batch_size, seq_len, 384)
        phonetic_embedding = torch.randn(batch_size, 128)
        language_id = 0

        output = adapter(hidden_states, phonetic_embedding, language_id)

        assert output.shape == hidden_states.shape
        assert not torch.isnan(output).any()

    def test_different_languages(self):
        """Test adapter with different languages."""
        adapter = PhoneticAdapterLayer(
            hidden_dim=384,
            adapter_dim=64,
            phonetic_dim=128,
            num_languages=8,
        )

        hidden_states = torch.randn(1, 100, 384)
        phonetic_embedding = torch.randn(1, 128)

        outputs = []
        for lang_id in range(3):
            output = adapter(hidden_states, phonetic_embedding, lang_id)
            outputs.append(output)

        # Different languages should produce different outputs
        assert not torch.allclose(outputs[0], outputs[1])


class TestPhoneticContrastiveLoss:
    """Tests for PhoneticContrastiveLoss."""

    def test_initialization(self):
        """Test loss initialization."""
        loss_fn = PhoneticContrastiveLoss(temperature=0.07)
        assert loss_fn.temperature == 0.07

    def test_forward(self):
        """Test loss computation."""
        loss_fn = PhoneticContrastiveLoss(temperature=0.07)

        batch_size = 4
        phonetic_embeddings = torch.randn(batch_size, 128)
        phonetic_features = torch.randn(batch_size, 50, 24)
        language_ids = torch.tensor([0, 1, 2, 3])

        loss = loss_fn(phonetic_embeddings, phonetic_features, language_ids)

        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0  # Scalar
        assert loss.item() >= 0
        assert not torch.isnan(loss)


class TestCurriculumScheduler:
    """Tests for CurriculumScheduler."""

    def test_initialization(self):
        """Test scheduler initialization."""
        stages = [
            {'name': 'vowels', 'duration_epochs': 5, 'phonetic_complexity': 1},
            {'name': 'consonants', 'duration_epochs': 5, 'phonetic_complexity': 2},
        ]
        scheduler = CurriculumScheduler(stages=stages, total_epochs=10)

        assert len(scheduler.stages) == 2
        assert len(scheduler.epoch_to_stage) == 10

    def test_get_current_stage(self):
        """Test getting current stage."""
        stages = [
            {'name': 'stage1', 'duration_epochs': 5, 'phonetic_complexity': 1},
            {'name': 'stage2', 'duration_epochs': 5, 'phonetic_complexity': 2},
        ]
        scheduler = CurriculumScheduler(stages=stages, total_epochs=10)

        stage_0 = scheduler.get_current_stage(0)
        assert stage_0['name'] == 'stage1'

        stage_6 = scheduler.get_current_stage(6)
        assert stage_6['name'] == 'stage2'

    def test_get_complexity_threshold(self):
        """Test complexity threshold."""
        stages = [
            {'name': 'stage1', 'duration_epochs': 5, 'phonetic_complexity': 1},
            {'name': 'stage2', 'duration_epochs': 5, 'phonetic_complexity': 3},
        ]
        scheduler = CurriculumScheduler(stages=stages, total_epochs=10)

        assert scheduler.get_complexity_threshold(0) == 1
        assert scheduler.get_complexity_threshold(6) == 3

    def test_filter_batch_by_complexity(self):
        """Test batch filtering."""
        stages = [
            {'name': 'stage1', 'duration_epochs': 5, 'phonetic_complexity': 2},
        ]
        scheduler = CurriculumScheduler(stages=stages, total_epochs=5)

        batch = {
            'audio': torch.randn(4, 16000),
            'complexity': torch.tensor([1, 2, 3, 4]),
            'text': ['a', 'b', 'c', 'd'],
        }

        filtered = scheduler.filter_batch_by_complexity(batch, epoch=0)

        # Should keep only complexity <= 2
        assert filtered['audio'].shape[0] <= 2
        assert all(c <= 2 for c in filtered['complexity'].tolist())


class TestAdapterRegularization:
    """Tests for AdapterRegularization."""

    def test_l2_regularization(self):
        """Test L2 regularization."""
        reg = AdapterRegularization(regularization_type='l2')

        params = [torch.randn(10, 10), torch.randn(5, 5)]
        loss = reg(params)

        assert isinstance(loss, torch.Tensor)
        assert loss.item() >= 0

    def test_l1_regularization(self):
        """Test L1 regularization."""
        reg = AdapterRegularization(regularization_type='l1')

        params = [torch.randn(10, 10)]
        loss = reg(params)

        assert isinstance(loss, torch.Tensor)
        assert loss.item() >= 0


class TestPhoneticAdapterASR:
    """Tests for PhoneticAdapterASR model."""

    def test_initialization(self):
        """Test model initialization."""
        model = PhoneticAdapterASR(
            base_model='openai/whisper-tiny',
            num_adapters=4,
            phonetic_dim=128,
            adapter_dim=32,
            freeze_base=True,
        )
        assert model.num_adapters == 4
        assert model.phonetic_dim == 128

    def test_initialization_without_adapters(self):
        """Test model without adapters."""
        model = PhoneticAdapterASR(
            num_adapters=0,
            phonetic_dim=128,
        )
        assert len(model.adapters) == 0

    def test_forward(self, sample_batch):
        """Test forward pass."""
        model = PhoneticAdapterASR(
            num_adapters=2,
            phonetic_dim=128,
            adapter_dim=32,
            adapter_layers=[0, 1],
        )
        model.eval()

        with torch.no_grad():
            outputs = model(
                audio=sample_batch['audio'],
                phonetic_features=sample_batch['phonetic_features'],
                languages=sample_batch['language'],
            )

        assert 'logits' in outputs
        assert 'phonetic_embedding' in outputs
        assert 'hidden_states' in outputs

        assert outputs['phonetic_embedding'].shape[0] == sample_batch['audio'].shape[0]

    def test_get_adapter_parameters(self):
        """Test getting adapter parameters."""
        model = PhoneticAdapterASR(
            num_adapters=2,
            phonetic_dim=128,
            adapter_dim=32,
        )

        params = model.get_adapter_parameters()
        assert len(params) > 0
        assert all(isinstance(p, torch.Tensor) for p in params)

    def test_save_and_load(self, temp_dir):
        """Test model saving and loading."""
        model = PhoneticAdapterASR(
            num_adapters=2,
            phonetic_dim=128,
            adapter_dim=32,
        )

        # Save model
        save_path = temp_dir / "test_model.pt"
        model.save_pretrained(str(save_path))

        assert save_path.exists()

        # Load model
        loaded_model = PhoneticAdapterASR.load_pretrained(str(save_path), device='cpu')

        assert loaded_model.num_adapters == model.num_adapters
        assert loaded_model.phonetic_dim == model.phonetic_dim

    def test_language_mapping(self):
        """Test language to ID mapping."""
        model = PhoneticAdapterASR()

        assert 'en' in model.language_to_id
        assert 'es' in model.language_to_id
        assert model.language_to_id['en'] >= 0
